"""Winnowed character fingerprints and session-wide (chunked / joined / volume) leak accounting."""

from __future__ import annotations

import random
import sys

import pytest

from conftest import SECRET_CODE
from sealbox.compute import Prepared
from sealbox.fingerprint import (
    CHAR_K,
    CHAR_W,
    CharIndex,
    FingerprintIndex,
    rolling_hashes,
    squash,
    winnow,
)
from sealbox.gate import MAX_FRAGMENTS, EffectivePolicy, EgressGate, SessionLeakState, _estimate
from sealbox.policy import DatasetPolicy, LeakConfig

SQ = squash(SECRET_CODE)


# ------------------------------------------------------------------------------------------ primitives
def test_rolling_hashes_match_direct_hashing() -> None:
    hs = rolling_hashes("abcdefghijklmnop", 4)
    assert len(hs) == 13
    # same window -> same hash wherever it occurs; different windows differ
    again = rolling_hashes("xxabcdyy", 4)
    assert again[2] == hs[0]
    assert len(set(hs)) == 13
    assert rolling_hashes("abc", 4) == []
    assert len(rolling_hashes("abcd", 4)) == 1


def test_winnow_density_and_guarantee() -> None:
    rng = random.Random(1)
    hs = [rng.getrandbits(64) for _ in range(5000)]
    sel = winnow(hs, 6)
    # expected density 2/(w+1) ~ 0.286
    assert 0.2 < len(sel) / len(hs) < 0.36
    # guarantee: every window of w consecutive hashes contains a selected value
    chosen = set(sel)
    for i in range(len(hs) - 6 + 1):
        assert chosen & set(hs[i : i + 6])


def test_winnow_edge_cases() -> None:
    assert winnow([], 4) == []
    assert winnow([5, 3, 9], 4) == [3]  # shorter than a window: its minimum
    assert winnow([5, 3, 9], 1) == [5, 3, 9]
    assert winnow([4, 4, 4, 4, 4], 2) == [4, 4, 4, 4]  # ties: rightmost, each position once
    assert winnow([1, 2, 3, 4], 2) == [1, 2, 3]


def test_char_index_hits_spaced_and_recased() -> None:
    idx = CharIndex()
    idx.add(SECRET_CODE)
    assert len(idx) > 0 and idx.n_chars == len(SQ)
    snippet = "decay = 0.9731 ** event.age_days"
    assert idx.hits(snippet)
    assert idx.hits(" ".join(snippet.upper())) == idx.hits(snippet)
    assert idx.hits("the quarterly revenue grew in every region") == set()
    assert idx.hits("") == set()
    assert "not-an-int" not in idx
    assert CharIndex().hits(snippet) == set()


def test_char_index_guarantee_on_every_15_char_substring() -> None:
    idx = CharIndex()
    idx.add(SECRET_CODE)
    n = CHAR_K + CHAR_W - 1
    for i in range(len(SQ) - n + 1):
        assert idx.hits_squashed(SQ[i : i + n]), i


def test_char_index_incremental_adds_and_dedup() -> None:
    idx = CharIndex()
    idx.add(SECRET_CODE)
    n1 = len(idx)
    idx.add(SECRET_CODE)  # same content again: no growth
    assert len(idx) == n1
    idx.add("an entirely different protected paragraph about ledgers and tolerances")
    assert len(idx) > n1
    assert idx.nbytes == len(idx) * 8


def test_char_index_memory_is_far_below_one_int_per_char() -> None:
    rng = random.Random(0)
    corpus = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz_(){}=+0123456789") for _ in range(200_000))
    idx = CharIndex()
    idx.add(corpus)
    per_char = idx.nbytes / len(corpus)
    assert per_char < 3.0  # ~2/(w+1) * 8 bytes
    old_style = {hash(corpus[i : i + 24]) for i in range(0, 20_000)}  # the previous per-char set, 1/10 scale
    old_per_char = (sys.getsizeof(old_style) + 32 * len(old_style)) / 20_000
    assert old_per_char > 10 * per_char


def test_longest_run() -> None:
    idx = CharIndex()
    idx.add(SECRET_CODE)
    assert idx.longest_run("nothing to see here at all, friend") == 0
    assert idx.longest_run(SQ[:20]) >= 15
    assert idx.longest_run(SQ[:200]) >= 190
    # two separate copied stretches separated by unrelated text are not chained together
    two = SQ[:20] + "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz" + SQ[300:320]
    assert idx.longest_run(two) <= 25
    assert CharIndex().longest_run(SQ) == 0


def test_seam_hits() -> None:
    idx = CharIndex()
    idx.add(SECRET_CODE)
    left, right = SQ[100:112], SQ[112:124]
    assert idx.seam_hits(left, right)  # adjacent pieces join into protected text
    assert idx.seam_hits(right, left) == set()
    assert idx.seam_hits("", right) == set()


def test_fingerprint_index_char_hits_delegates() -> None:
    fp = FingerprintIndex()
    fp.add_text(SECRET_CODE)
    assert fp.char_hits("total += weights[event.kind]") == fp.chars.hits("total += weights[event.kind]")


# ------------------------------------------------------------------------------------------ gate session
def _gate(kind: str = "docs", **leak: object) -> EgressGate:
    fp = FingerprintIndex()
    fp.add_text(SECRET_CODE)
    pol = DatasetPolicy(dataset="d", leak=LeakConfig(**leak))  # type: ignore[arg-type]
    p = Prepared(name="d", kind=kind, policy=pol, meta={}, columns=["path"], rows=[], protected=set(), fingerprints=fp)
    return EgressGate({"d": p})


def _send(g: EgressGate, s: SessionLeakState, value: object) -> bool:
    return g.inspect({"result": value}, ["d"], check_protected=True, session=s).allowed


def test_contiguous_copy_denied_per_call_even_without_session() -> None:
    g = _gate()
    g = _gate(max_shared_shingles=1000)
    r = g.inspect({"result": SECRET_CODE[1:60]}, ["d"], check_protected=True)
    assert not r.allowed and "stretch" in r.reasons[0].detail
    assert g.inspect({"result": SECRET_CODE[1:28]}, ["d"], check_protected=True).allowed


def test_contiguous_threshold_boundary() -> None:
    loose = {"max_shared_shingles": 1000, "minhash_threshold": 1.1, "max_session_protected_chars": 10_000}
    g = _gate(max_contiguous_chars=1000, **loose)
    run = g.prepared["d"].fingerprints.chars.longest_run(SQ[:80])
    s = SessionLeakState()
    assert (
        _gate(max_contiguous_chars=run, **loose)
        .inspect({"result": SQ[:80]}, ["d"], check_protected=True, session=s)
        .allowed
    )
    r = _gate(max_contiguous_chars=run - 1, **loose).inspect({"result": SQ[:80]}, ["d"], check_protected=True)
    assert not r.allowed


def test_scrambled_small_chunks_capped_by_session_estimate() -> None:
    g = _gate(max_session_seam_hits=10_000)
    s = SessionLeakState()
    starts = list(range(0, len(SQ) - 20, 40))  # never adjacent: no seams
    random.Random(3).shuffle(starts)
    released = [i for i in starts if _send(g, s, SQ[i : i + 20])]
    assert 0 < len(released) < len(starts)
    assert sum(20 for _ in released) <= 160 + 40  # about the cap, never the whole file
    assert s.protected_chars["d"] <= 160


def test_seam_join_across_calls_denied() -> None:
    g = _gate()
    s = SessionLeakState()
    assert _send(g, s, SQ[200:212])
    assert _send(g, s, "an unrelated finding about the code")
    assert not _send(g, s, SQ[212:224])  # joins the first fragment (released two calls earlier)
    assert len(s.fragments) == 3  # two pieces + the dict key, deduplicated


def test_seam_join_reverse_order_and_within_one_call() -> None:
    g = _gate()
    s = SessionLeakState()
    assert _send(g, s, SQ[212:224])
    assert not _send(g, s, SQ[200:212])  # earlier piece requested later
    s2 = SessionLeakState()
    assert not _send(g, s2, [SQ[312:324], SQ[300:312]])  # shuffled pieces inside one output


def test_seam_threshold_is_configurable() -> None:
    g = _gate(max_session_seam_hits=10_000, max_session_protected_chars=10_000)
    s = SessionLeakState()
    assert _send(g, s, SQ[200:212]) and _send(g, s, SQ[212:224])
    assert s.seam_hits["d"]


def test_denied_output_does_not_update_state() -> None:
    g = _gate()
    s = SessionLeakState()
    assert not _send(g, s, SECRET_CODE)
    assert s.hits == {} and s.derived_chars == 0 and s.fragments == [] and s.protected_chars == {}


def test_repeating_the_same_snippet_costs_only_once() -> None:
    g = _gate(max_session_derived_fraction=1.0)
    s = SessionLeakState()
    assert _send(g, s, "proprietary_score")
    first = s.protected_chars["d"]
    for _ in range(10):
        assert _send(g, s, "proprietary_score")
    assert s.protected_chars["d"] == first


def test_derived_volume_cap_relative_to_corpus() -> None:
    g = _gate(max_session_derived_fraction=0.5)
    s = SessionLeakState()
    cap = max(256, int(0.5 * len(SQ)))
    sent = 0
    while _send(g, s, [7] * 20):  # numbers carry no fingerprints; only the volume cap stops them
        sent += 1
        assert sent < 1000
    assert s.derived_chars <= cap
    r = g.inspect({"result": [7] * 20}, ["d"], check_protected=True, session=s)
    assert r.reasons[0].rule == "derived_volume"


def test_derived_volume_absolute_cap_and_floor() -> None:
    g = _gate(max_session_derived_chars=300, max_session_derived_fraction=1.0)
    s = SessionLeakState()
    while _send(g, s, [1] * 10):
        pass
    assert 250 < s.derived_chars <= 300
    tiny = _gate(max_session_derived_fraction=0.0001)  # corpus-relative cap below the 256-byte floor
    s2 = SessionLeakState()
    while _send(tiny, s2, [1] * 10):
        pass
    assert 200 < s2.derived_chars <= 256


def test_volume_not_counted_for_sql_or_tables_without_content() -> None:
    g = _gate()
    s = SessionLeakState()
    g.inspect({"result": [1] * 50}, ["d"], check_protected=False, session=s)
    assert s.derived_chars == 0
    plain = DatasetPolicy(dataset="t")
    p = Prepared(name="t", kind="table", policy=plain, meta={}, columns=["a"], rows=[], protected=set())
    EgressGate({"t": p}).inspect({"result": [1] * 50}, ["t"], check_protected=True, session=s)
    assert s.derived_chars == 0


def test_fragments_are_bounded() -> None:
    s = SessionLeakState()
    s.remember([f"fragment number {i}" for i in range(MAX_FRAGMENTS + 50)], 9)
    assert len(s.fragments) == MAX_FRAGMENTS
    assert s.fragments[-1][1].endswith(str(MAX_FRAGMENTS + 49))
    s.remember(["   "], 9)  # whitespace-only strings are not fragments
    assert len(s.fragments) == MAX_FRAGMENTS


def test_estimate() -> None:
    idx = CharIndex()
    idx.add(SECRET_CODE)
    assert _estimate(idx, ["x"], ["x"], set()) == 0.0
    piece = SQ[:40]
    fps = idx.hits(piece)
    est = _estimate(idx, [piece], [piece], fps)
    assert est == pytest.approx(len(fps) * idx.chars_per_fingerprint + (CHAR_K - 1))
    assert 25 <= est <= 60


def test_effective_policy_takes_strictest_session_caps() -> None:
    a = DatasetPolicy(dataset="a")
    b = DatasetPolicy(dataset="b")
    b.leak.max_contiguous_chars = 20
    b.leak.max_session_protected_chars = 50
    a.leak.max_session_seam_hits = 2
    b.leak.max_session_seam_hits = 1
    a.leak.max_session_derived_chars = 100
    b.leak.max_session_derived_fraction = 0.1
    eff = EffectivePolicy.combine([a, b])
    assert eff.leak.max_contiguous_chars == 20
    assert eff.leak.max_session_protected_chars == 50
    assert eff.leak.max_session_seam_hits == 1
    assert eff.leak.max_session_derived_chars == 100
    assert eff.leak.max_session_derived_fraction == 0.1


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_contiguous_chars", -1),
        ("max_session_protected_chars", -1),
        ("max_session_seam_hits", -1),
        ("max_session_derived_chars", -1),
        ("max_session_derived_fraction", 0.0),
        ("max_session_derived_fraction", 1.5),
    ],
)
def test_policy_validates_session_caps(field: str, value: float) -> None:
    from sealbox.errors import PolicyError

    pol = DatasetPolicy(dataset="x")
    setattr(pol.leak, field, value)
    with pytest.raises(PolicyError):
        pol.validate()
