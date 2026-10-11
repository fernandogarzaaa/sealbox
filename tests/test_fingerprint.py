from __future__ import annotations

import base64
import codecs
import urllib.parse
import zlib

from conftest import SECRET_CODE
from sealbox.fingerprint import FingerprintIndex, expand_encodings, jaccard_est, minhash, shingles, structural, tokens


def _idx() -> FingerprintIndex:
    idx = FingerprintIndex(n=8)
    idx.add_text(SECRET_CODE)
    return idx


def test_tokens_and_structural() -> None:
    toks = tokens('x = foo(1.5, "s")  # Hi')
    assert toks == ["x", "=", "foo", "(", "1.5", ",", '"s"', ")", "#", "hi"]
    assert structural(toks) == ["I", "=", "I", "(", "N", ",", "S", ")", "#", "I"]
    assert structural(["def", "return", "_x"]) == ["def", "return", "I"]


def test_shingles_and_minhash() -> None:
    assert shingles(["a"] * 3, 4) == set()
    assert len(shingles(list("abcdef"), 3)) == 4
    s = shingles(list("abcdefghij"), 3)
    assert jaccard_est(minhash(s), minhash(s)) == 1.0
    assert minhash(set()) == minhash(set())


def test_verbatim_and_reformatted_copy() -> None:
    idx = _idx()
    assert len(idx) > 50
    assert idx.shared_exact(SECRET_CODE) > 50
    snippet = "decay = 0.9731 ** event.age_days\n        total += weights[event.kind] * decay * event.amount"
    assert idx.shared_exact(snippet) >= 3
    assert idx.shared_exact(snippet.upper().replace(" ", "  ")) >= 3  # case/whitespace changes
    assert idx.shared_exact("the average salary in engineering is higher than in sales this year") == 0


def test_renamed_copy_caught_structurally() -> None:
    idx = _idx()
    import re

    names: dict[str, str] = {}
    keep = {"def", "for", "in", "if", "and", "return", "class", "self", "round", "max", "len", "sum", "abs"}

    def rename(m: re.Match[str]) -> str:
        w = m.group(0)
        return w if w in keep else names.setdefault(w, f"v{len(names)}")

    renamed = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", rename, SECRET_CODE.replace("0.9731", "0.95"))
    assert idx.shared_exact(renamed) <= 2  # every identifier renamed: exact fingerprints miss it
    assert idx.max_similarity(renamed) >= 0.55
    benign = "def summarize(rows):\n    return {'count': len(rows), 'mean': sum(r['x'] for r in rows) / len(rows)}"
    assert idx.max_similarity(benign) < 0.55


def test_expand_encodings() -> None:
    secret = "decay = 0.9731 ** event.age_days and total += weights"
    b64 = base64.b64encode(secret.encode()).decode()
    hx = secret.encode().hex()
    pct = urllib.parse.quote(secret, safe="")
    esc = "".join(f"\\x{b:02x}" for b in secret.encode())
    zb = base64.b64encode(zlib.compress(secret.encode())).decode()
    for enc in (b64, hx, pct, esc, zb):
        d = expand_encodings(f"result: {enc}")
        assert any(secret in t for t in d.texts), enc[:20]
    d = expand_encodings("prefix " + base64.b64encode(base64.b64encode(secret.encode())).decode())
    assert any(secret in t for t in d.texts)  # double encoding
    assert any(secret in t for t in expand_encodings(codecs.encode(secret, "rot13")).texts)
    assert any(secret in t for t in expand_encodings(secret[::-1]).texts)


def test_opaque_blobs() -> None:
    import os

    blob = base64.b64encode(os.urandom(96)).decode()
    assert expand_encodings(blob).opaque_runs
    assert expand_encodings(os.urandom(48).hex()).opaque_runs
    assert not expand_encodings(base64.b64encode(os.urandom(12)).decode()).opaque_runs  # short
    assert not expand_encodings("a_very_long_snake_case_identifier_name_that_goes_on_and_on_forever_ok").opaque_runs
    assert not expand_encodings("12345678901234567890123").opaque_runs
