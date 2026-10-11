"""Proprietary-content leak detection.

Two complementary fingerprints of everything sensitive in the vault (source files,
documents, long text cells):

* **Exact shingles**: hashed n-grams of normalized tokens. Catches verbatim and
  whitespace/case-reformatted copies, even a few lines long.
* **Character fingerprints** (:class:`CharIndex`): Rabin-Karp rolling hashes of 10-char
  windows of the text with ALL whitespace removed and case folded, thinned by winnowing
  (Schleimer et al., 2003) so only ~2/(w+1) of positions are stored, in a sorted
  ``array('Q')``. Any shared run of at least ``k + w - 1`` (=15) squashed characters is
  guaranteed to share a stored fingerprint. Catches spaced-out / re-wrapped copies and,
  accumulated per session by the gate (including joins across separately released
  fragments), content smuggled out a few characters at a time.
* **Structural MinHash**: identifiers, numbers and string literals collapse to
  placeholders, then windows of tokens are MinHashed. Catches near-verbatim copies
  (renamed variables, changed constants).

Before scanning, outputs are expanded with decodings of embedded base64 / hex /
percent-encoding / ``\\xNN`` escapes, and a run of encoded data that does not decode to
text is reported as an opaque blob (it could be compressed or encrypted smuggling).
"""

from __future__ import annotations

import base64
import binascii
import codecs
import hashlib
import re
import struct
import urllib.parse
import zlib
from array import array
from bisect import bisect_left
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\"[^\"\n]*\"|'[^'\n]*'|[^\sA-Za-z0-9_]")
_KEYWORDS = frozenset(
    (
        "and as assert async await break class continue def del elif else except false finally for "
        "from global if import in is lambda none nonlocal not or pass raise return true try while "
        "with yield fn let mut pub impl struct enum use mod match const static function var return "
        "new this public private protected void int float double string bool select where group by "
        "order"
    ).split()
)
_MASK = (1 << 61) - 1
_NUM_PERM = 64
_WINDOW = 40
_STEP = 10
CHAR_K = 10  # squashed characters per rolling-hash window
CHAR_W = 6  # winnowing window: any shared run of >= CHAR_K + CHAR_W - 1 squashed chars shares a fingerprint
_RK_BASE = 1_000_003
_WS_RE = re.compile(r"\s+")

_B64_RE = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}")
_HEX_RE = re.compile(r"(?:[0-9a-fA-F]{2}[\s:,]?){12,}")
_PCT_RE = re.compile(r"%[0-9a-fA-F]{2}")
_ESC_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2}){6,}")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def structural(toks: list[str]) -> list[str]:
    out = []
    for t in toks:
        if t[0] in "\"'":
            out.append("S")
        elif t[0].isdigit():
            out.append("N")
        elif (t[0].isalpha() or t[0] == "_") and t not in _KEYWORDS:
            out.append("I")
        else:
            out.append(t)
    return out


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def squash(text: str) -> str:
    return _WS_RE.sub("", text).lower()


def _mix(x: int) -> int:
    """splitmix64 finaliser: spreads Rabin-Karp values so winnowing minima are well distributed."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def rolling_hashes(sq: str, k: int = CHAR_K) -> list[int]:
    """Hash of every k-char window of ``sq`` (already squashed), in O(len(sq))."""
    n = len(sq)
    if n < k:
        return []
    top = pow(_RK_BASE, k - 1, _MASK)
    h = 0
    for ch in sq[:k]:
        h = (h * _RK_BASE + ord(ch)) % _MASK
    out = [_mix(h)]
    for i in range(k, n):
        h = ((h - ord(sq[i - k]) * top) * _RK_BASE + ord(sq[i])) % _MASK
        out.append(_mix(h))
    return out


def winnow(hashes: list[int], w: int = CHAR_W) -> list[int]:
    """Robust winnowing: the minimum of every window of ``w`` consecutive hashes (rightmost on ties), each
    selected position reported once. O(n) with a monotonic deque."""
    if w <= 1:
        return list(hashes)
    out: list[int] = []
    dq: deque[int] = deque()  # positions with increasing hash values
    last = -1
    for i, h in enumerate(hashes):
        while dq and hashes[dq[-1]] >= h:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - w:
            dq.popleft()
        if i >= w - 1 and dq[0] != last:
            last = dq[0]
            out.append(hashes[last])
    if hashes and len(hashes) < w:  # shorter than one window: keep its minimum so short texts are indexed
        out.append(min(hashes))
    return out


class CharIndex:
    """Winnowed whitespace/case-insensitive character fingerprints of protected content.

    Memory is ~8 bytes per stored fingerprint and ~2/(w+1) fingerprints per protected character
    (about 2.3 bytes/char at the defaults), instead of one Python int in a set per character. Queries hash
    EVERY window of the output and look each up, so detection keeps the winnowing guarantee."""

    def __init__(self, k: int = CHAR_K, w: int = CHAR_W) -> None:
        self.k = k
        self.w = w
        self._arr: array[int] = array("Q")
        self._pending: list[int] = []
        self.n_chars = 0

    def add(self, text: str) -> None:
        sq = squash(text)
        self.n_chars += len(sq)
        self._pending.extend(winnow(rolling_hashes(sq, self.k), self.w))

    def _frozen(self) -> array[int]:
        if self._pending:
            merged = sorted(set(self._arr).union(self._pending))
            self._arr = array("Q", merged)
            self._pending = []
        return self._arr

    def __len__(self) -> int:
        return len(self._frozen())

    def __contains__(self, h: object) -> bool:
        arr = self._frozen()
        if not isinstance(h, int):
            return False
        i = bisect_left(arr, h)
        return i < len(arr) and arr[i] == h

    @property
    def nbytes(self) -> int:
        return len(self._frozen()) * self._arr.itemsize

    @property
    def chars_per_fingerprint(self) -> float:
        """Expected spacing of winnowed fingerprints (density 2/(w+1)), used to turn a count of matched
        fingerprints into an estimate of protected characters revealed."""
        return (self.w + 1) / 2

    def hits_squashed(self, sq: str) -> set[int]:
        if not self._frozen():
            return set()
        return {h for h in set(rolling_hashes(sq, self.k)) if h in self}

    def hits(self, text: str) -> set[int]:
        """Stored fingerprints that occur anywhere in ``text`` (whitespace/case-insensitive)."""
        return self.hits_squashed(squash(text))

    def longest_run(self, text: str) -> int:
        """Length (squashed chars) of the longest stretch of ``text`` that looks copied from protected content:
        matching windows chained while consecutive matches are at most ``w`` positions apart (the winnowing
        spacing bound inside a genuine copy). 0 when nothing matches."""
        sq = squash(text)
        if not self._frozen():
            return 0
        pos = [i for i, h in enumerate(rolling_hashes(sq, self.k)) if h in self]
        if not pos:
            return 0
        best = 0
        start = prev = pos[0]
        for i in pos[1:]:
            if i - prev > self.w:
                best = max(best, prev - start + self.k)
                start = i
            prev = i
        return max(best, prev - start + self.k)

    def seam_hits(self, left_sq: str, right_sq: str) -> set[int]:
        """Fingerprints of windows that straddle the join ``left + right`` (both squashed): content split
        across two separately released fragments and re-joined offline."""
        k = self.k
        if not left_sq or not right_sq:
            return set()
        return self.hits_squashed(left_sq[-(k - 1) :] + right_sq[: k - 1])


def shingles(toks: list[str], n: int) -> set[int]:
    if len(toks) < n:
        return set()
    return {_h("\x1f".join(toks[i : i + n])) for i in range(len(toks) - n + 1)}


_PERMS = [
    (
        struct.unpack(">Q", hashlib.sha256(f"a{i}".encode()).digest()[:8])[0] % _MASK | 1,
        struct.unpack(">Q", hashlib.sha256(f"b{i}".encode()).digest()[:8])[0] % _MASK,
    )
    for i in range(_NUM_PERM)
]


def minhash(items: set[int]) -> tuple[int, ...]:
    if not items:
        return tuple([_MASK] * _NUM_PERM)
    return tuple(min((a * x + b) & _MASK for x in items) for a, b in _PERMS)


def _windows(toks: list[str]) -> Iterable[list[str]]:
    if len(toks) <= _WINDOW:
        if len(toks) >= 20:
            yield toks
        return
    for i in range(0, len(toks) - _WINDOW + 1, _STEP):
        yield toks[i : i + _WINDOW]


def jaccard_est(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(1 for x, y in zip(a, b, strict=False) if x == y) / len(a)


@dataclass
class FingerprintIndex:
    n: int = 8
    exact: set[int] = field(default_factory=set)
    sigs: list[tuple[int, ...]] = field(default_factory=list)
    chars: CharIndex = field(default_factory=CharIndex)
    # LSH buckets: band -> hash -> signature indexes
    _bands: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    def add_text(self, text: str) -> None:
        toks = tokens(text)
        self.exact |= shingles(toks, self.n)
        self.chars.add(text)
        st = structural(toks)
        for win in _windows(st):
            sig = minhash(shingles(win, 5))
            idx = len(self.sigs)
            self.sigs.append(sig)
            for band in range(16):
                key = (band, hash(sig[band * 4 : band * 4 + 4]))
                self._bands.setdefault(key, []).append(idx)

    def __len__(self) -> int:
        return len(self.exact)

    def shared_exact(self, text: str) -> int:
        return len(shingles(tokens(text), self.n) & self.exact)

    def char_hits(self, text: str) -> set[int]:
        """Winnowed protected character fingerprints that appear in ``text``."""
        return self.chars.hits(text)

    def max_similarity(self, text: str) -> float:
        best = 0.0
        st = structural(tokens(text))
        for win in _windows(st):
            sig = minhash(shingles(win, 5))
            cands: set[int] = set()
            for band in range(16):
                cands.update(self._bands.get((band, hash(sig[band * 4 : band * 4 + 4])), ()))
            for c in cands:
                best = max(best, jaccard_est(sig, self.sigs[c]))
        return best


def _printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    ok = sum(1 for b in data if 32 <= b < 127 or b in (9, 10, 13))
    return ok / len(data)


def _as_text(data: bytes) -> str | None:
    for attempt in (data, _try_decompress(data)):
        if attempt and _printable_ratio(attempt) > 0.9:
            return attempt.decode("utf-8", "replace")
    return None


def _try_decompress(data: bytes) -> bytes | None:
    for wbits in (15, 31, -15):
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            continue
    return None


@dataclass
class Decoded:
    texts: list[str]
    opaque_runs: list[int]  # lengths of encoded runs that did not decode to text


def expand_encodings(text: str, opaque_min_len: int = 64, depth: int = 2) -> Decoded:
    """Return decoded variants of encoded substrings, recursively."""
    texts: list[str] = []
    opaque: list[int] = []

    def visit(s: str, d: int) -> None:
        if d <= 0:
            return
        if len(_PCT_RE.findall(s)) >= 3:
            dec = urllib.parse.unquote(s)
            texts.append(dec)
            visit(dec, d - 1)
        for m in _ESC_RE.finditer(s):
            esc_raw = bytes.fromhex(m.group(0).replace("\\x", ""))
            t = _as_text(esc_raw)
            if t is not None:
                texts.append(t)
                visit(t, d - 1)
            elif len(m.group(0)) >= opaque_min_len:
                opaque.append(len(m.group(0)))
        for m in _HEX_RE.finditer(s):
            run = re.sub(r"[\s:,]", "", m.group(0))
            if len(run) % 2:
                run = run[:-1]
            try:
                hex_raw = bytes.fromhex(run)
            except ValueError:
                continue
            t = _as_text(hex_raw)
            if t is not None:
                texts.append(t)
                visit(t, d - 1)
            elif len(run) >= opaque_min_len:
                opaque.append(len(run))
        for m in _B64_RE.finditer(s):
            run = m.group(0)
            if run.isdigit() or re.fullmatch(r"[0-9a-fA-F]+", run):
                continue  # handled as hex / plain number
            raw: bytes | None = None
            for decoder in (base64.b64decode, base64.urlsafe_b64decode):
                try:
                    raw = decoder(run + "=" * (-len(run) % 4))
                    break
                except (binascii.Error, ValueError):
                    continue
            if raw is None:
                continue
            t = _as_text(raw)
            if t is not None and len(t) >= 8:
                texts.append(t)
                visit(t, d - 1)
            elif len(run) >= opaque_min_len and not _looks_like_identifier(run):
                opaque.append(len(run))
        # ROT13 is cheap to check and a classic smuggling trick
        if d == depth and sum(c.isalpha() for c in s) >= 20:
            texts.append(codecs.decode(s, "rot13"))
        # reversed text
        if d == depth and len(s) >= 40:
            texts.append(s[::-1])

    visit(text, depth)
    return Decoded(texts=texts, opaque_runs=opaque)


def _looks_like_identifier(run: str) -> bool:
    # long snake_case / CamelCase identifiers or paths are not encoded data
    return bool(re.fullmatch(r"[A-Za-z_]+", run)) or run.count("_") + run.count("/") > len(run) / 8
