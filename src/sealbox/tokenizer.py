"""Deterministic, reversible-by-owner tokenization.

``<PERSON_3f9a2c1b>``-style tokens are ``HMAC-SHA256(token_key, column_label || value)``
truncated. Tokens are stable within a vault (joins and GROUP BY keep working) but cannot
be inverted without the master key. Reversal happens only on the owner's side
(``sealbox detokenize``), never through an MCP tool.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Iterable

TOKEN_RE = re.compile(r"<([A-Z][A-Z0-9_]{0,23})_([0-9a-f]{8,16})>")

_LABELS = (
    (re.compile(r"e_?mail", re.I), "EMAIL"),
    (re.compile(r"phone|mobile", re.I), "PHONE"),
    (re.compile(r"address|street", re.I), "ADDRESS"),
    (re.compile(r"(^|_)ip($|_)", re.I), "IP"),
    (re.compile(r"iban|account", re.I), "ACCOUNT"),
    (re.compile(r"name|person|employee|patient|customer|user|client|member", re.I), "PERSON"),
)


def label_for(column: str) -> str:
    for pattern, label in _LABELS:
        if pattern.search(column):
            return label
    clean = re.sub(r"[^A-Za-z0-9]", "_", column).upper().strip("_") or "VALUE"
    return clean[:24] if clean[0].isalpha() else ("V" + clean)[:24]


class Tokenizer:
    def __init__(self, key: bytes, digest_chars: int = 8) -> None:
        if len(key) < 16:
            raise ValueError("token key too short")
        self._key = key
        self.digest_chars = digest_chars

    def token(self, column: str, value: object) -> str | None:
        if value is None:
            return None
        label = label_for(column)
        msg = f"{label}\x00{value}".encode()
        digest = hmac.new(self._key, msg, hashlib.sha256).hexdigest()[: self.digest_chars]
        return f"<{label}_{digest}>"

    def build_map(self, column: str, values: Iterable[object]) -> dict[str, str]:
        """Token -> original value, for owner-side detokenization."""
        out: dict[str, str] = {}
        for v in values:
            tok = self.token(column, v)
            if tok is not None:
                out[tok] = str(v)
        return out


def detokenize_text(text: str, mapping: dict[str, str]) -> str:
    return TOKEN_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
