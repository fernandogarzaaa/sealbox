"""Hash-chained, append-only audit log.

Each JSONL entry carries ``prev`` (the previous entry's MAC) and ``mac`` =
HMAC-SHA256(audit_key, prev || canonical_json(entry)). Editing, deleting or reordering any
entry breaks the chain; forging a valid chain requires the master key.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


class AuditLog:
    def __init__(self, path: Path, key: bytes) -> None:
        self.path = path
        self._key = key
        self._lock = threading.Lock()
        self._prev, self._seq = self._tail()

    def _tail(self) -> tuple[str, int]:
        if not self.path.exists():
            return GENESIS, 0
        prev, seq = GENESIS, 0
        with self.path.open("rb") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    prev, seq = rec["mac"], rec["seq"]
        return prev, seq

    def _mac(self, prev: str, body: dict[str, Any]) -> str:
        return hmac.new(self._key, prev.encode() + _canon(body), hashlib.sha256).hexdigest()

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            body = {"seq": self._seq + 1, "ts": round(time.time(), 3), "event": event, **fields}
            rec = {**body, "prev": self._prev, "mac": self._mac(self._prev, body)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "ab") as fh:
                fh.write(_canon(rec) + b"\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._prev, self._seq = rec["mac"], body["seq"]
            return rec


@dataclass
class VerifyResult:
    ok: bool
    entries: int
    error: str = ""
    bad_seq: int | None = None


def verify(path: Path, key: bytes) -> VerifyResult:
    if not path.exists():
        return VerifyResult(True, 0)
    prev, n = GENESIS, 0
    with path.open("rb") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                mac = rec.pop("mac")
                rec_prev = rec.pop("prev")
            except (ValueError, KeyError):
                return VerifyResult(False, n, f"line {lineno}: malformed entry", lineno)
            if rec_prev != prev:
                return VerifyResult(
                    False, n, f"line {lineno}: chain broken (entry removed or reordered)", rec.get("seq")
                )
            if rec.get("seq") != n + 1:
                return VerifyResult(False, n, f"line {lineno}: sequence gap", rec.get("seq"))
            expect = hmac.new(key, prev.encode() + _canon(rec), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expect, mac):
                return VerifyResult(False, n, f"line {lineno}: MAC mismatch (entry modified)", rec.get("seq"))
            prev, n = mac, n + 1
    return VerifyResult(True, n)
