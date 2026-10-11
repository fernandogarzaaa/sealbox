from __future__ import annotations

import json
from pathlib import Path

from sealbox.audit import GENESIS, AuditLog, verify
from sealbox.budget import SessionBudget
from sealbox.policy import SessionConfig

KEY = b"a" * 32


def _log(tmp_path: Path, n: int = 4) -> Path:
    log = AuditLog(tmp_path / "audit.jsonl", KEY)
    for i in range(n):
        log.append("tool_call", tool="run_sql", i=i)
    return tmp_path / "audit.jsonl"


def test_chain_ok_and_resume(tmp_path: Path) -> None:
    p = _log(tmp_path)
    assert verify(p, KEY).ok and verify(p, KEY).entries == 4
    first = json.loads(p.read_text().splitlines()[0])
    assert first["prev"] == GENESIS and first["seq"] == 1
    AuditLog(p, KEY).append("again")  # reopen continues the chain
    r = verify(p, KEY)
    assert r.ok and r.entries == 5
    assert verify(tmp_path / "missing.jsonl", KEY).ok


def test_modification_detected(tmp_path: Path) -> None:
    p = _log(tmp_path)
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["i"] = 99
    lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    r = verify(p, KEY)
    assert not r.ok and "MAC mismatch" in r.error and r.bad_seq == 2 and r.entries == 1


def test_deletion_and_reorder_detected(tmp_path: Path) -> None:
    p = _log(tmp_path)
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], *lines[2:]]) + "\n")
    assert "chain broken" in verify(p, KEY).error
    p.write_text("\n".join([lines[1], lines[0], *lines[2:]]) + "\n")
    assert not verify(p, KEY).ok


def test_wrong_key_and_malformed(tmp_path: Path) -> None:
    p = _log(tmp_path)
    assert not verify(p, b"b" * 32).ok
    p.write_text(p.read_text() + "not json\n")
    r = verify(p, KEY)
    assert not r.ok and "malformed" in r.error and r.entries == 4


def test_sequence_gap_detected(tmp_path: Path) -> None:
    p = _log(tmp_path, 1)
    rec = json.loads(p.read_text())
    rec["seq"] = 5
    p.write_text(json.dumps(rec) + "\n")
    assert "sequence gap" in verify(p, KEY).error


def test_budget_flow() -> None:
    b = SessionBudget(SessionConfig(epsilon_budget=2.0, bytes_budget=100, max_queries=3, max_errors=2))
    assert b.admit_query().ok
    assert b.charge(60, 1.0).ok
    assert not b.charge(41).ok  # 60 + 41 > 100
    assert b.charge(40).ok  # exactly 100 allowed
    assert not b.admit_query().ok  # bytes exhausted
    s = b.status()
    assert s["bytes"] == {"released": 100, "budget": 100, "remaining": 0}
    assert s["queries"] == {"used": 1, "limit": 3}


def test_budget_epsilon_and_limits() -> None:
    b = SessionBudget(SessionConfig(epsilon_budget=2.0, bytes_budget=10_000, max_queries=2, max_errors=1))
    assert b.can_spend_epsilon(2.0).ok
    assert not b.can_spend_epsilon(2.01).ok
    assert b.charge(1, 1.5).ok
    assert not b.charge(1, 0.6).ok
    assert b.status()["epsilon"] == {"spent": 1.5, "budget": 2.0, "remaining": 0.5}
    assert b.admit_query().ok and b.admit_query().ok
    d = b.admit_query()
    assert not d.ok and "query limit" in d.reason
    b2 = SessionBudget(SessionConfig(max_errors=1))
    b2.record_error()
    assert "error limit" in b2.admit_query().reason
    b2.record_denial()
    assert b2.status()["denials"] == 1 and b2.status()["errors"] == {"used": 1, "limit": 1}
