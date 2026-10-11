from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import yaml

from sealbox import engine as engine_mod
from sealbox.engine import Engine, laplace
from sealbox.gate import INJECTION_PLACEHOLDER
from sealbox.policy import SessionConfig
from sealbox.vault import Vault


def set_policy(vault: Vault, name: str, **changes: object) -> None:
    path = vault.policy_path(name)
    data = yaml.safe_load(path.read_text())
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    path.write_text(yaml.safe_dump(data))


def fresh(vault: Vault, engine: Engine, **session: object) -> Engine:
    cfg = SessionConfig(python_time_quantum_s=0.0, **session)  # type: ignore[arg-type]
    return Engine(vault, engine.vault.keys, cfg)


def test_list_and_describe(engine: Engine) -> None:
    out = engine.list_datasets()
    assert out["status"] == "ok"
    assert [d["name"] for d in out["datasets"]] == ["people", "secret_repo"]
    assert out["datasets"][0]["tools"] == ["run_sql", "run_python"]
    d = engine.describe_dataset("people")
    assert d["n_rows"] == 60 and d["kind"] == "table"
    cols = {c["name"]: c for c in d["columns"]}
    assert cols["ssn"]["tag"] == "deny" and "mean" not in cols["ssn"]
    assert cols["name"]["token_format"] == "<PERSON_xxxxxxxx>"
    assert cols["dept"]["top_values"] == [["Eng", 20], ["Ops", 19], ["Sales", 19]]  # 'Tiny' (2 rows) hidden
    assert "approx_range" in cols["salary"] and "min" not in cols["salary"]
    text = json.dumps(d)
    assert "Person Number" not in text and "-45-" not in text
    assert len(d["synthetic_sample_rows"]) == 5
    assert set(d["synthetic_sample_rows"][0]) == {"person_id", "name", "dept", "salary", "notes"}
    docs = engine.describe_dataset("secret_repo")
    assert docs["n_files"] == 2 and "proprietary_score" not in json.dumps(docs)
    assert engine.describe_dataset("nope")["status"] == "error"


def test_sql_k_anonymity_suppresses_small_groups(vault: Vault, engine: Engine) -> None:
    set_policy(vault, "people", dp={"enabled": False})
    e = fresh(vault, engine)
    out = e.run_sql("SELECT dept, count(*) AS n, avg(salary) AS s FROM people GROUP BY dept ORDER BY dept")
    assert out["status"] == "ok"
    assert [r[0] for r in out["rows"]] == ["Eng", "Ops", "Sales"]
    assert [r[1] for r in out["rows"]] == [20, 19, 19]
    assert out["columns"] == ["dept", "n", "s"]
    assert any("1 group(s) suppressed" in n for n in out["notes"])
    one = e.run_sql("SELECT name, count(*) FROM people GROUP BY name")
    assert one["status"] == "denied" and one["reasons"][0]["rule"] == "k_anonymity"


def test_k_cannot_be_inflated_by_joins(vault: Vault, engine: Engine) -> None:
    set_policy(vault, "people", dp={"enabled": False})
    e = fresh(vault, engine)
    # self-join multiplies rows, but support counts DISTINCT entities
    out = e.run_sql("SELECT a.name, avg(a.salary) FROM people a JOIN people b ON a.dept = b.dept GROUP BY a.name")
    assert out["status"] == "denied" and out["reasons"][0]["rule"] == "k_anonymity"


def test_dp_noise_and_epsilon_accounting(vault: Vault, engine: Engine) -> None:
    e = fresh(vault, engine, epsilon_budget=2.5)
    out = e.run_sql("SELECT dept, count(*), avg(salary) FROM people GROUP BY dept")
    assert out["status"] == "ok" and any("Laplace" in n for n in out["notes"])
    assert e.budget.epsilon_spent == pytest.approx(2.0)
    denied = e.run_sql("SELECT dept, count(*) FROM people GROUP BY dept")
    assert denied["status"] == "denied" and denied["reasons"][0]["rule"] == "privacy_budget"
    assert e.budget_status()["epsilon"]["spent"] == 2.0
    d2 = e.run_sql("SELECT dept, sum(salary) / count(*) FROM people GROUP BY dept")
    assert d2["reasons"][0]["rule"] == "dp_uncalibrated"
    pub = e.run_sql("SELECT dept, max(dept) FROM people GROUP BY dept")  # no sensitive output: no epsilon
    assert pub["status"] == "ok" and e.budget.epsilon_spent == pytest.approx(2.0)


def test_noise_function(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_mod, "laplace", lambda scale: scale)
    b = {("people", "salary"): (-10.0, 100.0)}
    assert engine._noise(None, "count", None, b, 10, 1.0) is None
    assert engine._noise(10, "count", None, b, 10, 0.5) == 12
    assert engine._noise(0, "count_star", None, {}, 10, 1.0) == 1
    assert engine._noise(5.0, "sum", ("people", "salary"), b, 10, 1.0) == 105.0
    assert engine._noise(5.0, "avg", ("people", "salary"), b, 10, 1.0) == pytest.approx(16.0)
    assert engine._noise(5.0, "sum", None, b, 10, 2.0) == 5.5


def test_laplace_distribution() -> None:
    xs = [laplace(2.0) for _ in range(4000)]
    mean = sum(xs) / len(xs)
    mad = sum(abs(x) for x in xs) / len(xs)
    assert abs(mean) < 0.3 and 1.7 < mad < 2.3  # E|X| = scale


def test_sql_rejections_are_denials_and_count_as_errors(engine: Engine) -> None:
    out = engine.run_sql("SELECT name, salary FROM people")
    assert out["status"] == "denied" and out["reasons"][0]["rule"] == "row_level" and "hint" in out
    assert engine.budget.errors == 1 and engine.budget.denials == 1


def test_sql_error_messages_are_sanitized(vault: Vault, engine: Engine) -> None:
    out = engine.run_sql("SELECT dept, count(*) FROM people WHERE CAST(notes AS INTEGER) = 1 GROUP BY dept")
    assert out["status"] == "error"
    assert "routine quarterly" not in out["error"] and "withheld" in out["error"]
    assert engine.budget.data_errors == 1
    out2 = engine.run_sql("SELECT dept, count(*) FROM people WHERE nosuchcol = 'abc' GROUP BY dept")
    assert out2["status"] == "error" and "nosuchcol" in out2["error"]


def test_get_result_paging_and_budget(vault: Vault, engine: Engine) -> None:
    set_policy(vault, "people", k_min=1, dp={"enabled": False})
    e = fresh(vault, engine)
    out = e.run_sql("SELECT person_id, count(*) FROM people GROUP BY person_id ORDER BY 1", max_rows=10)
    assert out["status"] == "ok" and out["total_rows"] == 60 and out["returned_rows"] == 10
    before = e.budget.bytes_released
    page = e.get_result(out["result_id"], offset=50, limit=50)
    assert page["status"] == "ok" and len(page["rows"]) == 10 and page["offset"] == 50
    assert e.budget.bytes_released > before
    assert e.get_result("r_missing")["status"] == "error"
    assert len(e.get_result(out["result_id"], offset=-5, limit=999)["rows"]) == 50


def test_output_row_cap(vault: Vault, engine: Engine) -> None:
    set_policy(vault, "people", k_min=1, max_output_rows=10, dp={"enabled": False})
    e = fresh(vault, engine)
    out = e.run_sql("SELECT person_id, count(*) FROM people GROUP BY person_id")
    assert out["status"] == "denied" and out["reasons"][0]["rule"] == "output_rows"


def test_bytes_budget_exhaustion(vault: Vault, engine: Engine) -> None:
    e = fresh(vault, engine, bytes_budget=900)
    first = e.describe_dataset("people")
    assert first["status"] == "denied" and first["reasons"][0]["rule"] == "leakage_budget"
    probe = fresh(vault, engine)
    probe.run_sql("SELECT count(*) FROM people")
    one = probe.budget.bytes_released
    e2 = fresh(vault, engine, bytes_budget=int(one * 1.5))
    ok = e2.run_sql("SELECT count(*) FROM people")
    assert ok["status"] == "ok"
    nxt = e2.run_sql("SELECT count(*) FROM people")
    assert nxt["status"] == "denied"


def test_query_and_error_limits(vault: Vault, engine: Engine) -> None:
    e = fresh(vault, engine, max_queries=1)
    assert e.run_sql("SELECT count(*) FROM people")["status"] == "ok"
    assert "query limit" in e.run_sql("SELECT count(*) FROM people")["reasons"][0]["detail"]
    e2 = fresh(vault, engine, max_errors=1)
    e2.run_sql("SELECT nonsense FROM people")
    assert "error limit" in e2.run_sql("SELECT count(*) FROM people")["reasons"][0]["detail"]


def test_injection_in_data_is_stripped(vault: Vault, engine: Engine) -> None:
    set_policy(
        vault,
        "people",
        k_min=1,
        columns={
            "person_id": "tokenize",
            "name": "tokenize",
            "dept": "public",
            "salary": "aggregate_only",
            "notes": "public",
            "ssn": "deny",
        },
        dp={"enabled": False},
    )
    e = fresh(vault, engine)
    out = e.run_sql("SELECT notes, count(*) FROM people GROUP BY notes ORDER BY 2")
    assert out["status"] == "ok"
    flat = json.dumps(out["rows"])
    assert "evil.example" not in flat and INJECTION_PLACEHOLDER in flat
    assert any("prompt-injection" in n for n in out["notes"])
    set_policy(vault, "people", injection_action="deny")
    e2 = fresh(vault, engine)
    assert e2.run_sql("SELECT notes, count(*) FROM people GROUP BY notes")["reasons"][0]["rule"] == "prompt_injection"


def test_python_happy_path_and_raw_value_block(engine: Engine) -> None:
    ok = engine.run_python("result = {'n': len(rows('people')), 'depts': sorted({r['dept'] for r in rows('people')})}")
    assert ok["status"] == "ok" and ok["result"] == {"n": 60, "depts": ["Eng", "Ops", "Sales", "Tiny"]}
    assert "netns" in ok["isolation"] or ok.get("warnings")
    leak = engine.run_python("result = [r['salary'] for r in rows('people')][:3]")
    assert leak["status"] == "denied" and leak["reasons"][0]["rule"] == "raw_value"
    leak2 = engine.run_python("r = rows('people')[7]\nresult = f\"salary is {r['salary']}\"")
    assert leak2["status"] == "denied"
    assert "ssn" not in engine.run_python("result = list(rows('people')[0])")["result"]


def test_python_code_leak_blocked_including_encodings(engine: Engine) -> None:
    src = "docs['secret_repo']['scoring.py']"
    for code in (
        f"result = {src}[:600]",
        f"import base64\nresult = base64.b64encode({src}[:400].encode()).decode()",
        f"result = {src}[:400].encode().hex()",
        f"import zlib, base64\nresult = base64.b64encode(zlib.compress({src}.encode())).decode()",
    ):
        out = engine.run_python(code, ["secret_repo"])
        assert out["status"] == "denied", code
        assert {r["rule"] for r in out["reasons"]} & {"content_leak", "encoded_blob", "raw_value"}
    ok = engine.run_python(
        f"import ast\nt = ast.parse({src})\nresult = sorted(n.name for n in ast.walk(t) "
        "if isinstance(n, (ast.FunctionDef, ast.ClassDef)))",
        ["secret_repo"],
    )
    assert ok["status"] == "ok" and ok["result"] == ["LedgerReconciler", "__init__", "proprietary_score", "reconcile"]


def test_python_entity_listing_and_rows_cap(engine: Engine) -> None:
    out = engine.run_python("result = [r['person_id'] for r in rows('people')][:30]")
    assert out["status"] == "denied" and out["reasons"][0]["rule"] == "entity_listing"
    out = engine.run_python("result = list(range(1000))")
    assert out["status"] == "denied" and out["reasons"][0]["rule"] in ("output_rows", "output_bytes")


def test_python_errors_sanitized(engine: Engine) -> None:
    out = engine.run_python("raise ValueError(str(rows('people')[3]['salary']))")
    assert out["status"] == "error" and "withheld" in out["error"]
    out = engine.run_python("x = 1\nraise KeyError('missing_column')")
    assert "KeyError at line 2" in out["error"] and "missing_column" in out["error"]
    assert engine.run_python("result = 1", ["nope"])["status"] == "error"


def test_python_disabled_by_policy(vault: Vault, engine: Engine) -> None:
    set_policy(vault, "people", allow_python=False)
    e = fresh(vault, engine)
    out = e.run_python("result = 1", ["people"])
    assert out["status"] == "denied" and out["reasons"][0]["rule"] == "tool_disabled"


def test_sql_disabled_dataset_invisible(vault: Vault, engine: Engine) -> None:
    set_policy(vault, "people", allow_sql=False)
    e = fresh(vault, engine)
    assert e.run_sql("SELECT count(*) FROM people")["reasons"][0]["rule"] == "unknown_table"


def test_audit_records_everything(vault: Vault, engine: Engine) -> None:
    engine.run_sql("SELECT count(*) FROM people")
    engine.run_sql("SELECT name FROM people")
    engine.budget_status()
    lines = [json.loads(x) for x in vault.audit_path.read_text().splitlines()]
    assert lines[0]["event"] == "session_start"
    statuses = [(r.get("tool"), r.get("status")) for r in lines[1:]]
    assert statuses == [("run_sql", "ok"), ("run_sql", "denied"), ("budget_status", "ok")]
    assert lines[1]["bytes_released"] > 0 and "output_sha256" in lines[1]
    assert "Person Number" not in vault.audit_path.read_text()


def test_owner_side_token_map(engine: Engine) -> None:
    m = engine.token_map("people")
    assert len(m) == 120  # 60 person ids + 60 names
    assert "Person Number3" in m.values()
    assert engine.public_columns("people") == ["dept"]


def test_pii_redacted_in_outputs(vault: Vault, engine: Engine, tmp_path: Path) -> None:
    (tmp_path / "contacts.csv").write_text(
        "team,contact\n" + "".join(f"t{i % 2},ops{i}@corp.example\n" for i in range(20))
    )
    vault.ingest(tmp_path / "contacts.csv")
    set_policy(vault, "contacts", columns={"team": "public", "contact": "public"}, k_min=1)
    e = fresh(vault, engine)
    out = e.run_sql("SELECT contact FROM contacts ORDER BY contact LIMIT 2")
    assert out["status"] == "ok" and out["rows"] == [["[EMAIL]"], ["[EMAIL]"]]
    assert any("PII redacted: EMAILx2" in n for n in out["notes"])
    blob = base64.b64encode(b"x").decode()
    assert blob  # keep import used


def test_data_dependent_error_oracle_capped(vault: Vault, engine: Engine) -> None:
    e = fresh(vault, engine, max_data_errors=2)
    q = "SELECT dept, count(*) FROM people WHERE ln(salary - {x}) > 0 GROUP BY dept"
    assert e.run_sql(q.format(x=10**9))["status"] == "error"
    assert e.run_sql(q.format(x=10**9))["status"] == "error"
    blocked = e.run_sql("SELECT count(*) FROM people")
    assert blocked["status"] == "denied" and "data-dependent" in blocked["reasons"][0]["detail"]
    assert e.budget_status()["data_errors"] == {"used": 2, "limit": 2}


def test_public_only_table_errors_not_data_capped(vault: Vault, engine: Engine, tmp_path: Path) -> None:
    (tmp_path / "cats.csv").write_text("color\nred\nblue\n")
    vault.ingest(tmp_path / "cats.csv")
    set_policy(vault, "cats", columns={"color": "public"})
    e = fresh(vault, engine)
    out = e.run_sql("SELECT CAST(color AS INTEGER) FROM cats")
    assert out["status"] == "error" and "Conversion" in out["error"] and e.budget.data_errors == 0
    assert "'<redacted>'" in out["error"] or "red" not in out["error"]
