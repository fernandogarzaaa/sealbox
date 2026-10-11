"""DP for STDDEV/VARIANCE via operator clamp bounds, arithmetic over DP aggregates, CASE bucketing,
length() calibration, and ast.get_docstring in the sandbox."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pytest
import yaml

import sealbox.engine as engine_mod
from sealbox.engine import Engine
from sealbox.errors import PolicyError, QueryRejected
from sealbox.policy import DatasetPolicy, default_policy, load_policy, nice_bounds
from sealbox.sandbox import run_python
from sealbox.sqlguard import SqlGuard, evaluate
from sealbox.vault import Vault


# ------------------------------------------------------------------------------------------------ guard
def _guard(bounds: dict[str, list[float]] | None = None) -> SqlGuard:
    conn = duckdb.connect()
    conn.execute("CREATE TABLE emp (pid VARCHAR, dept VARCHAR, salary DOUBLE, bonus DOUBLE)")
    conn.execute("INSERT INTO emp VALUES ('a', 'x', 10, 1), ('b', 'x', 500, 2), ('c', 'y', -50, 3)")
    pol = DatasetPolicy(
        dataset="emp",
        columns={"pid": "tokenize", "dept": "public", "salary": "aggregate_only", "bonus": "aggregate_only"},
        entity_column="pid",
    )
    pol.dp.bounds = bounds or {}
    return SqlGuard(conn, {"emp": pol}, {"emp": ["pid", "dept", "salary", "bonus"]}, {})


def test_case_bucket_allowed_in_select_and_group() -> None:
    g = _guard()
    plan = g.plan(
        "SELECT CASE WHEN salary < 100 THEN 'low' WHEN salary BETWEEN 100 AND 200 THEN 'mid' ELSE 'high' END AS band, "
        "count(*) FROM emp GROUP BY band"
    )
    assert plan.aggregate and plan.n_support == 1
    plan2 = g.plan("SELECT count(*) FROM emp GROUP BY CASE WHEN salary < 100 AND bonus > 1 THEN 1 ELSE 0 END")
    assert plan2.aggregate


@pytest.mark.parametrize(
    "sql",
    [
        # THEN returns the protected value itself
        "SELECT CASE WHEN salary > 0 THEN salary ELSE 0 END AS b, count(*) FROM emp GROUP BY b",
        # condition is not a plain comparison (function of the value)
        "SELECT CASE WHEN round(salary) = 10 THEN 'a' ELSE 'b' END AS b, count(*) FROM emp GROUP BY b",
        # string function in THEN
        "SELECT CASE WHEN salary > 0 THEN upper(dept) END AS b, count(*) FROM emp GROUP BY b",
        # not aggregated: row-level listing of buckets per row
        "SELECT CASE WHEN salary < 100 THEN 'low' ELSE 'high' END FROM emp",
    ],
)
def test_case_bucket_rejections(sql: str) -> None:
    with pytest.raises(QueryRejected):
        _guard().plan(sql)


def test_case_bucket_branch_limit() -> None:
    whens = " ".join(f"WHEN salary < {i} THEN '{i}'" for i in range(13))
    with pytest.raises(QueryRejected):
        _guard().plan(f"SELECT CASE {whens} END AS b, count(*) FROM emp GROUP BY b")
    whens12 = " ".join(f"WHEN salary < {i} THEN '{i}'" for i in range(12))
    assert _guard().plan(f"SELECT CASE {whens12} END AS b, count(*) FROM emp GROUP BY b").aggregate


def test_arith_expression_becomes_hidden_leaves() -> None:
    g = _guard()
    plan = g.plan("SELECT dept, round(sum(bonus) / sum(salary) * 100, 2) AS r FROM emp GROUP BY dept")
    o = plan.outputs[1]
    assert o.agg == "expr" and [leaf.agg for leaf in o.leaves] == ["sum", "sum"] and plan.n_hidden == 2
    assert "__sb_leaf_0" in plan.sql and "__sb_leaf_1" in plan.sql
    assert o.tree == ("op", "round", [("op", "*", [("op", "/", [("l", 0), ("l", 1)]), ("c", 100.0)]), ("c", 2.0)])


def test_stddev_is_leaf_and_clamped_with_policy_bounds() -> None:
    g = _guard({"salary": [0, 100]})
    plan = g.plan("SELECT dept, stddev(salary), avg(salary) FROM emp GROUP BY dept")
    sd, av = plan.outputs[1], plan.outputs[2]
    assert sd.agg == "expr" and sd.leaves[0].agg == "stddev" and sd.leaves[0].bounds == (0.0, 100.0)
    assert av.agg == "avg" and av.bounds == (0.0, 100.0)
    assert "least(greatest(salary" in plan.sql.replace('"', "")
    rows = {r[0]: r[2] for r in g.conn.execute(plan.sql).fetchall()}
    assert rows["x"] == pytest.approx(55.0)  # 10 and min(500, 100) -> clamped mean


def test_stddev_without_bounds_has_no_bounds() -> None:
    plan = _guard().plan("SELECT dept, variance(salary) FROM emp GROUP BY dept")
    assert plan.outputs[1].leaves[0].bounds is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT dept, sum(salary * 1000) FROM emp GROUP BY dept",  # rescaled value: uncalibratable
        "SELECT dept, avg(salary + bonus) FROM emp GROUP BY dept",
        "SELECT dept, sum(salary) / max(salary) FROM emp GROUP BY dept",
        "SELECT dept, sum(DISTINCT salary) / count(*) FROM emp GROUP BY dept",
        "SELECT dept, " + " + ".join(["sum(salary)"] * 9) + " FROM emp GROUP BY dept",  # > MAX_LEAVES
        "SELECT dept, avg(length(CAST(salary AS VARCHAR))) FROM emp GROUP BY dept",
        "SELECT dept, avg(length(salary)) FROM emp GROUP BY dept",  # no length bound known
    ],
)
def test_uncalibratable_expressions_stay_derived(sql: str) -> None:
    try:
        plan = _guard().plan(sql)
    except QueryRejected:
        return  # max() on aggregate_only is rejected outright: also fine
    assert plan.outputs[1].agg == "derived"


def test_length_with_policy_bound_is_clamped() -> None:
    g = _guard({"length(dept)": [0, 10]})
    plan = g.plan("SELECT dept, avg(length(dept)) FROM emp GROUP BY dept")
    assert plan.outputs[1].agg == "avg" and plan.outputs[1].bounds == (0.0, 10.0)
    assert "least(greatest(length(dept)" in plan.sql.replace('"', "")


def test_avg_over_constant_case_uses_constant_bounds() -> None:
    plan = _guard().plan("SELECT dept, avg(CASE WHEN salary > 100 THEN 1 ELSE 0 END) FROM emp GROUP BY dept")
    o = plan.outputs[1]
    assert o.agg == "avg" and o.bounds == (0.0, 1.0) and o.column is None
    plan2 = _guard().plan("SELECT dept, sum(CASE WHEN salary > 100 THEN 5 END) FROM emp GROUP BY dept")
    assert plan2.outputs[1].bounds == (5.0, 6.0)
    plan3 = _guard().plan("SELECT dept, sum(CASE WHEN salary > 100 THEN 'a' ELSE 'b' END) FROM emp GROUP BY dept")
    assert plan3.outputs[1].agg == "derived"


def test_evaluate() -> None:
    assert evaluate(("op", "/", [("l", 0), ("l", 1)]), [1.0, 4.0]) == 0.25
    assert evaluate(("op", "/", [("l", 0), ("l", 1)]), [1.0, 0.0]) is None
    assert evaluate(("op", "+", [("l", 0), ("c", 2.0)]), [None]) is None
    assert evaluate(("op", "+", [("l", 0), ("c", 2.0)]), [1.5]) == 3.5
    assert evaluate(("op", "-", [("c", 3.0)]), []) == -3.0
    assert evaluate(("op", "-", [("c", 3.0), ("c", 1.0)]), []) == 2.0
    assert evaluate(("op", "*", [("c", 3.0), ("c", 2.0)]), []) == 6.0
    assert evaluate(("op", "round", [("c", 3.14159), ("c", 2.0)]), []) == 3.14
    assert evaluate(("op", "round", [("c", 2.6)]), []) == 3.0
    assert evaluate(("op", "abs", [("c", -2.0)]), []) == 2.0
    assert evaluate(("op", "floor", [("c", 2.7)]), []) == 2.0
    assert evaluate(("op", "ceil", [("c", 2.1)]), []) == 3.0
    assert evaluate(("op", "cast", [("c", 2.5)]), []) == 2.5


# ------------------------------------------------------------------------------------------------ policy
def test_nice_bounds() -> None:
    assert nice_bounds(45_123, 212_345) == [0.0, 300_000.0]
    assert nice_bounds(-3.2, 7.9) == [-4.0, 8.0]
    assert nice_bounds(5, 5) == [5.0, 6.0]
    assert nice_bounds(0.2, 0.7) == [0.0, 1.0]


def test_default_policy_fills_bounds_for_numeric_aggregate_only() -> None:
    pol = default_policy(
        "t",
        [
            {"name": "salary", "n_distinct": 90, "n_rows": 100, "is_numeric": True, "avg_len": 0, "min": 41, "max": 99},
            {"name": "dept", "n_distinct": 3, "n_rows": 100, "is_numeric": False, "avg_len": 3},
        ],
        "table",
    )
    assert pol.dp.bounds == {"salary": [40.0, 100.0]}


@pytest.mark.parametrize("bounds", ["[1, 2]", "{a: [2, 1]}", "{a: [1]}", "{a: [x, 2]}", "{a: [true, 2]}", "{a: 3}"])
def test_bounds_validation(tmp_path: Path, bounds: str) -> None:
    p = tmp_path / "p.yaml"
    p.write_text(f"dp: {{bounds: {bounds}}}\n")
    with pytest.raises(PolicyError):
        load_policy(p, "t")


def test_bounds_load(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text("dp: {bounds: {salary: [0, 200000]}}\n")
    assert load_policy(p, "t").dp.bounds == {"salary": [0, 200000]}


# ------------------------------------------------------------------------------------------------ engine
def _set_dp(vault: Vault, **dp: object) -> None:
    path = vault.policy_path("people")
    data = yaml.safe_load(path.read_text())
    data.setdefault("dp", {}).update(dp)
    path.write_text(yaml.safe_dump(data))


def test_engine_generated_policy_has_salary_bounds(vault: Vault) -> None:
    assert "salary" in vault.policy("people").dp.bounds


def test_engine_stddev_ratio_and_bands(vault: Vault, engine: Engine) -> None:
    sd = engine.run_sql("SELECT dept, stddev(salary), avg(salary) FROM people GROUP BY dept")
    assert sd["status"] == "ok", sd
    assert any("clamped" in n for n in sd["notes"])
    assert engine.budget.epsilon_spent == pytest.approx(2.0)
    ratio = engine.run_sql("SELECT dept, sum(salary) / count(*) AS mean_ish FROM people GROUP BY dept")
    assert ratio["status"] == "ok"
    assert engine.budget.epsilon_spent == pytest.approx(4.0)  # two leaves, one epsilon each
    vals = [r[1] for r in ratio["rows"]]
    # sum noise is calibrated to the clamp bound, so values are noisy but finite floats (or None on /0)
    assert vals and all(v is None or (isinstance(v, float) and math.isfinite(v)) for v in vals)
    bands = engine.run_sql(
        "SELECT CASE WHEN salary < 70000 THEN 'low' ELSE 'high' END AS band, count(*) FROM people GROUP BY band"
    )
    assert bands["status"] == "ok" and {r[0] for r in bands["rows"]} <= {"low", "high"}


def test_engine_stddev_without_bounds_denied(vault: Vault, engine: Engine) -> None:
    _set_dp(vault, bounds={})
    e = Engine(vault, engine.vault.keys, engine.config)
    out = e.run_sql("SELECT dept, stddev(salary) FROM people GROUP BY dept")
    assert out["status"] == "denied" and out["reasons"][0]["rule"] == "dp_uncalibrated"
    assert "dp.bounds" in out["reasons"][0]["detail"]


def test_engine_expr_without_dp_returns_exact(vault: Vault, engine: Engine) -> None:
    _set_dp(vault, enabled=False)
    e = Engine(vault, engine.vault.keys, engine.config)
    out = e.run_sql("SELECT dept, round(sum(salary) / count(*), 2) FROM people GROUP BY dept ORDER BY dept")
    ref = e.run_sql("SELECT dept, round(avg(salary), 2) FROM people GROUP BY dept ORDER BY dept")
    assert out["status"] == "ok" and out["rows"] == ref["rows"]
    assert len(out["columns"]) == 2  # hidden leaf columns never reach the agent


def test_length_of_text_column_is_calibrated(engine: Engine) -> None:
    prep = engine._prep("people")
    assert prep.bounds["length(notes)"][0] == 0.0 and prep.bounds["length(notes)"][1] >= 80
    out = engine.run_sql("SELECT dept, avg(length(notes)) FROM people GROUP BY dept")
    assert out["status"] == "ok" and any("Laplace" in n for n in out["notes"])


def test_noise_variance_and_stddev(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_mod, "laplace", lambda scale: scale)
    assert engine._noise(4.0, "variance", None, {}, 11, 1.0, (0.0, 10.0)) == pytest.approx(14.0)
    assert engine._noise(4.0, "stddev", None, {}, 26, 1.0, (0.0, 10.0)) == pytest.approx(6.0)
    assert engine._noise(4.0, "stddev_pop", None, {}, 1, 1.0, (0.0, 10.0)) == pytest.approx(14.0)
    monkeypatch.setattr(engine_mod, "laplace", lambda scale: -scale)
    assert engine._noise(1.0, "var_pop", None, {}, 11, 1.0, (0.0, 10.0)) == 0.0  # clipped at zero
    # operator clamp bounds override the data range for avg/sum
    monkeypatch.setattr(engine_mod, "laplace", lambda scale: scale)
    b = {("people", "salary"): (0.0, 1e6)}
    assert engine._noise(5.0, "avg", ("people", "salary"), b, 10, 1.0, (0.0, 100.0)) == pytest.approx(15.0)
    assert engine._noise(5.0, "sum", ("people", "salary"), b, 10, 1.0, (0.0, 100.0)) == pytest.approx(105.0)
    assert math.isfinite(engine._noise(5.0, "avg", None, {}, 0, 1.0))


# ------------------------------------------------------------------------------------------------ sandbox
def test_sandbox_get_docstring_works_and_marshal_still_blocked() -> None:
    r = run_python(
        "import ast\nresult = ast.get_docstring(ast.parse(docs['d']['a.py']))",
        {},
        {"d": {"a.py": '"""Module doc."""\nx = 1\n'}},
        timeout_s=15,
    )
    assert r.ok and r.result == "Module doc."
    r2 = run_python("import marshal\nresult = marshal.loads(b'')", {}, {}, timeout_s=15)
    assert not r2.ok and r2.error_type == "PermissionError"
    # file reads by inspect/linecache still go through the open() audit hook
    r3 = run_python("import linecache\nresult = linecache.getlines('/etc/passwd')", {}, {}, timeout_s=15)
    assert not r3.ok or r3.result == []
