from __future__ import annotations

import duckdb
import pytest

from sealbox.errors import QueryRejected
from sealbox.policy import DatasetPolicy
from sealbox.sqlguard import SUPPORT_PREFIX, SqlGuard


@pytest.fixture()
def guard() -> SqlGuard:
    conn = duckdb.connect()
    conn.execute("CREATE TABLE emp (pid VARCHAR, name VARCHAR, dept VARCHAR, salary DOUBLE, yr INT)")
    conn.execute("CREATE TABLE depts (dept VARCHAR, floor INT)")
    conn.execute("CREATE TABLE orders (cust VARCHAR, amount DOUBLE, region VARCHAR)")
    emp = DatasetPolicy(
        dataset="emp",
        columns={"pid": "tokenize", "name": "tokenize", "dept": "public", "salary": "aggregate_only", "yr": "public"},
        entity_column="pid",
    )
    depts = DatasetPolicy(dataset="depts", columns={"dept": "public", "floor": "public"})
    orders = DatasetPolicy(
        dataset="orders",
        columns={"cust": "tokenize", "amount": "aggregate_only", "region": "public"},
        entity_column="cust",
    )
    return SqlGuard(
        conn,
        {"emp": emp, "depts": depts, "orders": orders},
        {
            "emp": ["pid", "name", "dept", "salary", "yr"],
            "depts": ["dept", "floor"],
            "orders": ["cust", "amount", "region"],
        },
        {("emp", "salary"): (1.0, 9.0)},
    )


def rejected(guard: SqlGuard, sql: str) -> str:
    with pytest.raises(QueryRejected) as ei:
        guard.plan(sql)
    return ei.value.rule


def test_aggregate_rewrite_adds_entity_support(guard: SqlGuard) -> None:
    plan = guard.plan("SELECT dept, avg(salary) AS s, count(*) FROM emp GROUP BY dept ORDER BY 2;")
    assert plan.aggregate and plan.n_support == 1 and plan.tables == ["emp"] and plan.sensitive_tables == ["emp"]
    assert f"count(DISTINCT emp.pid) AS {SUPPORT_PREFIX}0" in plan.sql
    assert [(o.agg, o.column, o.sensitive) for o in plan.outputs] == [
        (None, None, False),
        ("avg", ("emp", "salary"), True),
        ("count_star", None, True),
    ]


def test_rowid_support_when_no_entity_and_join_alias(guard: SqlGuard) -> None:
    plan = guard.plan("SELECT d.floor, sum(o.amount) FROM orders o JOIN depts d ON o.region = d.dept GROUP BY 1")
    assert "count(DISTINCT o.cust)" in plan.sql and plan.sensitive_tables == ["orders"]
    guard.policies["orders"].entity_column = None
    plan = guard.plan("SELECT region, sum(amount) FROM orders GROUP BY region")
    assert "count(DISTINCT orders.rowid)" in plan.sql


def test_public_only_tables_need_no_support(guard: SqlGuard) -> None:
    plan = guard.plan("SELECT floor, count(*) FROM depts GROUP BY floor")
    assert plan.aggregate and plan.n_support == 0 and plan.sensitive_tables == []


def test_row_level_public_allowed(guard: SqlGuard) -> None:
    plan = guard.plan("SELECT dept, floor FROM depts WHERE floor > 1 ORDER BY floor LIMIT 3")
    assert not plan.aggregate and plan.sql.startswith("SELECT dept")
    plan = guard.plan("SELECT dept, yr FROM emp")
    assert not plan.aggregate
    guard.policies["emp"].row_level_public = False
    assert rejected(guard, "SELECT dept FROM emp") == "row_level"


def test_distinct_becomes_group_by(guard: SqlGuard) -> None:
    plan = guard.plan("SELECT DISTINCT name FROM emp")
    assert plan.aggregate and 'GROUP BY "name"' in plan.sql and "SELECT DISTINCT" not in plan.sql
    assert rejected(guard, "SELECT DISTINCT * FROM depts") == "unsupported_shape"


@pytest.mark.parametrize(
    ("sql", "rule"),
    [
        ("SELECT name, salary FROM emp", "row_level"),
        ("SELECT * FROM emp", "row_level"),
        ("SELECT pid FROM emp WHERE salary > 5", "row_level"),
        ("SELECT dept, salary FROM emp GROUP BY dept, salary", "aggregate_only"),
        ("SELECT dept, count(*) FROM emp GROUP BY dept ORDER BY salary", "aggregate_only"),
        ("SELECT dept, max(salary) FROM emp GROUP BY dept", "extremal_aggregate"),
        ("SELECT median(salary) FROM emp", "extremal_aggregate"),
        ("SELECT list(salary) FROM emp", "extremal_aggregate"),
        ("SELECT string_agg(name, ',') FROM emp", "tokenized_column"),
        ("SELECT max(name) FROM emp", "tokenized_column"),
        (
            "SELECT dept, count(*) FROM emp WHERE salary > (SELECT avg(salary) FROM emp) GROUP BY dept",
            "unsupported_shape",
        ),
        ("SELECT count(*) FROM (SELECT * FROM emp)", "unsupported_shape"),
        ("WITH x AS (SELECT * FROM emp) SELECT count(*) FROM x", "unsupported_shape"),
        ("SELECT dept FROM emp UNION SELECT dept FROM depts", "unsupported_shape"),
        ("SELECT dept, sum(salary) OVER (PARTITION BY dept) FROM emp", "unsupported_shape"),
        ("SELECT count(*) FROM range(10)", "unsupported_shape"),
        ("SELECT count(*) FROM read_csv('/etc/passwd')", "unsupported_shape"),
        ("SELECT count(*) FROM secrets", "unknown_table"),
        ("SELECT count(*) FROM main.emp", "unknown_table"),
        ("DELETE FROM emp", "read_only"),
        ("SELECT 1; SELECT 2", "read_only"),
        ("SELEC nonsense", "parse_error"),
        ("   ", "parse_error"),
        ("SELECT 1", "unsupported_shape"),
        ("SELECT count(*) FROM emp, depts USING SAMPLE 10", "unsupported_shape"),
        ("SELECT error('x') FROM depts", "forbidden_function"),
        ("SELECT dept, count(*) FROM emp GROUP BY dept QUALIFY count(*) > 1", "unsupported_shape"),
        ("SELECT sum(count(*)) FROM emp", "unsupported_shape"),
        ("SELECT list_transform([1], x -> x + 1) FROM depts", "unsupported_shape"),
        ("SELECT *, count(*) FROM depts GROUP BY ALL", "unsupported_shape"),
    ],
)
def test_rejections(guard: SqlGuard, sql: str, rule: str) -> None:
    assert rejected(guard, sql) == rule


def test_allowed_usages(guard: SqlGuard) -> None:
    for sql in [
        "SELECT dept, count(DISTINCT name) FROM emp GROUP BY dept",
        "SELECT dept, stddev(salary), corr(salary, yr) FROM emp WHERE salary > 2 GROUP BY dept HAVING count(*) > 3",
        "SELECT name, count(*) FROM emp GROUP BY name",
        "SELECT round(avg(salary), 2) FROM emp",
        "SELECT e.dept, max(e.yr) FROM emp e JOIN depts USING (dept) GROUP BY e.dept",
    ]:
        assert guard.plan(sql).aggregate, sql


def test_output_meta(guard: SqlGuard) -> None:
    plan = guard.plan(
        "SELECT round(avg(salary), 1), sum(salary) / count(*), CAST(sum(salary) AS INT), "
        "count(DISTINCT name), max(yr) FROM emp"
    )
    got = [(o.agg, o.sensitive) for o in plan.outputs]
    assert got == [("avg", True), ("expr", True), ("sum", True), ("count", True), ("derived", False)]
    assert plan.outputs[0].column == ("emp", "salary")


def test_join_condition_columns_checked(guard: SqlGuard) -> None:
    assert rejected(guard, "SELECT d.floor FROM depts d JOIN emp e ON e.salary = d.floor") == "row_level"
    plan = guard.plan("SELECT d.floor, count(*) FROM depts d JOIN emp e ON e.dept = d.dept GROUP BY 1")
    assert plan.sensitive_tables == ["emp"]
