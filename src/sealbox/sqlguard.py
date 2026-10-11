"""Static analysis + rewriting of agent-written SQL (DuckDB AST).

v0.1 accepts a deliberately small, analyzable subset:

* one ``SELECT`` over vault tables (joins allowed); no CTEs, subqueries, set operations,
  window functions, table functions or ``VALUES``;
* column tags are enforced per reference (``aggregate_only`` only inside
  COUNT/SUM/AVG/STDDEV-style aggregates, ``tokenize`` only raw or in COUNT, ...);
* aggregate queries are rewritten to also return ``count(DISTINCT <entity>)`` per output
  row for every sensitive table, so the egress gate can enforce k-anonymity on the real
  number of contributing individuals (joins and ``range()`` cannot inflate it);
* non-aggregate (row-level) queries are allowed only over ``public`` columns.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

import duckdb

from .errors import QueryRejected
from .policy import AGGREGATE_ONLY, PUBLIC, TOKENIZE, DatasetPolicy

# Aggregates that do not expose any single row's value when the group is large enough.
SAFE_AGGS = frozenset(
    {
        "count",
        "count_star",
        "sum",
        "avg",
        "mean",
        "stddev",
        "stddev_samp",
        "stddev_pop",
        "variance",
        "var_samp",
        "var_pop",
        "corr",
        "covar_pop",
        "covar_samp",
        "approx_count_distinct",
        "count_if",
        "sum_no_overflow",
        "fsum",
        "favg",
        "kahan_sum",
        "regr_slope",
        "regr_intercept",
        "regr_r2",
        "regr_count",
        "bool_and",
        "bool_or",
        "entropy",
    }
)
# Aggregates that return (or can return) one row's value: allowed on public columns only.
EXTREMAL_AGGS = frozenset(
    {
        "min",
        "max",
        "median",
        "quantile",
        "quantile_cont",
        "quantile_disc",
        "mode",
        "first",
        "last",
        "any_value",
        "arg_min",
        "arg_max",
        "argmin",
        "argmax",
        "min_by",
        "max_by",
        "list",
        "array_agg",
        "string_agg",
        "group_concat",
        "listagg",
        "histogram",
        "approx_quantile",
        "reservoir_quantile",
        "bit_and",
        "bit_or",
        "bit_xor",
        "product",
        "arbitrary",
        "kurtosis",
        "skewness",
        "list_agg",
    }
)
ALL_AGGS = SAFE_AGGS | EXTREMAL_AGGS
DP_AGGS = frozenset({"count", "count_star", "sum", "avg", "mean", "count_if", "approx_count_distinct"})
# Wrappers that keep a single aggregate "plain" for DP purposes.
_TRANSPARENT = frozenset({"round", "cast", "try_cast", "floor", "ceil", "abs"})
FORBIDDEN_FUNCS = frozenset(
    {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_json_auto",
        "read_text",
        "read_blob",
        "glob",
        "parquet_scan",
        "query",
        "query_table",
        "sniff_csv",
        "duckdb_settings",
        "duckdb_secrets",
        "current_setting",
        "error",
    }
)

SUPPORT_PREFIX = "__sb_support_"


@dataclass
class OutputCol:
    index: int
    agg: str | None  # plain DP-able aggregate, if any
    column: tuple[str, str] | None  # (table, column) the aggregate is over
    sensitive: bool


@dataclass
class SqlPlan:
    sql: str  # SQL to execute (possibly rewritten)
    aggregate: bool
    tables: list[str]
    sensitive_tables: list[str]
    n_support: int  # number of trailing support columns
    outputs: list[OutputCol] = field(default_factory=list)
    has_star: bool = False


@dataclass
class _Ref:
    table: str
    column: str
    clause: str
    agg: str | None


def _serialize(conn: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    raw = conn.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]  # type: ignore[index]
    data: dict[str, Any] = json.loads(raw)
    if data.get("error"):
        msg = str(data.get("error_message", "could not parse"))
        if "Only SELECT" in msg:
            raise QueryRejected("read_only", "only a single read-only SELECT statement is allowed")
        raise QueryRejected("parse_error", f"SQL parse error: {msg[:300]}")
    return data


class SqlGuard:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        policies: dict[str, DatasetPolicy],
        columns: dict[str, list[str]],
        numeric_bounds: dict[tuple[str, str], tuple[float, float]],
    ):
        self.conn = conn
        self.policies = policies  # table name -> policy
        self.columns = {t: {c.lower(): c for c in cols} for t, cols in columns.items()}
        self.bounds = numeric_bounds

    # ------------------------------------------------------------------ public
    def plan(self, sql: str) -> SqlPlan:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise QueryRejected("parse_error", "empty query")
        data = _serialize(self.conn, sql)
        stmts = data.get("statements", [])
        if len(stmts) != 1:
            raise QueryRejected("read_only", "exactly one SELECT statement is allowed")
        node = stmts[0]["node"]
        if node.get("type") != "SELECT_NODE":
            raise QueryRejected(
                "unsupported_shape", "set operations (UNION/INTERSECT/EXCEPT) are not supported in the safe SQL subset"
            )
        if node.get("cte_map", {}).get("map"):
            raise QueryRejected("unsupported_shape", "CTEs (WITH ...) are not supported in the safe SQL subset")
        for mod in node.get("modifiers", []):
            if mod.get("type") not in ("ORDER_MODIFIER", "LIMIT_MODIFIER", "DISTINCT_MODIFIER"):
                raise QueryRejected("unsupported_shape", f"modifier {mod.get('type')} is not supported")
        if node.get("qualify"):
            raise QueryRejected("unsupported_shape", "QUALIFY is not supported")
        if node.get("sample"):
            raise QueryRejected("unsupported_shape", "TABLESAMPLE is not supported")

        aliases: dict[str, str] = {}
        tables: list[str] = []
        join_refs: list[_Ref] = []
        self._walk_from(node.get("from_table"), aliases, tables, join_refs)
        if not tables:
            raise QueryRejected("unsupported_shape", "query must read from a vault table")

        refs: list[_Ref] = []
        has_star = False
        select_aggs: list[bool] = []
        for item in node.get("select_list", []):
            if item.get("class") == "STAR":
                has_star = True
            found = self._walk_expr(item, "select", None, aliases, tables, refs)
            select_aggs.append(found)
        group = node.get("group_expressions", []) or []
        for g in group:
            self._walk_expr(g, "group", None, aliases, tables, refs)
        for clause in ("where_clause", "having"):
            if node.get(clause):
                self._walk_expr(node[clause], "filter", None, aliases, tables, refs)
        for mod in node.get("modifiers", []):
            if mod.get("type") == "ORDER_MODIFIER":
                for o in mod.get("orders", []):
                    self._walk_expr(o.get("expression"), "order", None, aliases, tables, refs)
            elif mod.get("type") == "LIMIT_MODIFIER":
                for k in ("limit", "offset"):
                    if mod.get(k):
                        self._walk_expr(mod[k], "filter", None, aliases, tables, refs)

        distinct = any(m.get("type") == "DISTINCT_MODIFIER" for m in node.get("modifiers", []))
        aggregate = bool(group) or any(select_aggs)
        if distinct and not aggregate:
            # SELECT DISTINCT a, b == GROUP BY a, b: rewrite so k-anonymity applies
            if has_star:
                raise QueryRejected("unsupported_shape", "SELECT DISTINCT * is not supported; name the columns")
            node["modifiers"] = [m for m in node["modifiers"] if m.get("type") != "DISTINCT_MODIFIER"]
            node["group_expressions"] = [self._strip_alias(copy.deepcopy(e)) for e in node["select_list"]]
            node["group_sets"] = [list(range(len(node["select_list"])))]
            aggregate = True
        sensitive_tables = [t for t in dict.fromkeys(tables) if self._sensitive(t)]

        if has_star:
            for t in tables:
                for col in self.columns[t].values():
                    refs.append(_Ref(t, col, "select", None))

        self._check_refs(refs + join_refs, aggregate)

        if not aggregate:
            return SqlPlan(sql=sql, aggregate=False, tables=tables, sensitive_tables=[], n_support=0, has_star=has_star)

        if has_star:
            raise QueryRejected("unsupported_shape", "SELECT * cannot be combined with aggregation")

        outputs = [
            self._output_meta(i, item, aliases, tables, bool(sensitive_tables))
            for i, item in enumerate(node["select_list"])
        ]
        for i, t in enumerate(sensitive_tables):
            alias = next((a for a, real in aliases.items() if real == t), t)
            entity = self.policies[t].entity_column
            target = self.columns[t].get(entity.lower(), "rowid") if entity else "rowid"
            node["select_list"].append(self._count_distinct(alias, target, f"{SUPPORT_PREFIX}{i}"))
        rewritten = self.conn.execute("SELECT json_deserialize_sql(?)", [json.dumps(data)]).fetchone()[0]  # type: ignore[index]
        return SqlPlan(
            sql=rewritten,
            aggregate=True,
            tables=tables,
            sensitive_tables=sensitive_tables,
            n_support=len(sensitive_tables),
            outputs=outputs,
            has_star=False,
        )

    # ------------------------------------------------------------------ helpers
    def _sensitive(self, table: str) -> bool:
        pol = self.policies[table]
        return any(pol.tag(c) != PUBLIC for c in self.columns[table].values())

    @staticmethod
    def _strip_alias(expr: dict[str, Any]) -> dict[str, Any]:
        expr["alias"] = ""
        return expr

    @staticmethod
    def _count_distinct(alias: str, column: str, out_alias: str) -> dict[str, Any]:
        return {
            "class": "FUNCTION",
            "type": "FUNCTION",
            "alias": out_alias,
            "function_name": "count",
            "schema": "",
            "children": [{"class": "COLUMN_REF", "type": "COLUMN_REF", "alias": "", "column_names": [alias, column]}],
            "filter": None,
            "order_bys": {"type": "ORDER_MODIFIER", "orders": []},
            "distinct": True,
            "is_operator": False,
            "export_state": False,
            "catalog": "",
        }

    def _walk_from(
        self, node: dict[str, Any] | None, aliases: dict[str, str], tables: list[str], join_refs: list[_Ref]
    ) -> None:
        if node is None:
            return
        kind = node.get("type")
        if kind == "BASE_TABLE":
            name = node.get("table_name", "").lower()
            if node.get("schema_name") or node.get("catalog_name"):
                raise QueryRejected("unknown_table", "schema-qualified tables are not allowed")
            if name not in self.policies:
                raise QueryRejected("unknown_table", f"table {name!r} is not a vault dataset")
            if node.get("sample"):
                raise QueryRejected("unsupported_shape", "TABLESAMPLE is not supported")
            aliases[(node.get("alias") or name).lower()] = name
            tables.append(name)
        elif kind == "JOIN":
            self._walk_from(node.get("left"), aliases, tables, join_refs)
            self._walk_from(node.get("right"), aliases, tables, join_refs)
            if node.get("condition"):
                self._walk_expr(node["condition"], "filter", None, aliases, tables, join_refs)
            if node.get("using_columns"):
                for col in node["using_columns"]:
                    for t in tables:
                        real = self.columns[t].get(str(col).lower())
                        if real:
                            join_refs.append(_Ref(t, real, "filter", None))
        elif kind == "TABLE_FUNCTION":
            raise QueryRejected("unsupported_shape", "table functions are not allowed in FROM")
        elif kind == "SUBQUERY":
            raise QueryRejected("unsupported_shape", "subqueries in FROM are not supported in the safe SQL subset")
        else:
            raise QueryRejected("unsupported_shape", f"FROM clause type {kind} is not supported")

    def _walk_expr(
        self, expr: Any, clause: str, agg: str | None, aliases: dict[str, str], tables: list[str], refs: list[_Ref]
    ) -> bool:
        """Collect column references; return True if an aggregate function was found."""
        found = False
        if isinstance(expr, list):
            for e in expr:
                found |= self._walk_expr(e, clause, agg, aliases, tables, refs)
            return found
        if not isinstance(expr, dict):
            return False
        cls = expr.get("class")
        if cls == "SUBQUERY" or expr.get("type") in ("SUBQUERY",):
            raise QueryRejected("unsupported_shape", "subqueries are not supported in the safe SQL subset")
        if cls == "WINDOW":
            raise QueryRejected("unsupported_shape", "window functions are not supported (they expose row order)")
        if cls == "LAMBDA":
            raise QueryRejected("unsupported_shape", "lambda functions are not supported")
        if cls == "FUNCTION":
            fname = str(expr.get("function_name", "")).lower()
            if fname in FORBIDDEN_FUNCS:
                raise QueryRejected("forbidden_function", f"function {fname}() is not allowed")
            if fname in ALL_AGGS:
                if agg is not None:
                    raise QueryRejected("unsupported_shape", "nested aggregates are not supported")
                found = True
                agg = fname
        if cls == "COLUMN_REF":
            names = [str(n).lower() for n in expr.get("column_names", [])]
            for table, column in self._resolve(names, aliases, tables):
                refs.append(_Ref(table, column, clause, agg))
            return False
        for key, value in expr.items():
            if key in ("class", "type", "alias", "query_location", "function_name", "schema", "catalog"):
                continue
            if isinstance(value, (dict, list)):
                found |= self._walk_expr(value, clause, agg, aliases, tables, refs)
        return found

    def _resolve(self, names: list[str], aliases: dict[str, str], tables: list[str]) -> list[tuple[str, str]]:
        if not names:
            return []
        if len(names) >= 2 and names[-2] in aliases:
            table = aliases[names[-2]]
            col = self.columns[table].get(names[-1])
            return [(table, col)] if col else []
        out = []
        for t in dict.fromkeys(tables):
            col = self.columns[t].get(names[-1])
            if col:
                out.append((t, col))
        return out

    def _check_refs(self, refs: list[_Ref], aggregate: bool) -> None:
        for r in refs:
            tag = self.policies[r.table].tag(r.column)
            where = f"{r.table}.{r.column}"
            if not aggregate:
                if tag != PUBLIC:
                    raise QueryRejected(
                        "row_level",
                        f"row-level (non-aggregate) queries may only reference public columns; {where} is "
                        f"{tag}. Aggregate with GROUP BY instead.",
                    )
                if not self.policies[r.table].row_level_public:
                    raise QueryRejected("row_level", f"row-level queries are disabled for {r.table}")
                continue
            if tag == AGGREGATE_ONLY:
                if r.agg is None and r.clause in ("select", "group", "order"):
                    raise QueryRejected(
                        "aggregate_only",
                        f"{where} is aggregate_only: use it only inside COUNT/SUM/AVG/STDDEV-style "
                        f"aggregates (not in SELECT/GROUP BY/ORDER BY directly)",
                    )
                if r.agg is not None and r.agg not in SAFE_AGGS:
                    raise QueryRejected(
                        "extremal_aggregate",
                        f"{r.agg}() can reveal a single row's value; not allowed on aggregate_only column {where}",
                    )
            elif tag == TOKENIZE:
                if r.agg is not None and r.agg not in ("count", "count_star", "approx_count_distinct", "count_if"):
                    raise QueryRejected(
                        "tokenized_column", f"{where} is tokenized: only COUNT/COUNT(DISTINCT) aggregates are allowed"
                    )

    def _output_meta(
        self, index: int, item: dict[str, Any], aliases: dict[str, str], tables: list[str], over_sensitive: bool
    ) -> OutputCol:
        expr = item
        while expr.get("class") in ("FUNCTION", "CAST") and (
            str(expr.get("function_name", "")).lower() in _TRANSPARENT or expr.get("class") == "CAST"
        ):
            children = expr.get("children") or ([expr["child"]] if "child" in expr else [])
            if not children:
                break
            expr = children[0]
        refs: list[_Ref] = []
        is_agg = self._walk_expr(item, "select", None, aliases, tables, refs)
        # counting/summing rows of a table that holds individuals is sensitive even over public columns
        sensitive = any(self.policies[r.table].tag(r.column) != PUBLIC for r in refs if r.agg) or (
            over_sensitive and _has_counting_agg(item)
        )
        fname = str(expr.get("function_name", "")).lower() if expr.get("class") == "FUNCTION" else ""
        if fname in DP_AGGS:
            inner: list[_Ref] = []
            self._walk_expr(expr.get("children", []), "select", fname, aliases, tables, inner)
            column = (inner[0].table, inner[0].column) if len(inner) == 1 else None
            return OutputCol(index, fname, column, sensitive)
        return OutputCol(index, "derived" if is_agg else None, None, sensitive)


def _has_counting_agg(expr: Any) -> bool:
    if isinstance(expr, list):
        return any(_has_counting_agg(e) for e in expr)
    if not isinstance(expr, dict):
        return False
    if expr.get("class") == "FUNCTION" and str(expr.get("function_name", "")).lower() in SAFE_AGGS:
        return True
    return any(_has_counting_agg(v) for k, v in expr.items() if isinstance(v, (dict, list)))
