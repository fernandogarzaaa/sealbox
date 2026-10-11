"""Transport-independent tool logic. The MCP server is a thin wrapper around ``Engine``."""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .audit import AuditLog
from .budget import SessionBudget
from .compute import Prepared, jsonable, locked_connection, prepare
from .crypto import KeyProvider
from .errors import QueryRejected, SealboxError
from .gate import EgressGate, GateResult, Reason, canonical_bytes
from .policy import PUBLIC, SessionConfig
from .sandbox import pad_latency, run_python
from .sqlguard import SqlGuard
from .synth import describe
from .tokenizer import Tokenizer
from .vault import Vault

PAGE_ROWS = 50
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_RNG = random.SystemRandom()


def laplace(scale: float) -> float:
    """Laplace(0, scale) noise from the OS CSPRNG."""
    u = _RNG.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


@dataclass
class StoredResult:
    columns: list[str]
    rows: list[list[Any]]
    datasets: list[str]
    released_until: int


class Engine:
    def __init__(self, vault: Vault, keys: KeyProvider, session: SessionConfig | None = None) -> None:
        self.vault = vault
        self.session_id = secrets.token_hex(8)
        self.config = session or vault.session_config()
        self.budget = SessionBudget(self.config)
        self.audit = AuditLog(vault.audit_path, keys.derive("audit-mac"))
        self.tokenizer = Tokenizer(keys.derive("tokenize"))
        self._prepared: dict[str, Prepared] = {}
        self._results: dict[str, StoredResult] = {}
        self._conn: Any = None
        self._conn_key: tuple[str, ...] = ()
        self._lock = threading.RLock()
        self.audit.append("session_start", session=self.session_id, version=__version__)

    # ------------------------------------------------------------------ helpers
    def _prep(self, name: str) -> Prepared:
        with self._lock:
            if name not in self._prepared:
                self._prepared[name] = prepare(self.vault, name, self.tokenizer)
            return self._prepared[name]

    def _sensitive(self, name: str) -> bool:
        p = self._prepared[name]
        return any(p.policy.tag(c) != PUBLIC for c in p.columns) or len(p.columns) < len(p.meta.get("columns", []))

    def _gate(self) -> EgressGate:
        return EgressGate(self._prepared)

    def _audit(self, tool: str, args: dict[str, Any], status: str, **extra: Any) -> None:
        self.audit.append("tool_call", session=self.session_id, tool=tool, args=args, status=status, **extra)

    def _deny(self, tool: str, args: dict[str, Any], reasons: list[Reason], hint: str = "") -> dict[str, Any]:
        self.budget.record_denial()
        self._audit(tool, args, "denied", reasons=[r.as_dict() for r in reasons])
        out: dict[str, Any] = {"status": "denied", "reasons": [r.as_dict() for r in reasons]}
        if hint:
            out["hint"] = hint
        out["budget"] = self._short_budget()
        return out

    def _error(self, tool: str, args: dict[str, Any], message: str) -> dict[str, Any]:
        self.budget.record_error()
        self._audit(tool, args, "error", message=message)
        return {"status": "error", "error": message, "budget": self._short_budget()}

    def _short_budget(self) -> dict[str, Any]:
        s = self.budget.status()
        return {"epsilon_remaining": s["epsilon"]["remaining"], "bytes_remaining": s["bytes"]["remaining"]}  # type: ignore[index]

    def _release(
        self, tool: str, args: dict[str, Any], gate: GateResult, body: dict[str, Any], eps: float = 0.0
    ) -> dict[str, Any]:
        body = {"status": "ok", **body}
        if gate.notes:
            body.setdefault("notes", []).extend(gate.notes)
        nbytes = canonical_bytes(body)
        decision = self.budget.charge(nbytes, eps)
        if not decision.ok:
            return self._deny(
                tool,
                args,
                [Reason("leakage_budget", decision.reason)],
                "The session's leakage budget is spent; the operator can restart the session.",
            )
        body["budget"] = self._short_budget()
        self._audit(
            tool,
            args,
            "ok",
            bytes_released=nbytes,
            epsilon=round(eps, 4),
            output_sha256=hashlib.sha256(repr(body).encode()).hexdigest(),
        )
        return body

    def _sanitize_error(self, msg: str, sql: str, datasets: list[str]) -> str:
        """Engine errors can echo data values (e.g. conversion errors): redact quoted literals not in the query."""

        def repl(m: re.Match[str]) -> str:
            return m.group(0) if m.group(0)[1:-1] in sql else "'<redacted>'"

        msg = _QUOTED.sub(repl, msg.split("\n")[0])[:300]
        gate = self._gate().inspect({"error": msg}, datasets, check_protected=True, origin="error message")
        return msg if gate.allowed else "query failed (error details withheld: they would reveal data)"

    # ------------------------------------------------------------------ tools
    def list_datasets(self) -> dict[str, Any]:
        out = []
        for name in self.vault.names():
            man = self.vault.manifest(name)
            pol = self.vault.policy(name)
            item: dict[str, Any] = {
                "name": name,
                "kind": man["kind"],
                "description": pol.description,
                "tools": [t for t, ok in (("run_sql", pol.allow_sql), ("run_python", pol.allow_python)) if ok],
            }
            out.append(item)
        payload = {"datasets": out}
        gate = self._gate().inspect(payload, [], check_protected=False, origin="dataset list")
        if not gate.allowed:
            return self._deny("list_datasets", {}, gate.reasons)
        return self._release("list_datasets", {}, gate, gate.payload)

    def describe_dataset(self, name: str) -> dict[str, Any]:
        args = {"name": name}
        try:
            p = self._prep(name)
        except SealboxError as exc:
            return self._error("describe_dataset", args, str(exc))
        payload = describe(p)
        gate = self._gate().inspect(payload, [name], check_protected=False, origin="description")
        if not gate.allowed:
            return self._deny("describe_dataset", args, gate.reasons)
        return self._release("describe_dataset", args, gate, gate.payload)

    def budget_status(self) -> dict[str, Any]:
        status = self.budget.status()
        self._audit("budget_status", {}, "ok")
        return {"status": "ok", "session": self.session_id, **status}

    def run_sql(self, sql: str, max_rows: int = PAGE_ROWS) -> dict[str, Any]:
        args = {"sql": sql[:4000], "max_rows": max_rows}
        admit = self.budget.admit_query()
        if not admit.ok:
            return self._deny("run_sql", args, [Reason("leakage_budget", admit.reason)])
        try:
            preps = [self._prep(n) for n in self.vault.names()]
        except SealboxError as exc:
            return self._error("run_sql", args, str(exc))
        sql_preps = [p for p in preps if p.policy.allow_sql]
        with self._lock:
            key = tuple(p.name for p in sql_preps)
            if self._conn is None or self._conn_key != key:
                if self._conn is not None:
                    self._conn.close()
                self._conn = locked_connection(sql_preps)
                self._conn_key = key
            conn = self._conn.cursor()
        try:
            policies = {p.name: p.policy for p in sql_preps}
            bounds = {(p.name, c): b for p in sql_preps for c, b in p.bounds.items()}
            guard = SqlGuard(conn, policies, {p.name: p.columns for p in sql_preps}, bounds)
            try:
                plan = guard.plan(sql)
            except QueryRejected as rej:
                self.budget.record_error()
                return self._deny("run_sql", args, [Reason(rej.rule, rej.detail)], _SQL_HINT)
            try:
                cur = conn.execute(plan.sql)
                columns = [d[0] for d in cur.description]
                raw = cur.fetchmany(10_001)
            except Exception as exc:
                touches_sensitive = any(self._sensitive(t) for t in plan.tables)
                if touches_sensitive and _data_dependent(exc):
                    self.budget.record_data_error()
                    self._audit("run_sql", args, "error", message=type(exc).__name__, data_dependent=True)
                    return {
                        "status": "error",
                        "error": "query failed at runtime on protected data (details "
                        "withheld; data-dependent errors are capped per session)",
                        "budget": self._short_budget(),
                    }
                msg = self._sanitize_error(f"{type(exc).__name__}: {exc}", sql, plan.tables)
                return self._error("run_sql", args, msg)
        finally:
            conn.close()

        prep_by = {p.name: p for p in sql_preps}
        k = max(prep_by[t].policy.k_min for t in plan.tables)
        notes: list[str] = []
        eps = 0.0
        rows = [list(r) for r in raw]
        if len(rows) > 10_000:
            return self._deny("run_sql", args, [Reason("output_rows", "result exceeds 10000 rows")], _SQL_HINT)
        if plan.n_support:
            out_cols = columns[: -plan.n_support]
            kept = []
            for r in rows:
                support = min(int(x or 0) for x in r[-plan.n_support :])
                if support >= k:
                    kept.append([*r[: -plan.n_support], support])
            suppressed = len(rows) - len(kept)
            if suppressed:
                notes.append(f"{suppressed} group(s) suppressed: fewer than k={k} contributing entities")
            if rows and not kept:
                return self._deny(
                    "run_sql",
                    args,
                    [Reason("k_anonymity", f"every result group has fewer than k={k} contributing entities")],
                    "Broaden the groups (fewer GROUP BY keys, wider filters).",
                )
            dp_on = any(prep_by[t].policy.dp.enabled for t in plan.sensitive_tables)
            if dp_on:
                eps_q = min(prep_by[t].policy.dp.epsilon_per_query for t in plan.sensitive_tables)
                derived = [o for o in plan.outputs if o.sensitive and o.agg == "derived"]
                if derived:
                    self.budget.record_error()
                    return self._deny(
                        "run_sql",
                        args,
                        [
                            Reason(
                                "dp_uncalibrated",
                                f"output column {derived[0].index + 1} combines aggregates of protected "
                                "columns in a way differential-privacy noise cannot be calibrated for",
                            )
                        ],
                        "Select plain COUNT/SUM/AVG columns and combine them yourself, or ask the operator to "
                        "disable dp for this dataset.",
                    )
                noised = [o for o in plan.outputs if o.sensitive and o.agg is not None]
                need = eps_q * len(noised)
                ok = self.budget.can_spend_epsilon(need)
                if not ok.ok:
                    return self._deny("run_sql", args, [Reason("privacy_budget", ok.reason)])
                for o in noised:
                    for r in kept:
                        r[o.index] = self._noise(r[o.index], o.agg, o.column, bounds, r[-1], eps_q)
                eps = need
                if noised:
                    notes.append(
                        f"differential privacy: Laplace noise added to {len(noised)} column(s) "
                        f"(epsilon {eps_q} each, {need:.2f} charged)"
                    )
            rows = [r[:-1] for r in kept]
            columns = out_cols
        result = StoredResult(
            columns=columns, rows=[[jsonable(v) for v in r] for r in rows], datasets=plan.tables, released_until=0
        )
        return self._release_table("run_sql", args, result, max_rows, notes, eps, check_protected=False)

    def _noise(
        self,
        value: Any,
        agg: str | None,
        column: tuple[str, str] | None,
        bounds: dict[tuple[str, str], tuple[float, float]],
        support: int,
        eps: float,
    ) -> Any:
        if value is None:
            return None
        v = float(value)
        if agg in ("count", "count_star", "count_if", "approx_count_distinct"):
            return max(0, round(v + laplace(1.0 / eps)))
        lo, hi = bounds.get(column, (0.0, 0.0)) if column else (0.0, 0.0)
        # sum: one entity can move it by max|value|; avg: by range / n
        sens = max(abs(lo), abs(hi), 1.0) if agg == "sum" else max(hi - lo, 1e-9) / max(support, 1)
        return float(f"{v + laplace(sens / eps):.6g}")

    def _release_table(
        self,
        tool: str,
        args: dict[str, Any],
        result: StoredResult,
        max_rows: int,
        notes: list[str],
        eps: float,
        check_protected: bool,
    ) -> dict[str, Any]:
        gate = self._gate().inspect(
            {"columns": result.columns, "rows": result.rows},
            result.datasets,
            check_protected=check_protected,
            max_rows=len(result.rows),
            origin="result",
        )
        if not gate.allowed:
            return self._deny(tool, args, gate.reasons, _SQL_HINT if tool == "run_sql" else "")
        result.rows = gate.payload["rows"]
        result.columns = gate.payload["columns"]
        rid = "r_" + secrets.token_hex(6)
        limit = max(1, min(int(max_rows), PAGE_ROWS))
        page = result.rows[:limit]
        result.released_until = len(page)
        with self._lock:
            self._results[rid] = result
        body: dict[str, Any] = {
            "result_id": rid,
            "columns": result.columns,
            "rows": page,
            "total_rows": len(result.rows),
            "returned_rows": len(page),
        }
        all_notes = notes + gate.notes
        if all_notes:
            body["notes"] = all_notes
        gate.notes = []
        return self._release(tool, args, gate, body, eps)

    def get_result(self, result_id: str, offset: int = 0, limit: int = PAGE_ROWS) -> dict[str, Any]:
        args = {"result_id": result_id, "offset": offset, "limit": limit}
        with self._lock:
            res = self._results.get(result_id)
        if res is None:
            return self._error("get_result", args, "unknown result_id")
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), PAGE_ROWS))
        page = res.rows[offset : offset + limit]
        gate = GateResult(True, payload=None)
        return self._release(
            "get_result",
            args,
            gate,
            {
                "result_id": result_id,
                "columns": res.columns,
                "rows": page,
                "offset": offset,
                "total_rows": len(res.rows),
            },
        )

    def run_python(self, code: str, datasets: list[str] | None = None) -> dict[str, Any]:
        started = time.monotonic()
        args = {"code": code[:8000], "datasets": datasets}
        try:
            return self._run_python(code, datasets, args)
        finally:
            pad_latency(started, self.config.python_time_quantum_s)

    def _run_python(self, code: str, datasets: list[str] | None, args: dict[str, Any]) -> dict[str, Any]:
        admit = self.budget.admit_query()
        if not admit.ok:
            return self._deny("run_python", args, [Reason("leakage_budget", admit.reason)])
        names = datasets or [n for n in self.vault.names() if self.vault.policy(n).allow_python]
        try:
            preps = [self._prep(n) for n in names]
        except SealboxError as exc:
            return self._error("run_python", args, str(exc))
        blocked = [p.name for p in preps if not p.policy.allow_python]
        if blocked:
            return self._deny(
                "run_python", args, [Reason("tool_disabled", f"run_python is disabled for: {', '.join(blocked)}")]
            )
        tables = {
            p.name: {"columns": p.columns, "rows": [[jsonable(v) for v in r] for r in p.rows]}
            for p in preps
            if p.kind == "table"
        }
        docs = {p.name: p.docs for p in preps if p.kind == "docs"}
        hide = [self.vault.root]
        if os.environ.get("SEALBOX_KEYFILE"):
            hide.append(Path(os.environ["SEALBOX_KEYFILE"]).parent)
        res = run_python(
            code,
            tables,
            docs,
            timeout_s=self.config.python_timeout_s,
            memory_mb=self.config.python_memory_mb,
            hide_paths=hide,
            require_network_isolation=self.config.require_network_isolation,
        )
        if not res.ok:
            msg = f"{res.error_type}" + (f" at line {res.lineno}" if res.lineno else "")
            if res.message:
                detail = self._sanitize_error(res.message, code, names)
                msg += f": {detail}"
            out = self._error("run_python", args, msg)
            if res.warnings:
                out["warnings"] = res.warnings
            return out
        payload: dict[str, Any] = {"result": jsonable(res.result)}
        if res.stdout:
            payload["stdout"] = res.stdout
        rows_hint = len(res.result) if isinstance(res.result, (list, dict)) else None
        gate = self._gate().inspect(
            payload, names, check_protected=True, max_rows=rows_hint, origin="python result", allow_tokens=False
        )
        if not gate.allowed:
            return self._deny(
                "run_python", args, gate.reasons, "Return aggregates/findings ABOUT the data, not the data itself."
            )
        body = dict(gate.payload)
        body["isolation"] = res.isolation
        if res.warnings:
            body["warnings"] = res.warnings
        body["notes"] = [
            "code-derived results are checked for raw values and content leaks, but k-anonymity "
            "and differential privacy apply to run_sql only"
        ]
        return self._release("run_python", args, gate, body)

    # ------------------------------------------------------------------ owner-side only
    def token_map(self, dataset: str) -> dict[str, str]:
        """Token -> raw value. NEVER exposed as an MCP tool."""
        return dict(self._prep(dataset).token_map)

    def public_columns(self, dataset: str) -> list[str]:
        p = self._prep(dataset)
        return [c for c in p.columns if p.policy.tag(c) == PUBLIC]


def _data_dependent(exc: BaseException) -> bool:
    """Conversion/overflow/invalid-input errors depend on cell values; binder/parser errors do not."""
    names = {c.__name__ for c in type(exc).__mro__}
    static = {"BinderException", "CatalogException", "ParserException", "SyntaxException", "NotImplementedException"}
    return not names & static


_SQL_HINT = (
    "Write one SELECT over vault tables with aggregates (COUNT/SUM/AVG/STDDEV) and GROUP BY; no "
    "subqueries/CTEs/UNION/window functions; aggregate_only columns only inside aggregates; groups "
    "must have at least k contributing entities. Call describe_dataset for column tags."
)
