"""Build policy-shaped, locked-down compute views of decrypted datasets (in memory only)."""

from __future__ import annotations

import datetime as dt
import decimal
import math
from dataclasses import dataclass, field
from typing import Any

import duckdb

from .fingerprint import FingerprintIndex
from .policy import AGGREGATE_ONLY, DENY, PUBLIC, TOKENIZE, DatasetPolicy
from .tokenizer import Tokenizer
from .vault import Dataset, Vault, parquet_to_conn

LONG_TEXT = 40  # string cells at least this long are fingerprinted as content


def norm_value(v: Any) -> str | None:
    """Canonical form used for protected-value matching; None = too trivial to protect."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float, decimal.Decimal)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        if f == int(f) and abs(f) < 1000:
            return None
        if f != int(f) and len(f"{abs(f):.10g}".replace(".", "").strip("0")) < 3:
            return None  # e.g. 0.5, 2.5: too coincidence-prone to treat as a leak
        return f"n:{f:.10g}"
    s = str(v).strip().lower()
    if len(s) < 4:
        return None
    return "s:" + " ".join(s.split())


def jsonable(v: Any) -> Any:
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(f"{v:.10g}")
    if isinstance(v, (dt.date, dt.datetime, dt.time)):
        return v.isoformat()
    if isinstance(v, dt.timedelta):
        return v.total_seconds()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "<binary>"
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (int, str)) or v is None:
        return v
    return str(v)


@dataclass
class Prepared:
    """One dataset, decrypted and shaped by its policy; cached for the session."""

    name: str
    kind: str
    policy: DatasetPolicy
    meta: dict[str, Any]
    columns: list[str]  # visible columns (deny removed)
    rows: list[tuple[Any, ...]]  # tokenized compute view
    docs: dict[str, str] = field(default_factory=dict)
    protected: set[str] = field(default_factory=set)
    fingerprints: FingerprintIndex = field(default_factory=FingerprintIndex)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    token_map: dict[str, str] = field(default_factory=dict)
    column_types: dict[str, str] = field(default_factory=dict)


def prepare(vault: Vault, name: str, tokenizer: Tokenizer) -> Prepared:
    ds: Dataset = vault.load(name)
    policy = vault.policy(name)
    fp = FingerprintIndex(n=policy.leak.shingle_tokens)
    conn = duckdb.connect()
    try:
        if ds.kind == "docs":
            docs = ds.docs
            raw_cols = ["path", "ext", "n_lines", "n_bytes", "content"]
            raw_rows: list[tuple[Any, ...]] = [
                (p, ("." + p.rsplit(".", 1)[-1].lower()) if "." in p else "", c.count("\n") + 1, len(c.encode()), c)
                for p, c in docs.items()
            ]
            types = {
                "path": "VARCHAR",
                "ext": "VARCHAR",
                "n_lines": "BIGINT",
                "n_bytes": "BIGINT",
                "content": "VARCHAR",
            }
            for text in docs.values():
                fp.add_text(text)
        else:
            docs = {}
            parquet_to_conn(conn, "src", ds.payload)
            desc = conn.execute('DESCRIBE "src"').fetchall()
            raw_cols = [d[0] for d in desc]
            types = {d[0]: d[1] for d in desc}
            raw_rows = conn.execute('SELECT * FROM "src"').fetchall()
    finally:
        conn.close()

    visible = [c for c in raw_cols if policy.tag(c) != DENY]
    idx = {c: i for i, c in enumerate(raw_cols)}
    protected: set[str] = set()
    token_map: dict[str, str] = {}
    rows: list[tuple[Any, ...]] = []
    for row in raw_rows:
        out = []
        for c in raw_cols:
            v = row[idx[c]]
            tag = policy.tag(c)
            if tag != PUBLIC:
                nv = norm_value(v)
                if nv is not None:
                    protected.add(nv)
                if isinstance(v, str) and len(v) >= LONG_TEXT and ds.kind != "docs":
                    fp.add_text(v)
            if tag == DENY:
                continue
            if tag == TOKENIZE and v is not None:
                tok = tokenizer.token(c, v)
                assert tok is not None
                token_map[tok] = str(v)
                v = tok
            out.append(v)
        rows.append(tuple(out))
    bounds: dict[str, tuple[float, float]] = {}
    for info in ds.meta.get("columns", []):
        if info.get("is_numeric") and info.get("min") is not None:
            bounds[info["name"]] = (float(info["min"]), float(info["max"]))
    if ds.kind == "docs":
        n_bytes = [r[3] for r in raw_rows]
        n_lines = [r[2] for r in raw_rows]
        bounds = {"n_bytes": (min(n_bytes), max(n_bytes)), "n_lines": (min(n_lines), max(n_lines))}
    vis_types = {c: ("VARCHAR" if policy.tag(c) == TOKENIZE else types[c]) for c in visible}
    return Prepared(
        name=name,
        kind=ds.kind,
        policy=policy,
        meta=ds.meta,
        columns=visible,
        rows=rows,
        docs=docs,
        protected=protected,
        fingerprints=fp,
        bounds=bounds,
        token_map=token_map,
        column_types=vis_types,
    )


def locked_connection(prepared: list[Prepared]) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with only the policy-shaped tables, external access disabled and config locked."""
    conn = duckdb.connect(":memory:", config={"threads": 2, "memory_limit": "1GB"})
    for p in prepared:
        cols = ", ".join(f'"{c}" {p.column_types[c]}' for c in p.columns)
        conn.execute(f'CREATE TABLE "{p.name}" ({cols})')
        if p.rows:
            placeholders = ", ".join("?" for _ in p.columns)
            conn.executemany(f'INSERT INTO "{p.name}" VALUES ({placeholders})', [list(r) for r in p.rows])
    conn.execute("SET enable_external_access = false")
    conn.execute("SET autoinstall_known_extensions = false")
    conn.execute("SET autoload_known_extensions = false")
    conn.execute("SET lock_configuration = true")
    return conn


def tag_summary(p: Prepared) -> dict[str, str]:
    return {c: p.policy.tag(c) for c in p.columns}


__all__ = ["AGGREGATE_ONLY", "Prepared", "jsonable", "locked_connection", "norm_value", "prepare", "tag_summary"]
