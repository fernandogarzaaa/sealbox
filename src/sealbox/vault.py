"""Encrypted dataset vault.

Layout::

    <vault>/vault.json                  vault id + key check (no secrets)
    <vault>/sealbox.yaml                vault-wide session/budget settings
    <vault>/policies/<dataset>.yaml     per-dataset policy (operator-editable)
    <vault>/datasets/<name>/manifest.json   kind, wrapped data key (no data)
    <vault>/datasets/<name>/data.enc    AES-256-GCM(parquet bytes | JSON docs)
    <vault>/datasets/<name>/meta.enc    AES-256-GCM(schema + stats)
    <vault>/audit.jsonl                 hash-chained audit log

Plaintext only ever exists in process memory: Parquet is (de)serialized through an
in-memory filesystem, never a temp file.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import fsspec

from . import crypto
from .errors import VaultError
from .policy import DatasetPolicy, SessionConfig, default_policy, load_policy, load_session_config

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
TABULAR_EXT = {
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
}
TEXT_EXT = {
    ".py",
    ".rs",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".java",
    ".kt",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".scala",
    ".sql",
    ".sh",
    ".md",
    ".txt",
    ".rst",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".html",
    ".css",
    ".tex",
    ".csv",
    ".json",
}
MAX_TEXT_FILE = 2_000_000
NUMERIC_TYPES = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
)

_MEMFS = fsspec.filesystem("memory")

DEFAULT_SEALBOX_YAML = """\
# Vault-wide settings. The leakage budget applies per server process (one agent session).
session:
  epsilon_budget: 30.0        # total differential-privacy epsilon per session
  bytes_budget: 262144        # total bytes of tool output released per session
  max_queries: 500
  max_errors: 25              # errors are a 1-bit oracle; cap them
  max_data_errors: 3          # data-dependent SQL errors (cast/overflow) on sensitive tables
  python_timeout_s: 20
  python_memory_mb: 1024
  python_time_quantum_s: 0.5  # run_python latency padded to multiples of this
  require_network_isolation: false   # true: refuse run_python if no network namespace
"""


def is_numeric_type(t: str) -> bool:
    return t.upper().startswith(NUMERIC_TYPES) or t.upper().startswith("DECIMAL")


def safe_name(raw: str) -> str:
    name = re.sub(r"[^a-z0-9_]", "_", raw.lower()).strip("_")
    if not name or not name[0].isalpha():
        name = "d_" + name
    return name[:63]


@dataclass
class Dataset:
    name: str
    kind: str  # "table" | "docs"
    meta: dict[str, Any]
    payload: bytes  # parquet bytes (table) or JSON (docs); plaintext, memory only

    @property
    def docs(self) -> dict[str, str]:
        if self.kind != "docs":
            raise VaultError(f"{self.name} is not a docs dataset")
        result: dict[str, str] = json.loads(self.payload)
        return result


def parquet_to_conn(conn: duckdb.DuckDBPyConnection, table: str, payload: bytes) -> None:
    path = f"memory://sealbox/{uuid.uuid4().hex}.parquet"
    _MEMFS.pipe(path, payload)
    with contextlib.suppress(Exception):  # already registered on this connection
        conn.register_filesystem(_MEMFS)
    try:
        conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM read_parquet(?)', [path])
    finally:
        _MEMFS.rm(path)


def conn_to_parquet(conn: duckdb.DuckDBPyConnection, table: str) -> bytes:
    path = f"memory://sealbox/{uuid.uuid4().hex}.parquet"
    with contextlib.suppress(Exception):
        conn.register_filesystem(_MEMFS)
    conn.execute(f"COPY \"{table}\" TO '{path}' (FORMAT PARQUET)")
    try:
        data: bytes = _MEMFS.cat(path)
    finally:
        _MEMFS.rm(path)
    return data


def table_stats(conn: duckdb.DuckDBPyConnection, table: str) -> dict[str, Any]:
    cols = conn.execute(f'DESCRIBE "{table}"').fetchall()
    n_rows = int(conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])  # type: ignore[index]
    out = []
    for name, ctype, *_ in cols:
        q = f'"{name}"'
        n_null, n_distinct = conn.execute(
            f'SELECT count(*) - count({q}), approx_count_distinct({q}) FROM "{table}"'
        ).fetchone()  # type: ignore[misc]
        info: dict[str, Any] = {
            "name": name,
            "type": ctype,
            "n_rows": n_rows,
            "n_null": int(n_null),
            "n_distinct": int(min(n_distinct, n_rows)),
            "is_numeric": is_numeric_type(ctype),
            "avg_len": 0.0,
        }
        if info["is_numeric"]:
            mn, mx, mean, sd = conn.execute(
                f'SELECT min({q})::DOUBLE, max({q})::DOUBLE, avg({q})::DOUBLE, stddev_samp({q})::DOUBLE FROM "{table}"'
            ).fetchone()  # type: ignore[misc]
            info.update(min=mn, max=mx, mean=mean, std=sd)
        else:
            avg_len = conn.execute(f'SELECT avg(length(CAST({q} AS VARCHAR))) FROM "{table}"').fetchone()[0]  # type: ignore[index]
            info["avg_len"] = float(avg_len or 0.0)
            top = conn.execute(
                f'SELECT CAST({q} AS VARCHAR) v, count(*) c FROM "{table}" WHERE {q} IS NOT NULL '
                f"GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 10"
            ).fetchall()
            info["top"] = [[v, int(c)] for v, c in top]
            if ctype.upper() in ("DATE", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE"):
                mn, mx = conn.execute(f'SELECT min({q})::VARCHAR, max({q})::VARCHAR FROM "{table}"').fetchone()  # type: ignore[misc]
                info.update(min=mn, max=mx)
        out.append(info)
    return {"n_rows": n_rows, "columns": out}


class Vault:
    def __init__(self, root: Path, keys: crypto.KeyProvider) -> None:
        self.root = Path(root)
        self.keys = keys
        meta_path = self.root / "vault.json"
        if not meta_path.exists():
            raise VaultError(f"{self.root} is not a sealbox vault (run `sealbox init`)")
        self.meta = json.loads(meta_path.read_text())
        check = bytes.fromhex(self.meta["key_check"])
        try:
            crypto.decrypt(keys.derive("key-check"), check, b"sealbox-key-check")
        except crypto.CryptoError as exc:
            raise VaultError("master key does not match this vault") from exc

    # ------------------------------------------------------------------ creation
    @classmethod
    def init(cls, root: Path, keys: crypto.KeyProvider) -> Vault:
        root = Path(root)
        if (root / "vault.json").exists():
            raise VaultError(f"vault already exists at {root}")
        for sub in ("datasets", "policies"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        check = crypto.encrypt(keys.derive("key-check"), b"ok", b"sealbox-key-check")
        (root / "vault.json").write_text(
            json.dumps(
                {"version": 1, "vault_id": uuid.uuid4().hex, "created": int(time.time()), "key_check": check.hex()},
                indent=2,
            )
        )
        if not (root / "sealbox.yaml").exists():
            (root / "sealbox.yaml").write_text(DEFAULT_SEALBOX_YAML)
        return cls(root, keys)

    # ------------------------------------------------------------------ paths
    def _dir(self, name: str) -> Path:
        if not NAME_RE.match(name):
            raise VaultError(f"invalid dataset name {name!r}")
        return self.root / "datasets" / name

    def policy_path(self, name: str) -> Path:
        return self.root / "policies" / f"{name}.yaml"

    @property
    def audit_path(self) -> Path:
        return self.root / "audit.jsonl"

    def session_config(self) -> SessionConfig:
        return load_session_config(self.root / "sealbox.yaml")

    def names(self) -> list[str]:
        base = self.root / "datasets"
        return sorted(p.name for p in base.iterdir() if (p / "manifest.json").exists()) if base.exists() else []

    def manifest(self, name: str) -> dict[str, Any]:
        path = self._dir(name) / "manifest.json"
        if not path.exists():
            raise VaultError(f"unknown dataset {name!r}")
        result: dict[str, Any] = json.loads(path.read_text())
        return result

    def policy(self, name: str) -> DatasetPolicy:
        self.manifest(name)
        path = self.policy_path(name)
        if not path.exists():
            return DatasetPolicy(dataset=name)
        return load_policy(path, name)

    # ------------------------------------------------------------------ ingest
    def ingest(self, source: Path, name: str | None = None, overwrite: bool = False) -> str:
        source = Path(source)
        if not source.exists():
            raise VaultError(f"source not found: {source}")
        name = name or safe_name(source.stem if source.is_file() else source.name)
        ddir = self._dir(name)
        if ddir.exists() and not overwrite:
            raise VaultError(f"dataset {name!r} already exists (use --overwrite)")
        if source.is_dir() or source.suffix.lower() not in TABULAR_EXT:
            kind, payload, meta = "docs", *self._read_docs(source)
        else:
            kind, payload, meta = "table", *self._read_table(source)
        dek = crypto.new_data_key()
        aad = name.encode()
        ddir.mkdir(parents=True, exist_ok=True)
        (ddir / "data.enc").write_bytes(crypto.encrypt(dek, payload, aad + b":data"))
        (ddir / "meta.enc").write_bytes(crypto.encrypt(dek, json.dumps(meta).encode(), aad + b":meta"))
        (ddir / "manifest.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "kind": kind,
                    "created": int(time.time()),
                    "source_format": meta.get("format"),
                    "wrapped_dek": self.keys.wrap(dek, aad).hex(),
                },
                indent=2,
            )
        )
        ppath = self.policy_path(name)
        if not ppath.exists() or overwrite:
            pol = default_policy(name, meta.get("columns", []), kind)
            ppath.write_text(
                "# sealbox policy - review the column tags before serving.\n"
                "# tags: public | tokenize | aggregate_only | deny\n" + pol.to_yaml()
            )
        return name

    def _read_table(self, source: Path) -> tuple[bytes, dict[str, Any]]:
        fmt = TABULAR_EXT[source.suffix.lower()]
        reader = {"csv": "read_csv_auto", "parquet": "read_parquet", "json": "read_json_auto"}[fmt]
        conn = duckdb.connect()
        try:
            conn.execute(f"CREATE TABLE src AS SELECT * FROM {reader}(?)", [str(source)])
            meta = table_stats(conn, "src")
            meta["format"] = fmt
            return conn_to_parquet(conn, "src"), meta
        finally:
            conn.close()

    def _read_docs(self, source: Path) -> tuple[bytes, dict[str, Any]]:
        files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
        docs: dict[str, str] = {}
        skipped = 0
        for path in files:
            rel = path.name if source.is_file() else str(path.relative_to(source))
            if any(part.startswith(".") for part in Path(rel).parts) or path.suffix.lower() not in TEXT_EXT:
                skipped += 1
                continue
            raw = path.read_bytes()
            if len(raw) > MAX_TEXT_FILE or b"\x00" in raw[:4096]:
                skipped += 1
                continue
            docs[rel] = raw.decode("utf-8", "replace")
        if not docs:
            raise VaultError(f"no ingestible text files under {source}")
        meta = {
            "format": "docs",
            "n_files": len(docs),
            "skipped": skipped,
            "n_bytes": sum(len(v.encode()) for v in docs.values()),
            "extensions": sorted({Path(p).suffix.lower() for p in docs}),
        }
        return json.dumps(docs).encode(), meta

    # ------------------------------------------------------------------ load
    def load(self, name: str) -> Dataset:
        man = self.manifest(name)
        ddir = self._dir(name)
        aad = name.encode()
        dek = self.keys.unwrap(bytes.fromhex(man["wrapped_dek"]), aad)
        payload = crypto.decrypt(dek, (ddir / "data.enc").read_bytes(), aad + b":data")
        meta = json.loads(crypto.decrypt(dek, (ddir / "meta.enc").read_bytes(), aad + b":meta"))
        return Dataset(name=name, kind=man["kind"], meta=meta, payload=payload)
