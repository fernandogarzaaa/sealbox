"""Per-dataset policy (YAML) and vault-wide session settings, with secure defaults."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import PolicyError

# Column sensitivity tags.
PUBLIC = "public"  # may appear raw in outputs and GROUP BY
TOKENIZE = "tokenize"  # replaced by deterministic HMAC tokens in every compute view
AGGREGATE_ONLY = "aggregate_only"  # only inside COUNT/SUM/AVG/STDDEV-style aggregates
DENY = "deny"  # removed from every compute view entirely
TAGS = (PUBLIC, TOKENIZE, AGGREGATE_ONLY, DENY)

_TOKENIZE_HINTS = re.compile(
    r"(^|_)(name|first_?name|last_?name|full_?name|email|e_?mail|phone|mobile|address|street|"
    r"username|user_?id|customer_?id|patient_?id|employee_?id|account|iban|ip|device_?id)($|_)",
    re.I,
)
_DENY_HINTS = re.compile(
    r"(^|_)(ssn|social_?security|passport|password|passwd|secret|api_?key|token|credit_?card|"
    r"card_?number|cvv|dob|date_?of_?birth|national_?id|tax_?id)($|_)",
    re.I,
)
_PUBLIC_HINTS = re.compile(
    r"(^|_)(dept|department|region|country|state|city|category|type|status|team|level|grade|"
    r"segment|channel|product|year|quarter|month|gender|ext|language|path|n_lines|n_bytes)($|_)",
    re.I,
)
_ENTITY_HINTS = re.compile(r"(^|_)(person|people|user|customer|employee|patient|client|member)", re.I)


@dataclass
class DPConfig:
    enabled: bool = True
    epsilon_per_query: float = 1.0


@dataclass
class LeakConfig:
    enabled: bool = True
    shingle_tokens: int = 8  # exact n-gram size (tokens)
    max_shared_shingles: int = 2  # more shared exact shingles than this -> deny
    minhash_threshold: float = 0.55  # estimated Jaccard of structural windows -> deny
    opaque_blob_min_len: int = 64  # base64/hex runs at least this long that don't decode to text -> deny


@dataclass
class DatasetPolicy:
    dataset: str
    description: str = ""
    k_min: int = 5
    allow_sql: bool = True
    allow_python: bool = True
    row_level_public: bool = True  # non-aggregate SELECT allowed only if every referenced column is public
    max_output_rows: int = 200
    max_output_bytes: int = 32_000
    default_column: str = AGGREGATE_ONLY
    columns: dict[str, str] = field(default_factory=dict)
    entity_column: str | None = None
    pii_action: str = "redact"  # redact | deny
    injection_action: str = "strip"  # strip | deny | flag
    dp: DPConfig = field(default_factory=DPConfig)
    leak: LeakConfig = field(default_factory=LeakConfig)

    def tag(self, column: str) -> str:
        return self.columns.get(column, self.default_column)

    def validate(self) -> None:
        if self.k_min < 1:
            raise PolicyError("k_min must be >= 1")
        for col, tag in {**self.columns, "<default>": self.default_column}.items():
            if tag not in TAGS:
                raise PolicyError(f"column {col!r}: unknown tag {tag!r} (use one of {', '.join(TAGS)})")
        if self.pii_action not in ("redact", "deny"):
            raise PolicyError("pii_action must be redact or deny")
        if self.injection_action not in ("strip", "deny", "flag"):
            raise PolicyError("injection_action must be strip, deny or flag")
        if self.dp.epsilon_per_query <= 0:
            raise PolicyError("dp.epsilon_per_query must be > 0")
        if self.max_output_rows < 1 or self.max_output_bytes < 64:
            raise PolicyError("output caps too small")

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False)


@dataclass
class SessionConfig:
    """Vault-wide leakage budget for one server process (one MCP session/agent)."""

    epsilon_budget: float = 30.0
    bytes_budget: int = 262_144
    max_queries: int = 500
    max_errors: int = 25
    python_timeout_s: float = 20.0
    python_memory_mb: int = 1024
    python_time_quantum_s: float = 0.5  # pad run_python latency to multiples of this (timing channel)
    require_network_isolation: bool = False


def _merge(dc: Any, data: dict[str, Any]) -> Any:
    for key, value in data.items():
        if not hasattr(dc, key):
            raise PolicyError(f"unknown policy key: {key}")
        current = getattr(dc, key)
        if isinstance(current, (DPConfig, LeakConfig)):
            if not isinstance(value, dict):
                raise PolicyError(f"{key} must be a mapping")
            _merge(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_policy(path: Path, dataset: str) -> DatasetPolicy:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise PolicyError(f"{path}: policy must be a mapping")
    data.setdefault("dataset", dataset)
    pol = _merge(DatasetPolicy(dataset=dataset), data)
    pol.validate()
    return pol  # type: ignore[no-any-return]


def load_session_config(path: Path) -> SessionConfig:
    if not path.exists():
        return SessionConfig()
    data = yaml.safe_load(path.read_text()) or {}
    session = data.get("session", {}) if isinstance(data, dict) else {}
    return _merge(SessionConfig(), session)  # type: ignore[no-any-return]


def infer_tag(column: str, n_distinct: int, n_rows: int, is_numeric: bool, avg_len: float) -> str:
    """Secure-by-default guess for a column's tag. Operators should review the generated YAML."""
    if _DENY_HINTS.search(column):
        return DENY
    if _TOKENIZE_HINTS.search(column):
        return TOKENIZE
    if _PUBLIC_HINTS.search(column) and n_distinct <= max(50, n_rows // 10):
        return PUBLIC
    if not is_numeric and n_rows and n_distinct / n_rows > 0.5:
        # high-cardinality free text / identifiers
        return TOKENIZE if avg_len < 40 else AGGREGATE_ONLY
    if not is_numeric and avg_len >= 40:
        return AGGREGATE_ONLY  # free text is content, not a category
    if not is_numeric and n_distinct <= 20:
        return PUBLIC  # low-cardinality categorical
    return AGGREGATE_ONLY


def default_policy(dataset: str, column_info: list[dict[str, Any]], kind: str) -> DatasetPolicy:
    pol = DatasetPolicy(dataset=dataset)
    if kind == "docs":
        pol.columns = {"path": PUBLIC, "ext": PUBLIC, "n_lines": PUBLIC, "n_bytes": PUBLIC, "content": AGGREGATE_ONLY}
        pol.k_min = 1  # documents are not individuals; leak detection is the main control
        pol.dp.enabled = False
        return pol
    for info in column_info:
        pol.columns[info["name"]] = infer_tag(
            info["name"], info["n_distinct"], info["n_rows"], info["is_numeric"], info["avg_len"]
        )
        if pol.entity_column is None and pol.columns[info["name"]] == TOKENIZE and _ENTITY_HINTS.search(info["name"]):
            pol.entity_column = info["name"]
    return pol
