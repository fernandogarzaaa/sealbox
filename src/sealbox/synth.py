"""Policy-aware dataset descriptions with SYNTHETIC sample rows (never real rows)."""

from __future__ import annotations

import datetime as dt
import math
import random
import string
from typing import Any

from .compute import Prepared
from .policy import AGGREGATE_ONLY, DENY, PUBLIC, TOKENIZE
from .tokenizer import label_for


def _sig(x: float | None, digits: int = 2) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{digits}g}")


def column_summary(p: Prepared) -> list[dict[str, Any]]:
    out = []
    k = p.policy.k_min
    for info in p.meta.get("columns", []):
        name = info["name"]
        tag = p.policy.tag(name)
        entry: dict[str, Any] = {"name": name, "type": info["type"], "tag": tag}
        if tag == DENY:
            entry["note"] = "hidden: not available to any computation"
            out.append(entry)
            continue
        entry["null_fraction"] = _sig(info["n_null"] / info["n_rows"], 2) if info["n_rows"] else 0.0
        if tag == PUBLIC:
            entry["n_distinct"] = info["n_distinct"]
            if info.get("top"):
                shown = [[v, c] for v, c in info["top"] if c >= k]
                entry["top_values"] = shown
                if len(shown) < len(info["top"]):
                    entry["top_values_note"] = f"values with fewer than k={k} rows are hidden"
            for key in ("min", "max", "mean"):
                if info.get(key) is not None:
                    entry[key] = info[key]
        elif tag == TOKENIZE:
            entry["n_distinct"] = info["n_distinct"]
            entry["token_format"] = f"<{label_for(name)}_xxxxxxxx>"
            entry["note"] = "values are deterministic pseudonymous tokens; joins/GROUP BY work, COUNT only"
        elif tag == AGGREGATE_ONLY:
            if info.get("is_numeric"):
                entry.update(
                    mean=_sig(info.get("mean"), 3),
                    std=_sig(info.get("std"), 3),
                    approx_range=[_sig(info.get("min")), _sig(info.get("max"))],
                )
            else:
                entry["avg_length"] = _sig(info.get("avg_len"), 2)
            entry["n_distinct_approx"] = _sig(float(info["n_distinct"]), 1)
            entry["note"] = "use only inside COUNT/SUM/AVG/STDDEV-style aggregates"
        out.append(entry)
    return out


def synthetic_rows(p: Prepared, n: int = 5, seed: int = 0) -> list[dict[str, Any]]:
    """Fake rows that match the schema shape. Values are drawn from coarse stats, not copied."""
    rng = random.Random(seed)  # noqa: S311 - synthetic data, not security
    infos = {i["name"]: i for i in p.meta.get("columns", [])}
    rows = []
    for _ in range(n):
        row: dict[str, Any] = {}
        for col in p.columns:
            info = infos.get(col, {})
            tag = p.policy.tag(col)
            typ = str(info.get("type", "VARCHAR")).upper()
            if tag == TOKENIZE:
                row[col] = f"<{label_for(col)}_{rng.getrandbits(32):08x}>"
            elif info.get("is_numeric"):
                mean = info.get("mean") or 0.0
                std = info.get("std") or abs(mean) * 0.2 or 1.0
                v = rng.gauss(mean, std)
                if tag == PUBLIC and info.get("min") is not None:
                    v = min(max(v, info["min"]), info["max"])
                row[col] = round(v) if "INT" in typ else round(v, 2)
            elif typ.startswith("DATE") or typ.startswith("TIMESTAMP"):
                base = dt.date(2020, 1, 1) + dt.timedelta(days=rng.randrange(0, 365 * 5))
                row[col] = base.isoformat()
            elif typ == "BOOLEAN":
                row[col] = rng.random() < 0.5
            elif tag == PUBLIC and info.get("top"):
                choices = [v for v, c in info["top"] if c >= p.policy.k_min] or ["<category>"]
                row[col] = rng.choice(choices)
            else:
                length = max(4, min(60, int(info.get("avg_len") or 8)))
                row[col] = "".join(rng.choice(string.ascii_lowercase + " ") for _ in range(length)).strip()
        rows.append(row)
    return rows


def describe(p: Prepared) -> dict[str, Any]:
    pol = p.policy
    base: dict[str, Any] = {
        "name": p.name,
        "kind": p.kind,
        "description": pol.description,
        "policy": {
            "k_min": pol.k_min,
            "allow_sql": pol.allow_sql,
            "allow_python": pol.allow_python,
            "row_level_public_only": pol.row_level_public,
            "differential_privacy": {"enabled": pol.dp.enabled, "epsilon_per_query": pol.dp.epsilon_per_query},
            "max_output_rows": pol.max_output_rows,
            "max_output_bytes": pol.max_output_bytes,
            "entity_column": pol.entity_column,
        },
    }
    if p.kind == "docs":
        exts: dict[str, int] = {}
        for path in p.docs:
            ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
            exts[ext] = exts.get(ext, 0) + 1
        base.update(
            n_files=len(p.docs),
            n_bytes=p.meta.get("n_bytes"),
            extensions=exts,
            sql_table={"name": p.name, "columns": {c: pol.tag(c) for c in p.columns}},
            usage=(
                "File contents never leave sealbox. Use run_python with docs['" + p.name + "'] "
                "(dict path -> text) to compute ABOUT the code/docs (counts, structure, findings); outputs "
                "that reproduce protected content are blocked."
            ),
            paths_sample=sorted(p.docs)[:25],
        )
        return base
    base.update(
        n_rows=p.meta.get("n_rows"),
        columns=column_summary(p),
        synthetic_sample_rows=synthetic_rows(p),
        synthetic_note="sample rows are SYNTHETIC (shape only); no real row is ever returned",
        usage=(
            "SQL: SELECT with GROUP BY over this table; groups with fewer than k contributing "
            "entities are suppressed. Python: rows('" + p.name + "') or df('" + p.name + "') in run_python."
        ),
    )
    return base
