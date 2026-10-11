from __future__ import annotations

from pathlib import Path

import pytest

from sealbox.errors import PolicyError
from sealbox.policy import DatasetPolicy, SessionConfig, default_policy, infer_tag, load_policy, load_session_config


@pytest.mark.parametrize(
    ("col", "nd", "n", "num", "alen", "tag"),
    [
        ("ssn", 100, 100, False, 11, "deny"),
        ("api_key", 100, 100, False, 30, "deny"),
        ("email", 100, 100, False, 20, "tokenize"),
        ("customer_id", 100, 100, False, 6, "tokenize"),
        ("department", 5, 100, False, 6, "public"),
        ("region", 80, 100, False, 6, "tokenize"),  # high-cardinality 'public-sounding' column is not public
        ("salary", 90, 100, True, 0, "aggregate_only"),
        ("comment", 90, 100, False, 80, "aggregate_only"),
        ("code", 90, 100, False, 8, "tokenize"),
        ("color", 4, 100, False, 5, "public"),
        ("description", 3, 100, False, 60, "aggregate_only"),
        ("score", 3, 100, True, 0, "aggregate_only"),
    ],
)
def test_infer_tag(col: str, nd: int, n: int, num: bool, alen: float, tag: str) -> None:
    assert infer_tag(col, nd, n, num, alen) == tag


def test_load_and_validate(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text("k_min: 10\ncolumns: {a: public}\ndp: {epsilon_per_query: 0.5}\n")
    pol = load_policy(p, "x")
    assert pol.k_min == 10 and pol.tag("a") == "public" and pol.tag("zzz") == "aggregate_only"
    assert pol.dp.epsilon_per_query == 0.5 and pol.dp.enabled is True
    assert "k_min: 10" in pol.to_yaml()


@pytest.mark.parametrize(
    "body",
    [
        "k_min: 0\n",
        "columns: {a: secret}\n",
        "pii_action: ignore\n",
        "injection_action: allow\n",
        "dp: {epsilon_per_query: 0}\n",
        "max_output_rows: 0\n",
        "max_output_bytes: 10\n",
        "bogus: 1\n",
        "dp: 3\n",
        "- a\n- b\n",
        "default_column: nope\n",
    ],
)
def test_invalid_policies(tmp_path: Path, body: str) -> None:
    p = tmp_path / "x.yaml"
    p.write_text(body)
    with pytest.raises(PolicyError):
        load_policy(p, "x")


def test_session_config(tmp_path: Path) -> None:
    assert load_session_config(tmp_path / "none.yaml") == SessionConfig()
    p = tmp_path / "s.yaml"
    p.write_text("session:\n  bytes_budget: 100\n")
    assert load_session_config(p).bytes_budget == 100
    p.write_text("session:\n  nope: 1\n")
    with pytest.raises(PolicyError):
        load_session_config(p)


def test_default_policy_docs_and_entity() -> None:
    d = default_policy("repo", [], "docs")
    assert d.columns["content"] == "aggregate_only" and d.columns["path"] == "public" and d.dp.enabled is False
    t = default_policy(
        "t",
        [
            {"name": "email", "n_distinct": 9, "n_rows": 9, "is_numeric": False, "avg_len": 9},
            {"name": "customer_id", "n_distinct": 9, "n_rows": 9, "is_numeric": False, "avg_len": 4},
        ],
        "table",
    )
    assert t.entity_column == "customer_id"
    assert DatasetPolicy(dataset="z").tag("q") == "aggregate_only"
