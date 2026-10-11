from __future__ import annotations

import json
from pathlib import Path

import pytest

from sealbox import crypto
from sealbox.errors import VaultError
from sealbox.vault import Vault, safe_name


def test_layout_and_no_plaintext_at_rest(vault: Vault) -> None:
    assert vault.names() == ["people", "secret_repo"]
    for name in vault.names():
        ddir = vault.root / "datasets" / name
        assert {p.name for p in ddir.iterdir()} == {"manifest.json", "data.enc", "meta.enc"}
    blob = b"".join(p.read_bytes() for p in (vault.root / "datasets").rglob("*") if p.is_file())
    for needle in (b"Person Number", b"proprietary_score", b"LedgerReconciler", b"45-6789", b"salary"):
        assert needle not in blob
    man = vault.manifest("people")
    assert man["kind"] == "table" and "wrapped_dek" in man
    assert vault.manifest("secret_repo")["kind"] == "docs"


def test_load_roundtrip(vault: Vault) -> None:
    ds = vault.load("people")
    assert ds.meta["n_rows"] == 60
    cols = {c["name"]: c for c in ds.meta["columns"]}
    assert cols["salary"]["is_numeric"] is True
    assert cols["dept"]["n_distinct"] == 4
    docs = vault.load("secret_repo").docs
    assert set(docs) == {"scoring.py", "README.md"}
    with pytest.raises(VaultError):
        _ = ds.docs


def test_default_policy_tags(vault: Vault) -> None:
    pol = vault.policy("people")
    assert pol.columns == {
        "person_id": "tokenize",
        "name": "tokenize",
        "dept": "public",
        "salary": "aggregate_only",
        "notes": "aggregate_only",
        "ssn": "deny",
    }
    assert pol.entity_column == "person_id"
    docs = vault.policy("secret_repo")
    assert docs.columns["content"] == "aggregate_only" and docs.k_min == 1


def test_wrong_key_rejected(vault: Vault) -> None:
    with pytest.raises(VaultError, match="does not match"):
        Vault(vault.root, crypto.LocalKeyProvider(crypto.new_data_key()))


def test_swapped_ciphertext_detected(vault: Vault) -> None:
    a = vault.root / "datasets" / "people" / "meta.enc"
    b = vault.root / "datasets" / "people" / "data.enc"
    a_bytes = a.read_bytes()
    a.write_bytes(b.read_bytes())
    b.write_bytes(a_bytes)
    with pytest.raises(crypto.CryptoError):
        vault.load("people")


def test_errors(tmp_path: Path, keys: crypto.LocalKeyProvider, vault: Vault) -> None:
    with pytest.raises(VaultError, match="not a sealbox vault"):
        Vault(tmp_path / "nothing", keys)
    with pytest.raises(VaultError, match="already exists"):
        Vault.init(vault.root, keys)
    with pytest.raises(VaultError, match="source not found"):
        vault.ingest(tmp_path / "missing.csv")
    with pytest.raises(VaultError, match="unknown dataset"):
        vault.manifest("nope")
    with pytest.raises(VaultError, match="invalid dataset name"):
        vault.manifest("../etc")
    with pytest.raises(VaultError, match="already exists"):
        vault.ingest(tmp_path / "people.csv")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "bin.dat").write_bytes(b"\x00\x01")
    with pytest.raises(VaultError, match="no ingestible"):
        vault.ingest(empty)


def test_overwrite_and_formats(tmp_path: Path, vault: Vault) -> None:
    rows = [{"region": "NA", "revenue": 10.5}, {"region": "EU", "revenue": 3.25}]
    (tmp_path / "r.json").write_text(json.dumps(rows))
    assert vault.ingest(tmp_path / "r.json") == "r"
    assert vault.load("r").meta["n_rows"] == 2
    assert vault.ingest(tmp_path / "r.json", overwrite=True) == "r"
    pq = tmp_path / "p.parquet"
    import duckdb

    duckdb.sql(f"COPY (SELECT 1 AS a, 'x' AS b) TO '{pq}' (FORMAT PARQUET)")
    assert vault.ingest(pq, name="pq_data") == "pq_data"
    assert vault.load("pq_data").meta["format"] == "parquet"
    single = tmp_path / "notes.txt"
    single.write_text("hello world")
    assert vault.ingest(single) == "notes"
    assert vault.load("notes").docs == {"notes.txt": "hello world"}


def test_hidden_and_binary_files_skipped(tmp_path: Path, vault: Vault) -> None:
    d = tmp_path / "repo2"
    (d / ".git").mkdir(parents=True)
    (d / ".git" / "config").write_text("secret")
    (d / "a.py").write_text("x = 1\n")
    (d / "img.png").write_bytes(b"\x89PNG")
    vault.ingest(d)
    ds = vault.load("repo2")
    assert ds.docs == {"a.py": "x = 1\n"}
    assert ds.meta["skipped"] == 2


@pytest.mark.parametrize(
    ("raw", "expected"), [("My Data-Set", "my_data_set"), ("2024 sales", "d_2024_sales"), ("___", "d_")]
)
def test_safe_name(raw: str, expected: str) -> None:
    assert safe_name(raw) == expected


def test_session_config_from_yaml(vault: Vault) -> None:
    cfg = vault.session_config()
    assert cfg.epsilon_budget == 30.0 and cfg.bytes_budget == 262144 and cfg.max_errors == 25
    (vault.root / "sealbox.yaml").write_text("session:\n  max_queries: 7\n")
    assert vault.session_config().max_queries == 7
