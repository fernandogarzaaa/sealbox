"""Shared helper: build a throwaway demo vault (fresh random key) for evals."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from sealbox import crypto
from sealbox.engine import Engine
from sealbox.policy import SessionConfig
from sealbox.vault import Vault

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo_data"


def demo_vault() -> tuple[Vault, crypto.LocalKeyProvider, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="sealbox-eval-"))
    keys = crypto.LocalKeyProvider(os.urandom(32))
    vault = Vault.init(tmp / "vault", keys)
    for p in ("employees.csv", "sales.csv", "acme_engine"):
        vault.ingest(DEMO / p)
    return vault, keys, tmp


def fresh_engine(vault: Vault, keys: crypto.LocalKeyProvider, **cfg: object) -> Engine:
    cfg.setdefault("python_time_quantum_s", 0)
    return Engine(vault, keys, SessionConfig(**cfg))  # type: ignore[arg-type]
