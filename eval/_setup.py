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


_WARM: dict[int, Engine] = {}


def fresh_engine(vault: Vault, keys: crypto.LocalKeyProvider, **cfg: object) -> Engine:
    """New session (fresh budget/results/audit session) that reuses the decrypted, read-only dataset cache.

    Prepared datasets hold no per-session state, so sharing them only skips ~2.5 s of decrypt+shape work.
    """
    cfg.setdefault("python_time_quantum_s", 0)
    eng = Engine(vault, keys, SessionConfig(**cfg))  # type: ignore[arg-type]
    warm = _WARM.get(id(vault))
    if warm is None:
        for n in vault.names():
            eng._prep(n)
        _WARM[id(vault)] = eng
    else:
        eng._prepared = dict(warm._prepared)
    return eng
