from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from sealbox import crypto


def test_roundtrip_and_format() -> None:
    key = crypto.new_data_key()
    blob = crypto.encrypt(key, b"secret", b"aad")
    assert blob.startswith(b"SBX1")
    assert len(blob) == 4 + 12 + len(b"secret") + 16
    assert b"secret" not in blob
    assert crypto.decrypt(key, blob, b"aad") == b"secret"


def test_nonce_is_random() -> None:
    key = crypto.new_data_key()
    assert crypto.encrypt(key, b"x", b"") != crypto.encrypt(key, b"x", b"")


@pytest.mark.parametrize("mutate", ["aad", "key", "tamper", "magic", "short"])
def test_decrypt_failures(mutate: str) -> None:
    key = crypto.new_data_key()
    blob = crypto.encrypt(key, b"payload-data", b"ds:data")
    aad = b"ds:data"
    if mutate == "aad":
        aad = b"other:data"
    elif mutate == "key":
        key = crypto.new_data_key()
    elif mutate == "tamper":
        blob = blob[:-1] + bytes([blob[-1] ^ 1])
    elif mutate == "magic":
        blob = b"XXXX" + blob[4:]
    elif mutate == "short":
        blob = blob[:20]
    with pytest.raises(crypto.CryptoError):
        crypto.decrypt(key, blob, aad)


def test_key_length_enforced() -> None:
    with pytest.raises(crypto.CryptoError):
        crypto.encrypt(b"short", b"x", b"")
    with pytest.raises(crypto.CryptoError):
        crypto.decrypt(b"short", b"SBX1" + b"0" * 40, b"")
    with pytest.raises(crypto.CryptoError):
        crypto.LocalKeyProvider(b"x" * 31)


def test_wrap_unwrap_binds_aad() -> None:
    kp = crypto.LocalKeyProvider(os.urandom(32))
    dek = crypto.new_data_key()
    w = kp.wrap(dek, b"employees")
    assert kp.unwrap(w, b"employees") == dek
    with pytest.raises(crypto.CryptoError):
        kp.unwrap(w, b"sales")
    assert len(kp.fingerprint()) == 16


def test_derive_is_deterministic_and_separated() -> None:
    m = os.urandom(32)
    assert crypto.derive_key(m, "a") == crypto.derive_key(m, "a")
    assert crypto.derive_key(m, "a") != crypto.derive_key(m, "b")
    assert len(crypto.derive_key(m, "a")) == 32


def test_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = os.urandom(32)
    monkeypatch.setenv(crypto.ENV_KEY, base64.urlsafe_b64encode(raw).decode().rstrip("="))
    assert crypto.load_master_key() == raw


def test_load_from_keyfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(crypto.ENV_KEY, raising=False)
    kf = tmp_path / "k" / "master.key"
    crypto.write_keyfile(kf)
    assert oct(kf.stat().st_mode & 0o777) == "0o600"
    assert len(crypto.load_master_key(kf)) == 32
    monkeypatch.setenv(crypto.ENV_KEYFILE, str(kf))
    assert crypto.load_master_key() == crypto.load_master_key(kf)
    with pytest.raises(crypto.CryptoError, match="already exists"):
        crypto.write_keyfile(kf)


def test_keyfile_permissions_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(crypto.ENV_KEY, raising=False)
    monkeypatch.delenv(crypto.ENV_KEYFILE, raising=False)
    with pytest.raises(crypto.CryptoError, match="no master key"):
        crypto.load_master_key()
    with pytest.raises(crypto.CryptoError, match="not found"):
        crypto.load_master_key(tmp_path / "nope.key")
    kf = tmp_path / "loose.key"
    kf.write_text(crypto.generate_master_key())
    kf.chmod(0o644)
    with pytest.raises(crypto.CryptoError, match="chmod 600"):
        crypto.load_master_key(kf)


@pytest.mark.parametrize("bad", ["!!!notbase64!!!", base64.urlsafe_b64encode(b"x" * 16).decode()])
def test_bad_env_key(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv(crypto.ENV_KEY, bad)
    with pytest.raises(crypto.CryptoError):
        crypto.load_master_key()


def test_default_keyfile_respects_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert crypto.default_keyfile() == tmp_path / "sealbox" / "master.key"
