"""Envelope encryption for the vault.

* Every dataset gets its own random 256-bit data key (DEK).
* Data is encrypted with AES-256-GCM under the DEK; the DEK is wrapped by a
  ``KeyProvider`` (by default a local master key from env or a keyfile).
* Associated data binds each ciphertext to its dataset and part name, so blobs
  cannot be swapped between datasets without detection.

Blob format: ``b"SBX1" || nonce(12) || ciphertext+tag``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import stat
from pathlib import Path
from typing import Protocol

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .errors import SealboxError

MAGIC = b"SBX1"
NONCE_LEN = 12
KEY_LEN = 32
ENV_KEY = "SEALBOX_MASTER_KEY"
ENV_KEYFILE = "SEALBOX_KEYFILE"


class CryptoError(SealboxError):
    """Raised when decryption or key handling fails."""


def encrypt(key: bytes, plaintext: bytes, aad: bytes) -> bytes:
    if len(key) != KEY_LEN:
        raise CryptoError("AES-256-GCM key must be 32 bytes")
    nonce = os.urandom(NONCE_LEN)
    return MAGIC + nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def decrypt(key: bytes, blob: bytes, aad: bytes) -> bytes:
    if len(key) != KEY_LEN:
        raise CryptoError("AES-256-GCM key must be 32 bytes")
    if len(blob) < len(MAGIC) + NONCE_LEN + 16 or not blob.startswith(MAGIC):
        raise CryptoError("not a sealbox ciphertext")
    nonce = blob[len(MAGIC) : len(MAGIC) + NONCE_LEN]
    try:
        return AESGCM(key).decrypt(nonce, blob[len(MAGIC) + NONCE_LEN :], aad)
    except Exception as exc:  # cryptography raises InvalidTag
        raise CryptoError("decryption failed (wrong key or tampered data)") from exc


def new_data_key() -> bytes:
    return AESGCM.generate_key(bit_length=256)


def derive_key(master: bytes, purpose: str) -> bytes:
    """Derive an independent sub-key (tokenization, audit MAC, ...) from the master key."""
    return HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=b"sealbox-v1", info=purpose.encode()).derive(master)


class KeyProvider(Protocol):
    """Wraps and unwraps per-dataset data keys. Implement this for a cloud KMS."""

    def wrap(self, dek: bytes, aad: bytes) -> bytes: ...

    def unwrap(self, wrapped: bytes, aad: bytes) -> bytes: ...

    def derive(self, purpose: str) -> bytes: ...


class LocalKeyProvider:
    """Master key held in process memory, loaded from env or a 0600 keyfile."""

    def __init__(self, master: bytes) -> None:
        if len(master) != KEY_LEN:
            raise CryptoError("master key must be 32 bytes")
        self._master = master

    def wrap(self, dek: bytes, aad: bytes) -> bytes:
        return encrypt(self._master, dek, b"wrap:" + aad)

    def unwrap(self, wrapped: bytes, aad: bytes) -> bytes:
        return decrypt(self._master, wrapped, b"wrap:" + aad)

    def derive(self, purpose: str) -> bytes:
        return derive_key(self._master, purpose)

    def fingerprint(self) -> str:
        """Non-secret identifier of the key (to detect using the wrong key)."""
        return hmac.new(self._master, b"sealbox-key-id", hashlib.sha256).hexdigest()[:16]


class KMSKeyProvider:  # pragma: no cover - interface stub
    """Hook for AWS KMS / GCP KMS / Azure Key Vault / HashiCorp Vault transit.

    Implement ``wrap``/``unwrap`` with the provider's encrypt/decrypt API (the DEK never
    leaves memory unwrapped) and ``derive`` with an HMAC/derive call or a locally cached
    derived key. Not implemented in v0.1.
    """

    def __init__(self, key_uri: str) -> None:
        self.key_uri = key_uri

    def wrap(self, dek: bytes, aad: bytes) -> bytes:
        raise NotImplementedError("KMS key provider is not implemented in sealbox v0.1")

    def unwrap(self, wrapped: bytes, aad: bytes) -> bytes:
        raise NotImplementedError("KMS key provider is not implemented in sealbox v0.1")

    def derive(self, purpose: str) -> bytes:
        raise NotImplementedError("KMS key provider is not implemented in sealbox v0.1")


def generate_master_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(KEY_LEN)).decode()


def _decode_key(text: str) -> bytes:
    text = text.strip()
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except Exception as exc:
        raise CryptoError("master key is not valid base64") from exc
    if len(raw) != KEY_LEN:
        raise CryptoError("master key must decode to 32 bytes")
    return raw


def write_keyfile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CryptoError(f"keyfile already exists: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(generate_master_key() + "\n")


def load_master_key(keyfile: Path | None = None) -> bytes:
    """Load the master key: ``$SEALBOX_MASTER_KEY`` wins, then the keyfile."""
    env = os.environ.get(ENV_KEY)
    if env:
        return _decode_key(env)
    if keyfile is None and os.environ.get(ENV_KEYFILE):
        keyfile = Path(os.environ[ENV_KEYFILE])
    if keyfile is None:
        raise CryptoError(f"no master key: set {ENV_KEY} or {ENV_KEYFILE}, or pass --keyfile")
    if not keyfile.exists():
        raise CryptoError(f"keyfile not found: {keyfile}")
    mode = keyfile.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise CryptoError(f"keyfile {keyfile} is readable by group/others; chmod 600 it")
    return _decode_key(keyfile.read_text())


def default_keyfile() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "sealbox" / "master.key"
