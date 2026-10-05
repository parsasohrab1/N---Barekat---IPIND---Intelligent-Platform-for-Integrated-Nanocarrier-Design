"""
Encryption of data at rest — AES-256-GCM (SEC-02).

The 32-byte key is read from the ``IPIND_ENCRYPTION_KEY`` environment variable (base64). The application
deliberately **does not start encrypting without a key** (fail-closed): a missing key in production should be seen as
an error, not data silently stored in plain text.

Output format: ``version(1B) || nonce(12B) || ciphertext+tag``; the version byte makes future key/algorithm
rotation possible without breaking old data. ``aad`` (associated data) binds the ciphertext to its
context (e.g., username or record ID) so that moving ciphertext between records
is detected.
"""

import base64
import os
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_ENV = "IPIND_ENCRYPTION_KEY"
_VERSION = b"\x01"
_NONCE_BYTES = 12


class EncryptionError(RuntimeError):
    """Encryption/decryption error (missing key, tampered data, ...)."""


def generate_key() -> str:
    """Random AES-256 key as base64 (for storing in a secret manager)."""
    return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")


def _load_key(key: Optional[str]) -> bytes:
    raw = key if key is not None else os.environ.get(KEY_ENV)
    if not raw:
        raise EncryptionError(
            f"Encryption key is not set; set the environment variable {KEY_ENV} (base64, 32 bytes)."
        )
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise EncryptionError("Encryption key is not valid base64") from exc
    if len(decoded) != 32:
        raise EncryptionError(f"Key must be exactly 32 bytes (AES-256), not {len(decoded)}")
    return decoded


def encrypt_bytes(plaintext: bytes, key: Optional[str] = None, aad: Optional[bytes] = None) -> bytes:
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(_load_key(key)).encrypt(nonce, plaintext, aad)
    return _VERSION + nonce + ciphertext


def decrypt_bytes(blob: bytes, key: Optional[str] = None, aad: Optional[bytes] = None) -> bytes:
    if len(blob) < 1 + _NONCE_BYTES + 16 or blob[:1] != _VERSION:
        raise EncryptionError("Encrypted data format is invalid")
    nonce = blob[1 : 1 + _NONCE_BYTES]
    try:
        return AESGCM(_load_key(key)).decrypt(nonce, blob[1 + _NONCE_BYTES :], aad)
    except InvalidTag as exc:
        raise EncryptionError("Decryption failed: wrong key or tampered data") from exc


def encrypt_text(text: str, key: Optional[str] = None, aad: Optional[str] = None) -> str:
    """Encrypt a string; the output is base64 suitable for storing in a database text column."""
    blob = encrypt_bytes(text.encode("utf-8"), key, aad.encode("utf-8") if aad else None)
    return base64.b64encode(blob).decode("ascii")


def decrypt_text(token: str, key: Optional[str] = None, aad: Optional[str] = None) -> str:
    blob = base64.b64decode(token.encode("ascii"))
    return decrypt_bytes(blob, key, aad.encode("utf-8") if aad else None).decode("utf-8")


def encrypt_file(source: str, destination: str, key: Optional[str] = None) -> Path:
    """Encrypt a file (e.g., backup or CSV output) with AES-256-GCM."""
    data = Path(source).read_bytes()
    out = Path(destination)
    out.write_bytes(encrypt_bytes(data, key, aad=out.name.encode("utf-8")))
    return out


def decrypt_file(source: str, destination: str, key: Optional[str] = None) -> Path:
    src = Path(source)
    out = Path(destination)
    out.write_bytes(decrypt_bytes(src.read_bytes(), key, aad=src.name.encode("utf-8")))
    return out
