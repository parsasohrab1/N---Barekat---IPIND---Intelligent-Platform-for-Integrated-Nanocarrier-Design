"""
رمزنگاری داده در حالت ذخیره‌سازی — AES-256-GCM (SEC-02).

کلید ۳۲ بایتی از متغیر محیطی ``IPIND_ENCRYPTION_KEY`` (base64) خوانده می‌شود. برنامه
عمداً **بدون کلید شروع به رمزنگاری نمی‌کند** (fail-closed): نبود کلید در تولید باید به‌صورت
خطا دیده شود، نه اینکه داده بی‌صدا ساده ذخیره شود.

قالب خروجی: ``version(1B) || nonce(12B) || ciphertext+tag``؛ بایت نسخه، چرخش کلید/الگوریتم
در آینده را بدون شکستن داده‌های قدیمی ممکن می‌کند. ``aad`` (associated data) رمزنص را به
زمینه‌اش (مثلاً نام کاربر یا شناسه رکورد) می‌بندد تا جابه‌جایی ciphertext بین رکوردها
شناسایی شود.
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
    """خطای رمزنگاری/رمزگشایی (کلید نبود، داده دستکاری‌شده، ...)."""


def generate_key() -> str:
    """کلید تصادفی AES-256 به‌صورت base64 (برای ذخیره در secret manager)."""
    return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")


def _load_key(key: Optional[str]) -> bytes:
    raw = key if key is not None else os.environ.get(KEY_ENV)
    if not raw:
        raise EncryptionError(
            f"کلید رمزنگاری تنظیم نشده است؛ متغیر محیطی {KEY_ENV} را (base64، ۳۲ بایت) مقداردهی کنید."
        )
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise EncryptionError("کلید رمزنگاری base64 معتبر نیست") from exc
    if len(decoded) != 32:
        raise EncryptionError(f"کلید باید دقیقاً ۳۲ بایت (AES-256) باشد، نه {len(decoded)}")
    return decoded


def encrypt_bytes(plaintext: bytes, key: Optional[str] = None, aad: Optional[bytes] = None) -> bytes:
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(_load_key(key)).encrypt(nonce, plaintext, aad)
    return _VERSION + nonce + ciphertext


def decrypt_bytes(blob: bytes, key: Optional[str] = None, aad: Optional[bytes] = None) -> bytes:
    if len(blob) < 1 + _NONCE_BYTES + 16 or blob[:1] != _VERSION:
        raise EncryptionError("قالب داده رمزشده نامعتبر است")
    nonce = blob[1 : 1 + _NONCE_BYTES]
    try:
        return AESGCM(_load_key(key)).decrypt(nonce, blob[1 + _NONCE_BYTES :], aad)
    except InvalidTag as exc:
        raise EncryptionError("رمزگشایی ناموفق: کلید اشتباه یا داده دستکاری‌شده") from exc


def encrypt_text(text: str, key: Optional[str] = None, aad: Optional[str] = None) -> str:
    """رمزنگاری رشته؛ خروجی base64 مناسب ذخیره در ستون متنی پایگاه داده."""
    blob = encrypt_bytes(text.encode("utf-8"), key, aad.encode("utf-8") if aad else None)
    return base64.b64encode(blob).decode("ascii")


def decrypt_text(token: str, key: Optional[str] = None, aad: Optional[str] = None) -> str:
    blob = base64.b64decode(token.encode("ascii"))
    return decrypt_bytes(blob, key, aad.encode("utf-8") if aad else None).decode("utf-8")


def encrypt_file(source: str, destination: str, key: Optional[str] = None) -> Path:
    """رمزنگاری یک فایل (مثلاً پشتیبان یا خروجی CSV) با AES-256-GCM."""
    data = Path(source).read_bytes()
    out = Path(destination)
    out.write_bytes(encrypt_bytes(data, key, aad=out.name.encode("utf-8")))
    return out


def decrypt_file(source: str, destination: str, key: Optional[str] = None) -> Path:
    src = Path(source)
    out = Path(destination)
    out.write_bytes(decrypt_bytes(src.read_bytes(), key, aad=src.name.encode("utf-8")))
    return out
