"""Security and privacy: 2FA, RBAC, AES-256 encryption, TLS 1.3, backup. See docs/SRS.md §6 (SEC-01..SEC-06)."""

from .auth import (
    PERMISSIONS,
    ROLES,
    AuthError,
    AuthService,
    PermissionDenied,
    authorize,
    decode_token,
    hash_password,
    issue_token,
    validate_password_policy,
    verify_password,
)
from .crypto import (
    EncryptionError,
    decrypt_bytes,
    decrypt_file,
    decrypt_text,
    encrypt_bytes,
    encrypt_file,
    encrypt_text,
    generate_key,
)
from .tls import backup_database, ssl_context

__all__ = [
    "PERMISSIONS",
    "ROLES",
    "AuthError",
    "AuthService",
    "PermissionDenied",
    "authorize",
    "decode_token",
    "hash_password",
    "issue_token",
    "validate_password_policy",
    "verify_password",
    "EncryptionError",
    "encrypt_bytes",
    "decrypt_bytes",
    "encrypt_text",
    "decrypt_text",
    "encrypt_file",
    "decrypt_file",
    "generate_key",
    "ssl_context",
    "backup_database",
]
