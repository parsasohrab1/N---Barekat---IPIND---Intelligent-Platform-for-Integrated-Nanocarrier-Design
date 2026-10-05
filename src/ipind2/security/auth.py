"""
Two-factor authentication, RBAC and session token (SEC-01, SEC-04, SEC-05).

* Password: bcrypt (cost 12) + minimum length policy;
* Second factor: TOTP (RFC 6238) for **all** users — login before TOTP enrollment confirmation is rejected;
* Account lockout: 5 consecutive failed logins → 15-minute lock;
* Token: JWT (HS256) with short expiry and role claim;
* Roles: admin / researcher / viewer with explicit permission mapping (deny-by-default);
* Every login/access-denial event is recorded in ``audit_log``.

The login error response is deliberately uniform ("incorrect username or password"), so the existence or non-existence of a
username cannot be guessed.
"""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, FrozenSet, Optional, Tuple

import bcrypt
import jwt
import pyotp
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database.models import User
from ..database.repository import AuditRepository
from .crypto import decrypt_text, encrypt_text

JWT_SECRET_ENV = "IPIND_JWT_SECRET"
JWT_ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = 30
MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15
MIN_PASSWORD_LENGTH = 12
ISSUER = "ipind2"

ROLES: Tuple[str, ...] = ("admin", "researcher", "viewer")

# Permissions; every API action must be explicitly in this table (deny-by-default).
PERMISSIONS: Dict[str, FrozenSet[str]] = {
    "viewer": frozenset({"read"}),
    "researcher": frozenset({"read", "generate", "predict", "optimize", "validate", "lab_ingest", "active_learning"}),
    "admin": frozenset(
        {"read", "generate", "predict", "optimize", "validate", "lab_ingest", "active_learning", "manage_users", "audit_read", "benchmark"}
    ),
}


class AuthError(Exception):
    """Authentication error (message deliberately generic)."""


class PermissionDenied(Exception):
    """The authenticated user does not have permission for this action."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite returns dates without tzinfo; we assume UTC for safe comparison."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def validate_password_policy(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if password.lower() == password or password.upper() == password or not any(c.isdigit() for c in password):
        raise ValueError("Password must combine lowercase and uppercase letters and digits")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("ascii"))
    except ValueError:
        return False


_DUMMY_HASH: Optional[str] = None


def _dummy_hash() -> str:
    """A valid bcrypt hash to equalize response time for a nonexistent user (computed once)."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("timing-equalisation-Placeholder-1")
    return _DUMMY_HASH


def jwt_secret() -> str:
    secret = os.environ.get(JWT_SECRET_ENV)
    if not secret or len(secret) < 32:
        raise AuthError(f"{JWT_SECRET_ENV} must be at least 32 characters (not set or too short)")
    return secret


def issue_token(username: str, role: str, secret: Optional[str] = None, ttl_minutes: int = TOKEN_TTL_MINUTES) -> str:
    now = _now()
    payload = {
        "sub": username,
        "role": role,
        "iss": ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(payload, secret or jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_token(token: str, secret: Optional[str] = None) -> Dict:
    try:
        return jwt.decode(
            token,
            secret or jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            issuer=ISSUER,
            options={"require": ["exp", "sub", "role", "iss"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError("Token is invalid or expired") from exc


def authorize(role: str, permission: str) -> None:
    """``PermissionDenied`` if the role lacks the permission (unknown role = no permission)."""
    if permission not in PERMISSIONS.get(role, frozenset()):
        raise PermissionDenied(f"Role '{role}' does not have permission '{permission}'")


class AuthService:
    """User logic, 2FA and login on a database ``Session``."""

    def __init__(self, session: Session, encryption_key: Optional[str] = None):
        self.session = session
        self.encryption_key = encryption_key
        self.audit = AuditRepository(session)

    # --- User management ------------------------------------------------
    def create_user(self, username: str, password: str, role: str = "viewer") -> Tuple[User, str]:
        """
        Create a user and TOTP secret. ``totp_enabled`` stays ``False`` until the first code is confirmed.

        Returns: (user, TOTP enrollment URI for the QR code in the authenticator app)
        """
        if role not in ROLES:
            raise ValueError(f"Invalid role: {role!r}")
        if not username or len(username) > 64:
            raise ValueError("Username is invalid")
        validate_password_policy(password)
        if self.session.scalar(select(User).where(User.username == username)) is not None:
            raise ValueError("This username is already registered")

        secret = pyotp.random_base32()
        user = User(
            username=username,
            password_hash=hash_password(password),
            role=role,
            totp_secret_encrypted=encrypt_text(secret, self.encryption_key, aad=username),
        )
        self.session.add(user)
        self.session.flush()
        self.audit.record("user_create", username, resource=f"user:{username}", detail={"role": role})
        uri = pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name="IPIND2")
        return user, uri

    def _totp(self, user: User) -> pyotp.TOTP:
        secret = decrypt_text(user.totp_secret_encrypted, self.encryption_key, aad=user.username)
        return pyotp.TOTP(secret)

    def confirm_totp(self, username: str, code: str) -> bool:
        """Confirm TOTP enrollment with the first valid code; then login is enabled for this user."""
        user = self.session.scalar(select(User).where(User.username == username))
        if user is None or not user.totp_secret_encrypted:
            return False
        if self._totp(user).verify(code, valid_window=1):
            user.totp_enabled = True
            self.audit.record("totp_enrolled", username)
            return True
        self.audit.record("totp_enroll_failed", username, status="denied")
        return False

    # --- Login ----------------------------------------------------------
    def login(self, username: str, password: str, totp_code: str, ip_address: Optional[str] = None) -> str:
        """
        Login with password + TOTP code. Returns a JWT token or raises ``AuthError``.

        The order of checks is such that the response time/message does not reveal user existence.
        """
        generic = AuthError("Username, password or two-factor code is incorrect")
        user = self.session.scalar(select(User).where(User.username == username))

        if user is None:
            # We also pay the bcrypt cost for a nonexistent user (time equalization)
            verify_password(password, _dummy_hash())
            self.audit.record("login", username, status="denied", detail={"reason": "unknown_user"}, ip_address=ip_address)
            raise generic

        locked_until = _aware(user.locked_until)
        if locked_until is not None and locked_until > _now():
            self.audit.record("login", username, status="denied", detail={"reason": "locked"}, ip_address=ip_address)
            raise AuthError("Account is temporarily locked; try again later")

        password_ok = verify_password(password, user.password_hash)
        # Even if the password is wrong we also check TOTP so the branches behave similarly
        totp_ok = bool(user.totp_enabled) and self._totp(user).verify(totp_code or "", valid_window=1)

        if not user.is_active or not password_ok or not totp_ok:
            user.failed_logins = (user.failed_logins or 0) + 1
            if user.failed_logins >= MAX_FAILED_LOGINS:
                user.locked_until = _now() + timedelta(minutes=LOCKOUT_MINUTES)
                user.failed_logins = 0
            reason = "2fa_not_enrolled" if not user.totp_enabled else "bad_credentials"
            self.audit.record("login", username, status="denied", detail={"reason": reason}, ip_address=ip_address)
            raise generic

        user.failed_logins = 0
        user.locked_until = None
        self.audit.record("login", username, ip_address=ip_address)
        return issue_token(user.username, user.role)
