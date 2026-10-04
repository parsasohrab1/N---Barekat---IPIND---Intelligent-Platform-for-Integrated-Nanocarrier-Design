"""
احراز هویت دو مرحله‌ای، RBAC و توکن جلسه (SEC-01, SEC-04, SEC-05).

* رمز عبور: bcrypt (cost 12) + سیاست حداقل طول؛
* عامل دوم: TOTP (RFC 6238) برای **همه** کاربران — ورود پیش از تأیید ثبت‌نام TOTP رد می‌شود؛
* قفل حساب: ۵ ورود ناموفق پشت‌سرهم → قفل ۱۵ دقیقه‌ای؛
* توکن: JWT (HS256) با انقضای کوتاه و ادعای نقش؛
* نقش‌ها: admin / researcher / viewer با نگاشت صریح مجوزها (deny-by-default)؛
* هر رویداد ورود/رد دسترسی در ``audit_log`` ثبت می‌شود.

پاسخ خطای ورود عمداً یکسان است («نام کاربری یا رمز نادرست»)، تا وجود/عدم وجود نام
کاربری قابل‌حدس نباشد.
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

# مجوزها؛ هر عمل API باید صریحاً در این جدول باشد (deny-by-default).
PERMISSIONS: Dict[str, FrozenSet[str]] = {
    "viewer": frozenset({"read"}),
    "researcher": frozenset({"read", "generate", "predict", "optimize", "validate", "lab_ingest", "active_learning"}),
    "admin": frozenset(
        {"read", "generate", "predict", "optimize", "validate", "lab_ingest", "active_learning", "manage_users", "audit_read", "benchmark"}
    ),
}


class AuthError(Exception):
    """خطای احراز هویت (پیام عمداً عمومی)."""


class PermissionDenied(Exception):
    """کاربر احرازشده مجوز این عمل را ندارد."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite تاریخ را بدون tzinfo برمی‌گرداند؛ برای مقایسه امن UTC فرض می‌کنیم."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def validate_password_policy(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"رمز عبور باید حداقل {MIN_PASSWORD_LENGTH} نویسه باشد")
    if password.lower() == password or password.upper() == password or not any(c.isdigit() for c in password):
        raise ValueError("رمز عبور باید ترکیبی از حروف کوچک و بزرگ و عدد باشد")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("ascii"))
    except ValueError:
        return False


_DUMMY_HASH: Optional[str] = None


def _dummy_hash() -> str:
    """هش bcrypt معتبر برای یکسان‌سازی زمان پاسخ کاربر ناموجود (یک‌بار محاسبه می‌شود)."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("timing-equalisation-Placeholder-1")
    return _DUMMY_HASH


def jwt_secret() -> str:
    secret = os.environ.get(JWT_SECRET_ENV)
    if not secret or len(secret) < 32:
        raise AuthError(f"{JWT_SECRET_ENV} باید حداقل ۳۲ نویسه باشد (تنظیم نشده یا کوتاه است)")
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
        raise AuthError("توکن نامعتبر یا منقضی است") from exc


def authorize(role: str, permission: str) -> None:
    """``PermissionDenied`` اگر نقش مجوز را نداشته باشد (نقش ناشناخته = بدون مجوز)."""
    if permission not in PERMISSIONS.get(role, frozenset()):
        raise PermissionDenied(f"نقش «{role}» مجوز «{permission}» را ندارد")


class AuthService:
    """منطق کاربران، 2FA و ورود روی یک ``Session`` پایگاه داده."""

    def __init__(self, session: Session, encryption_key: Optional[str] = None):
        self.session = session
        self.encryption_key = encryption_key
        self.audit = AuditRepository(session)

    # --- مدیریت کاربر ------------------------------------------------
    def create_user(self, username: str, password: str, role: str = "viewer") -> Tuple[User, str]:
        """
        ایجاد کاربر و راز TOTP. ``totp_enabled`` تا تأیید اولین کد ``False`` می‌ماند.

        Returns: (کاربر، URI ثبت‌نام TOTP برای QR در برنامه احراز هویت)
        """
        if role not in ROLES:
            raise ValueError(f"نقش نامعتبر: {role!r}")
        if not username or len(username) > 64:
            raise ValueError("نام کاربری نامعتبر است")
        validate_password_policy(password)
        if self.session.scalar(select(User).where(User.username == username)) is not None:
            raise ValueError("این نام کاربری قبلاً ثبت شده است")

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
        """تأیید ثبت‌نام TOTP با اولین کد معتبر؛ سپس ورود برای این کاربر فعال می‌شود."""
        user = self.session.scalar(select(User).where(User.username == username))
        if user is None or not user.totp_secret_encrypted:
            return False
        if self._totp(user).verify(code, valid_window=1):
            user.totp_enabled = True
            self.audit.record("totp_enrolled", username)
            return True
        self.audit.record("totp_enroll_failed", username, status="denied")
        return False

    # --- ورود ----------------------------------------------------------
    def login(self, username: str, password: str, totp_code: str, ip_address: Optional[str] = None) -> str:
        """
        ورود با رمز + کد TOTP. توکن JWT برمی‌گرداند یا ``AuthError`` می‌اندازد.

        ترتیب بررسی‌ها طوری است که زمان/پیام پاسخ، وجود کاربر را فاش نکند.
        """
        generic = AuthError("نام کاربری، رمز عبور یا کد دومرحله‌ای نادرست است")
        user = self.session.scalar(select(User).where(User.username == username))

        if user is None:
            # هزینه bcrypt را برای کاربر ناموجود هم می‌پردازیم (یکسان‌سازی زمان)
            verify_password(password, _dummy_hash())
            self.audit.record("login", username, status="denied", detail={"reason": "unknown_user"}, ip_address=ip_address)
            raise generic

        locked_until = _aware(user.locked_until)
        if locked_until is not None and locked_until > _now():
            self.audit.record("login", username, status="denied", detail={"reason": "locked"}, ip_address=ip_address)
            raise AuthError("حساب موقتاً قفل شده است؛ بعداً دوباره تلاش کنید")

        password_ok = verify_password(password, user.password_hash)
        # حتی اگر رمز غلط باشد TOTP را هم بررسی می‌کنیم تا شاخه‌ها مشابه رفتار کنند
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
