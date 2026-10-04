"""تست‌های امنیت: SEC-01 (2FA)، SEC-02 (AES-256)، SEC-04 (لاگ)، SEC-05 (RBAC)، SEC-06 (پشتیبان)."""

import base64
import sqlite3
from datetime import datetime, timedelta, timezone

import pyotp
import pytest

from ipind2.database import AuditRepository, init_db, make_engine, make_session_factory, session_scope
from ipind2.database.models import User
from ipind2.security import (
    AuthError,
    AuthService,
    EncryptionError,
    PermissionDenied,
    authorize,
    backup_database,
    decode_token,
    decrypt_bytes,
    decrypt_file,
    decrypt_text,
    encrypt_bytes,
    encrypt_file,
    encrypt_text,
    generate_key,
    issue_token,
    validate_password_policy,
)

GOOD_PASSWORD = "Str0ngPassword!!"


@pytest.fixture()
def factory():
    engine = make_engine("sqlite://")
    init_db(engine)
    return make_session_factory(engine)


def _enrolled_user(factory, key, username="alice", role="researcher"):
    with session_scope(factory) as s:
        svc = AuthService(s, key)
        _, uri = svc.create_user(username, GOOD_PASSWORD, role)
        secret = pyotp.parse_uri(uri).secret
        assert svc.confirm_totp(username, pyotp.TOTP(secret).now())
    return secret


class TestCrypto:
    def test_round_trip(self, encryption_key):
        blob = encrypt_bytes(b"hello nanoparticle", encryption_key)
        assert blob != b"hello nanoparticle"
        assert decrypt_bytes(blob, encryption_key) == b"hello nanoparticle"

    def test_nonce_is_random(self, encryption_key):
        assert encrypt_bytes(b"x", encryption_key) != encrypt_bytes(b"x", encryption_key)

    def test_tamper_detected(self, encryption_key):
        blob = bytearray(encrypt_bytes(b"payload", encryption_key))
        blob[-1] ^= 0x01
        with pytest.raises(EncryptionError):
            decrypt_bytes(bytes(blob), encryption_key)

    def test_wrong_key_rejected(self, encryption_key):
        blob = encrypt_bytes(b"payload", encryption_key)
        with pytest.raises(EncryptionError):
            decrypt_bytes(blob, generate_key())

    def test_aad_binds_context(self, encryption_key):
        token = encrypt_text("secret", encryption_key, aad="alice")
        assert decrypt_text(token, encryption_key, aad="alice") == "secret"
        with pytest.raises(EncryptionError):
            decrypt_text(token, encryption_key, aad="bob")

    def test_fail_closed_without_key(self, monkeypatch):
        monkeypatch.delenv("IPIND_ENCRYPTION_KEY", raising=False)
        with pytest.raises(EncryptionError):
            encrypt_text("x")

    def test_rejects_short_key(self):
        with pytest.raises(EncryptionError):
            encrypt_bytes(b"x", base64.b64encode(b"short").decode())

    def test_file_round_trip(self, tmp_path, encryption_key):
        source = tmp_path / "data.csv"
        source.write_bytes(b"a,b\n1,2\n")
        encrypted = encrypt_file(str(source), str(tmp_path / "data.csv.enc"), encryption_key)
        assert b"a,b" not in encrypted.read_bytes()
        restored = decrypt_file(str(encrypted), str(tmp_path / "restored.csv"), encryption_key)
        assert restored.read_bytes() == b"a,b\n1,2\n"


class TestPasswordPolicy:
    @pytest.mark.parametrize("password", ["short1A", "alllowercase12345", "ALLUPPERCASE12345", "NoDigitsHereAtAll"])
    def test_weak_rejected(self, password):
        with pytest.raises(ValueError):
            validate_password_policy(password)

    def test_strong_accepted(self):
        validate_password_policy(GOOD_PASSWORD)


class TestAuthentication:
    def test_login_requires_enrolled_2fa(self, factory, encryption_key):
        with session_scope(factory) as s:
            svc = AuthService(s, encryption_key)
            _, uri = svc.create_user("bob", GOOD_PASSWORD, "viewer")
            secret = pyotp.parse_uri(uri).secret
            with pytest.raises(AuthError):
                svc.login("bob", GOOD_PASSWORD, pyotp.TOTP(secret).now())

    def test_successful_login_returns_scoped_token(self, factory, encryption_key):
        secret = _enrolled_user(factory, encryption_key)
        with session_scope(factory) as s:
            token = AuthService(s, encryption_key).login("alice", GOOD_PASSWORD, pyotp.TOTP(secret).now())
        claims = decode_token(token)
        assert claims["sub"] == "alice" and claims["role"] == "researcher"

    def test_wrong_password_and_wrong_code_rejected(self, factory, encryption_key):
        secret = _enrolled_user(factory, encryption_key)
        with session_scope(factory) as s:
            svc = AuthService(s, encryption_key)
            with pytest.raises(AuthError):
                svc.login("alice", "WrongPassword123", pyotp.TOTP(secret).now())
            with pytest.raises(AuthError):
                svc.login("alice", GOOD_PASSWORD, "000000")

    def test_error_message_does_not_reveal_user_existence(self, factory, encryption_key):
        _enrolled_user(factory, encryption_key)
        with session_scope(factory) as s:
            svc = AuthService(s, encryption_key)
            messages = set()
            for username in ("alice", "nobody"):
                with pytest.raises(AuthError) as info:
                    svc.login(username, "WrongPassword123", "000000")
                messages.add(str(info.value))
        assert len(messages) == 1

    def test_lockout_after_repeated_failures(self, factory, encryption_key):
        secret = _enrolled_user(factory, encryption_key)
        with session_scope(factory) as s:
            svc = AuthService(s, encryption_key)
            for _ in range(5):
                with pytest.raises(AuthError):
                    svc.login("alice", "WrongPassword123", "000000")
            # حتی با اعتبارنامه درست، حساب قفل است
            with pytest.raises(AuthError):
                svc.login("alice", GOOD_PASSWORD, pyotp.TOTP(secret).now())

    def test_lock_expires(self, factory, encryption_key):
        secret = _enrolled_user(factory, encryption_key)
        with session_scope(factory) as s:
            user = s.query(User).filter_by(username="alice").one()
            user.locked_until = datetime.now(timezone.utc) - timedelta(minutes=1)
            token = AuthService(s, encryption_key).login("alice", GOOD_PASSWORD, pyotp.TOTP(secret).now())
        assert token

    def test_totp_secret_encrypted_at_rest(self, factory, encryption_key):
        secret = _enrolled_user(factory, encryption_key)
        with session_scope(factory) as s:
            stored = s.query(User).filter_by(username="alice").one().totp_secret_encrypted
        assert secret not in stored

    def test_passwords_are_hashed(self, factory, encryption_key):
        _enrolled_user(factory, encryption_key)
        with session_scope(factory) as s:
            stored = s.query(User).filter_by(username="alice").one().password_hash
        assert GOOD_PASSWORD not in stored and stored.startswith("$2")

    def test_duplicate_and_invalid_users_rejected(self, factory, encryption_key):
        with session_scope(factory) as s:
            svc = AuthService(s, encryption_key)
            svc.create_user("carol", GOOD_PASSWORD, "viewer")
            with pytest.raises(ValueError):
                svc.create_user("carol", GOOD_PASSWORD, "viewer")
            with pytest.raises(ValueError):
                svc.create_user("dave", GOOD_PASSWORD, "superuser")

    def test_login_attempts_are_audited(self, factory, encryption_key):
        secret = _enrolled_user(factory, encryption_key)
        with session_scope(factory) as s:
            svc = AuthService(s, encryption_key)
            with pytest.raises(AuthError):
                svc.login("alice", "WrongPassword123", "000000")
            svc.login("alice", GOOD_PASSWORD, pyotp.TOTP(secret).now())
            statuses = [(r.action, r.status) for r in AuditRepository(s).recent(50, "alice")]
        assert ("login", "denied") in statuses and ("login", "ok") in statuses


class TestTokensAndRBAC:
    def test_expired_token_rejected(self):
        token = issue_token("alice", "viewer", ttl_minutes=-1)
        with pytest.raises(AuthError):
            decode_token(token)

    def test_tampered_token_rejected(self):
        token = issue_token("alice", "viewer")
        with pytest.raises(AuthError):
            decode_token(token[:-2] + ("AA" if not token.endswith("AA") else "BB"))

    def test_token_signed_with_other_secret_rejected(self):
        token = issue_token("alice", "admin", secret="another-secret-that-is-long-enough-0123456789")
        with pytest.raises(AuthError):
            decode_token(token)

    @pytest.mark.parametrize(
        "role,permission,allowed",
        [
            ("viewer", "read", True),
            ("viewer", "optimize", False),
            ("viewer", "lab_ingest", False),
            ("researcher", "optimize", True),
            ("researcher", "lab_ingest", True),
            ("researcher", "manage_users", False),
            ("researcher", "audit_read", False),
            ("admin", "manage_users", True),
            ("admin", "audit_read", True),
            ("ghost", "read", False),
            ("admin", "undefined_permission", False),
        ],
    )
    def test_rbac_matrix(self, role, permission, allowed):
        if allowed:
            authorize(role, permission)
        else:
            with pytest.raises(PermissionDenied):
                authorize(role, permission)


class TestBackup:
    def test_encrypted_backup_restores(self, tmp_path, encryption_key):
        database = tmp_path / "live.db"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE t (x INTEGER)")
        connection.execute("INSERT INTO t VALUES (42)")
        connection.commit()
        connection.close()

        backup = backup_database(f"sqlite:///{database}", str(tmp_path / "backups"), encryption_key)
        assert backup.suffix == ".enc"
        assert not list((tmp_path / "backups").glob("*.dump")), "نسخه ساده نباید باقی بماند"
        assert b"SQLite format" not in backup.read_bytes()

        restored = tmp_path / "restored.db"
        decrypt_file(str(backup), str(restored), encryption_key)
        assert sqlite3.connect(restored).execute("SELECT x FROM t").fetchone() == (42,)

    def test_retention_prunes_old_backups(self, tmp_path, encryption_key):
        database = tmp_path / "live.db"
        sqlite3.connect(database).execute("CREATE TABLE t (x)").connection.commit()
        out = tmp_path / "backups"
        out.mkdir()
        for day in range(1, 6):
            (out / f"ipind2-2020010{day}T000000Z.dump.enc").write_bytes(b"old")
        backup_database(f"sqlite:///{database}", str(out), encryption_key, retention=3)
        assert len(list(out.glob("ipind2-*.dump.enc"))) == 3

    def test_memory_database_rejected(self, tmp_path, encryption_key):
        with pytest.raises(ValueError):
            backup_database("sqlite:///:memory:", str(tmp_path), encryption_key)
