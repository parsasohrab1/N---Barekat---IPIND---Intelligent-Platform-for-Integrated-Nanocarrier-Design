"""
تنظیمات TLS 1.3 (SEC-03) و پشتیبان‌گیری رمزشده روزانه (SEC-06).

TLS در لایه سرور اعمال می‌شود: ``ssl_context`` یک ``SSLContext`` با حداقل نسخه TLS 1.3
می‌سازد که به uvicorn/gunicorn (``--ssl-keyfile``/``--ssl-certfile``) یا یک reverse proxy
داده می‌شود؛ ``deploy/nginx.conf`` همان حداقل نسخه را اعمال می‌کند.
"""

import shutil
import sqlite3
import ssl
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .crypto import encrypt_file


def ssl_context(certfile: str, keyfile: str) -> ssl.SSLContext:
    """SSLContext سروری که فقط TLS 1.3 را می‌پذیرد."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return context


def backup_database(
    database_url: str,
    backup_dir: str,
    encryption_key: Optional[str] = None,
    retention: int = 14,
) -> Path:
    """
    پشتیبان رمزشده از پایگاه داده (SEC-06، باید روزانه از cron/scheduler اجرا شود).

    * SQLite: کپی سازگار با تراکنش (``sqlite3.Connection.backup``)؛
    * PostgreSQL: ``pg_dump`` (باید روی PATH باشد، اتصال از متغیرهای محیطی PG*).

    فایل خروجی با AES-256-GCM رمز می‌شود و نسخه ساده بلافاصله حذف می‌گردد؛ فقط
    ``retention`` پشتیبان اخیر نگه داشته می‌شود.
    """
    out_dir = Path(backup_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plain = out_dir / f"ipind2-{stamp}.dump"

    if database_url.startswith("sqlite"):
        source_path = database_url.split("///", 1)[1]
        if not source_path or source_path == ":memory:":
            raise ValueError("پشتیبان‌گیری از پایگاه داده حافظه‌ای ممکن نیست")
        source = sqlite3.connect(source_path)
        target = sqlite3.connect(str(plain))
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
    elif database_url.startswith("postgres"):
        if shutil.which("pg_dump") is None:
            raise RuntimeError("pg_dump روی PATH نیست")
        subprocess.run(["pg_dump", "--format=custom", f"--file={plain}", database_url], check=True)
    else:
        raise ValueError(f"نوع پایگاه داده پشتیبانی نمی‌شود: {database_url.split(':', 1)[0]}")

    encrypted = out_dir / (plain.name + ".enc")
    try:
        encrypt_file(str(plain), str(encrypted), encryption_key)
    finally:
        plain.unlink(missing_ok=True)

    backups: List[Path] = sorted(out_dir.glob("ipind2-*.dump.enc"))
    for old in backups[:-retention] if retention > 0 else []:
        old.unlink(missing_ok=True)
    return encrypted
