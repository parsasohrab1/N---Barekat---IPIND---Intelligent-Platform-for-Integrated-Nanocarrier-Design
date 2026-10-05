"""
TLS 1.3 settings (SEC-03) and daily encrypted backup (SEC-06).

TLS is enforced at the server layer: ``ssl_context`` builds an ``SSLContext`` with a minimum version of TLS 1.3
that is given to uvicorn/gunicorn (``--ssl-keyfile``/``--ssl-certfile``) or a reverse proxy;
``deploy/nginx.conf`` enforces the same minimum version.
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
    """Server SSLContext that accepts only TLS 1.3."""
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
    Encrypted backup of the database (SEC-06, must run daily from cron/scheduler).

    * SQLite: transaction-consistent copy (``sqlite3.Connection.backup``);
    * PostgreSQL: ``pg_dump`` (must be on PATH, connection from PG* environment variables).

    The output file is encrypted with AES-256-GCM and the plain version is deleted immediately; only the
    ``retention`` most recent backups are kept.
    """
    out_dir = Path(backup_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plain = out_dir / f"ipind2-{stamp}.dump"

    if database_url.startswith("sqlite"):
        source_path = database_url.split("///", 1)[1]
        if not source_path or source_path == ":memory:":
            raise ValueError("Backing up an in-memory database is not possible")
        source = sqlite3.connect(source_path)
        target = sqlite3.connect(str(plain))
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
    elif database_url.startswith("postgres"):
        if shutil.which("pg_dump") is None:
            raise RuntimeError("pg_dump is not on PATH")
        subprocess.run(["pg_dump", "--format=custom", f"--file={plain}", database_url], check=True)
    else:
        raise ValueError(f"Database type not supported: {database_url.split(':', 1)[0]}")

    encrypted = out_dir / (plain.name + ".enc")
    try:
        encrypt_file(str(plain), str(encrypted), encryption_key)
    finally:
        plain.unlink(missing_ok=True)

    backups: List[Path] = sorted(out_dir.glob("ipind2-*.dump.enc"))
    for old in backups[:-retention] if retention > 0 else []:
        old.unlink(missing_ok=True)
    return encrypted
