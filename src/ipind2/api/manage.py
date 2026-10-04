"""
مدیریت عملیاتی: ساخت کاربر اولیه و اجرای سرور.

    python -m ipind2.api.manage create-user admin --role admin     # رمز از stdin/prompt
    python -m ipind2.api.manage serve --host 0.0.0.0 --port 8443 --certfile c.pem --keyfile k.pem

متغیرهای محیطی لازم: IPIND_JWT_SECRET (≥۳۲ نویسه)، IPIND_ENCRYPTION_KEY (base64، ۳۲ بایت؛
با ``python -c "from ipind2.security import generate_key; print(generate_key())"`` بسازید)،
IPIND_DATABASE_URL، IPIND_MODEL_DIR.
"""

import argparse
import getpass
import sys

from ..database import init_db, make_engine, make_session_factory, session_scope
from ..security import AuthService


def create_user(username: str, role: str) -> None:
    password = getpass.getpass("رمز عبور (حداقل ۱۲ نویسه): ")
    if password != getpass.getpass("تکرار رمز: "):
        sys.exit("رمزها یکسان نیستند")
    engine = make_engine()
    init_db(engine)
    with session_scope(make_session_factory(engine)) as session:
        _, uri = AuthService(session).create_user(username, password, role)
    print("کاربر ساخته شد. این URI را در برنامه احراز هویت (Google Authenticator و مشابه) اسکن کنید:")
    print(uri)
    print("سپس با POST /auth/enroll/confirm اولین کد را تأیید کنید؛ تا آن زمان ورود ممکن نیست.")


def serve(host: str, port: int, certfile: str, keyfile: str) -> None:
    import ssl

    import uvicorn

    from .app import create_app

    kwargs = {}
    if certfile and keyfile:
        kwargs.update(ssl_certfile=certfile, ssl_keyfile=keyfile, ssl_version=ssl.PROTOCOL_TLS_SERVER)
    elif host not in ("127.0.0.1", "localhost", "::1"):
        sys.exit("اجرا روی رابط غیرمحلی بدون TLS مجاز نیست (SEC-03)؛ --certfile/--keyfile بدهید یا پشت reverse proxy با TLS 1.3 اجرا کنید.")
    uvicorn.run(create_app(), host=host, port=port, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ipind2.api.manage")
    sub = parser.add_subparsers(dest="command", required=True)
    cu = sub.add_parser("create-user")
    cu.add_argument("username")
    cu.add_argument("--role", default="researcher", choices=["admin", "researcher", "viewer"])
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--certfile", default="")
    sv.add_argument("--keyfile", default="")
    args = parser.parse_args()
    if args.command == "create-user":
        create_user(args.username, args.role)
    else:
        serve(args.host, args.port, args.certfile, args.keyfile)


if __name__ == "__main__":
    main()
