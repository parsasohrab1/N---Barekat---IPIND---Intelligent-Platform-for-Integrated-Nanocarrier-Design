"""
Operational management: creating the initial user and running the server.

    python -m ipind2.api.manage create-user admin --role admin     # password from stdin/prompt
    python -m ipind2.api.manage serve --host 0.0.0.0 --port 8443 --certfile c.pem --keyfile k.pem

Required environment variables: IPIND_JWT_SECRET (≥32 characters), IPIND_ENCRYPTION_KEY (base64, 32 bytes;
create it with ``python -c "from ipind2.security import generate_key; print(generate_key())"``),
IPIND_DATABASE_URL, IPIND_MODEL_DIR.
"""

import argparse
import getpass
import sys

from ..database import init_db, make_engine, make_session_factory, session_scope
from ..security import AuthService


def create_user(username: str, role: str) -> None:
    password = getpass.getpass("Password (at least 12 characters): ")
    if password != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match")
    engine = make_engine()
    init_db(engine)
    with session_scope(make_session_factory(engine)) as session:
        _, uri = AuthService(session).create_user(username, password, role)
    print("User created. Scan this URI in an authenticator app (Google Authenticator or similar):")
    print(uri)
    print("Then confirm the first code with POST /auth/enroll/confirm; login is not possible until then.")


def serve(host: str, port: int, certfile: str, keyfile: str) -> None:
    import ssl

    import uvicorn

    from .app import create_app

    kwargs = {}
    if certfile and keyfile:
        kwargs.update(ssl_certfile=certfile, ssl_keyfile=keyfile, ssl_version=ssl.PROTOCOL_TLS_SERVER)
    elif host not in ("127.0.0.1", "localhost", "::1"):
        sys.exit("Running on a non-local interface without TLS is not allowed (SEC-03); provide --certfile/--keyfile or run behind a reverse proxy with TLS 1.3.")
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
