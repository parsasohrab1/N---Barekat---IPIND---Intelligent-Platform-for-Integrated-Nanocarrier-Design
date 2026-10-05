# Deployment and Operations Guide

## Initial setup

```bash
pip install -r requirements.txt            # + requirements-optional.txt for PostgreSQL/Redis/OpenMM
python -m ipind2.training.train --profile standard --out models/v1     # ~20–30 minutes on CPU
python -m ipind2.training.validate --model-dir models/v1 --out docs/validation_report.json
```

### Secrets (never in git or the image)

```bash
export IPIND_JWT_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"
export IPIND_ENCRYPTION_KEY="$(python -c 'from ipind2.security import generate_key;print(generate_key())')"
export IPIND_DATABASE_URL="postgresql+psycopg2://ipind:***@db:5432/ipind2"   # default: local SQLite (development only)
export IPIND_MODEL_DIR=models/v1
```

- **Do not lose the encryption key:** users' TOTP secrets and backups are encrypted with it; without the key they cannot be recovered. Keep it in a secret manager and plan rotation (the data format carries a version byte).
- Without `IPIND_JWT_SECRET` (≥ 32 characters) login does not work, and without `IPIND_ENCRYPTION_KEY` user creation fails (fail-closed).

### First admin user

```bash
python -m ipind2.api.manage create-user admin --role admin
# Scan the output URI in an authenticator app, then:
curl -X POST https://HOST/auth/enroll/confirm -d '{"username":"admin","password":"…","code":"123456"}'
```
Login is not possible until TOTP confirmation (SEC-01 for all users).

### Running

```bash
python -m ipind2.api.manage serve --host 0.0.0.0 --port 8443 --certfile c.pem --keyfile k.pem
# or docker compose up (nginx with TLS 1.3 + PostgreSQL + Redis; deploy/nginx.conf)
```
Running on a non-local interface without TLS is deliberately rejected (SEC-03).

## Daily backup (SEC-06)

```bash
# cron: 02:00 every night
0 2 * * * python -c "from ipind2.security import backup_database; backup_database('$IPIND_DATABASE_URL', '/backups', retention=14)"
```
The output is `ipind2-<time>.dump.enc` (AES-256-GCM); restore with `decrypt_file` and then `pg_restore`/SQLite copy. **Practice a test restore quarterly** — a backup whose restore has not been tested is not a backup. (An automated SQLite restore test exists; the PostgreSQL path depends on `pg_dump` and is not tested in CI.)

## Model upgrade (FR-12)

1. Train the new version ⇒ 2. `ipind2-benchmark --model-dir models/new --history benchmarks/history.json` ⇒ 3. Replace `models/current` only on PASS.
The PASS gate means: relative RMSE degradation ≤ 5%, R² degradation ≤ 0.02 versus the previous version **and** NFR-01/02/03 holding. If the data generation rules/reference change, the reference fingerprint changes and the history starts over.

## Lab feedback (FR-06/FR-11)

Record results with `POST /lab/results` (or `lab_automation` CSV/REST); `POST /active-learning/retrain` fine-tunes the model after 10 new results (or `force`) and marks the results as "consumed". **Warning:** the current replay is synthetic data; in production supply real training data as `replay` to reduce catastrophic forgetting.

## Operational limitations

- A single API process runs design work on a thread pool (default 2); for more concurrency add multiple replicas with a work queue (Celery/RQ) — the current code has no distributed queue.
- The rate limiter is only account lockout after 5 failed logins; apply IP limiting in nginx/WAF.
- NFR-07 (99.9% availability) depends on infrastructure (multiple replicas, replicated DB, health checks) and is not provable by code.
