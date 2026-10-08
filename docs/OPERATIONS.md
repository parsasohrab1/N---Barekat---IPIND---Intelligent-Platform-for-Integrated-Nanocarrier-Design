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

## Real MD environment (FR-05)

The main environment does not need OpenMM. Real MD runs in a separate conda env that `CondaOpenMMEngine` calls through `src/ipind2/md_simulation/openmm_runner.py`:

```bash
conda create -n ipind-md -c conda-forge python=3.11 openmm openff-toolkit-base \
    openff-interchange-base openff-forcefields rdkit numpy
export IPIND_MD_PYTHON=/path/to/miniforge3/envs/ipind-md/python    # python.exe on Windows
python -m ipind2.md_simulation.campaign --model-dir models/v1 --n 3 --ns 100 --platform OpenCL --out docs/md_validation.json
```

What this MD is, and is not:

- OpenFF 2.2.0 "Sage" force field, **Gasteiger** partial charges, **OBC2 implicit solvent**, one solute molecule, Langevin 310 K, 2 fs.
- Not CHARMM36/OPLS-AA, no explicit water, no self-assembled particle, no MM-GBSA. AmberTools (AM1-BCC charges, GAFF) has no Windows conda build, so `openmmforcefields` was not usable on Windows; on Linux/macOS, AM1-BCC charges are preferable.
- Elements without Sage parameters (Au, Fe, Si, Zn, ...) are rejected cleanly and reported as skipped.
- Throughput measured on this project's laptop (8 CPU cores, MX450 via OpenCL), 162–221-atom lipids: **65–116 ns/day**, so 100 ns per candidate takes about 1–1.5 days. Plan the full requirement on a GPU server or cloud GPU.
- `docs/md_validation.json` has `complete: true` only if every candidate ran on a real engine for at least 100 ns; the TRL assessor (criterion R3) reads exactly that flag.
