# TRL Assessment (Evidence-Based)

> **Honest result: TRL 4 proven; TRL 5 not yet achieved.**
> The code and tests provide the engineering readiness for TRL 5, but TRL 5 means validation in a **relevant environment** (real data/experiments, real MD, staging deployment), and these are not achieved by writing code. The "Remaining steps" section states exactly what is needed.

The output below was computed by `python -m ipind2.trl --tests-passed yes` from `docs/validation_report.json` and `benchmarks/history.json` (criteria in [`src/ipind2/trl.py`](../src/ipind2/trl.py), with a test proving that synthetic data or a claim cannot reach TRL 5).

## TRL computed from evidence: **4**

| Level | Criterion | Status | Evidence |
|---|---|---|---|
| 4 | T1: Automated test suite passes | ✅ | pytest PASS |
| 4 | T2: Model accuracy on synthetic holdout: NFR-01/02/03 and FR-03 | ✅ | From validation_report.accuracy_holdout |
| 4 | T3: Performance NFRs (04, 05, 06, 08) measured and satisfied | ✅ | NFR-04=✓, NFR-05=✓, NFR-06=✓, NFR-08=✓ |
| 5 | R1: Frozen reference benchmark gate PASS (no degradation and NFRs hold) | ✅ | benchmark_gate.passed |
| 5 | R2: On a real public dataset, platform R² ≥ simple baseline (RandomForest+Morgan) | ✅ | lantern-hela: platform R² 0.507 vs. RandomForest+Morgan baseline 0.459 |
| 5 | R3: Real MD validation (≥100 ns, MD engine) for candidates | ❌ | 3 candidates, real MD, longest 2 ns of the required 100 ns — below the requirement |
| 5 | R4: Lab loop: ≥10 real results and improvement on a real holdout | ❌ | 0 real results; holdout improvement=no/not measured |
| 5 | R5: Deployment in a relevant environment (staging: PostgreSQL+TLS1.3+Redis) with signed attestation | ❌ | No manual attestation available |
| 5 | R6: Independent security review (SEC-01..06) with signed attestation | ❌ | No manual attestation available |

- Unmet criteria for TRL 5: R3, R4, R5, R6
- Without real MD, FR-05 (≥100 ns) is not confirmed; conformer sampling is not a substitute for it.

## Remaining steps for TRL 5 (in priority order)

| Criterion | Work needed | Who/what |
|---|---|---|
| R2 | ✅ **Met** (after adding Morgan fingerprints): platform R² 0.51 vs RandomForest 0.46 on real LANTERN/AGILE (5 splits, wins 5/5, p=0.008). Still far below the paper-reported 0.82 (different protocol). **This still does not test the platform's own properties (size, zeta, ...) on real data** — that needs R4 / LANCE / LNP-622 | The current criterion is only "≥ simple baseline"; making it stricter (e.g. ≥ 0.6) is a team decision |
| R3 | **Engine installed and working; length not reached.** Real OpenMM MD (Sage + Gasteiger + implicit solvent) ran 2 ns on 3 candidates (stable). Still needed: 100 ns each — about 17–19 h per candidate on this laptop at 125–144 ns/day, so ~2.5 days for 3, or a GPU server/cloud; ideally with explicit solvent and CHARMM36/OPLS-AA or AM1-BCC charges on Linux | Compute time (`python -m ipind2.md_simulation.campaign --ns 100`); methodological upgrade is a team decision |
| R4 | Record ≥ 10 **real** lab results with `POST /lab/results`, run `retrain` and show improvement on a real holdout | Partner lab; supply replay from real training data |
| R5 | Deploy `docker compose` in staging (PostgreSQL + TLS 1.3 + Redis) and a signed attestation in `docs/trl_attestations.json` | Infrastructure team |
| R6 | Independent security review (SEC-01..06) and signed attestation | Independent reviewer |

Manual attestation template:

```json
{"staging_deployment": {"signed_by": "Name", "date": "YYYY-MM-DD", "evidence": "link/report ID"},
 "security_review":    {"signed_by": "Name", "date": "YYYY-MM-DD", "evidence": "link/report ID"}}
```

## What is actually proven

- 345 automated tests pass; the security tests and bug regressions were confirmed with a **mutation check** (see [`MODEL_VALIDATION.md`](MODEL_VALIDATION.md)).
- NFR-01/02/03 and FR-03 on held-out **synthetic** data; NFR-04/05/06/08 with real measurement; NFR-10 in terms of size (not public origin).
- These numbers are not "accuracy on experimental data".
