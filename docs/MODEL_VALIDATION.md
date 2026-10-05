# Validation of Models and NFRs

> **Main warning:** All accuracy numbers in this document were obtained on **synthetic data** (structure→property rules + noise; `data_generation/properties.py`). They show that the pipeline can *learn* the structure-property relationship, not that it is accurate on real nanoparticles. Real accuracy can only be measured with experimental data or a public dataset (LNP-622/LANCE) — see [`TRL_ASSESSMENT.md`](TRL_ASSESSMENT.md).

Model version: `20261003-release-e99cbd64` (`release` profile: 10,000 training samples, 1,500 holdout, 3-member ensemble). Raw report: [`validation_report.json`](validation_report.json). Hardware: Windows 11, 8-core CPU, no GPU, torch 2.1.1.

## Accuracy (synthetic holdout, n = 1500)

| Requirement | Target | Result | Status | Theoretical ceiling* |
|---|---|---|---|---|
| NFR-01 | Size RMSE < 5 nm | **3.27 nm** | ✅ | R² ≤ 0.981 |
| NFR-02 | Zeta RMSE < 2 mV | **1.59 mV** | ✅ | R² ≤ 0.982 |
| NFR-03 | Loading R² > 0.85 | **0.868** | ✅ (small margin) | R² ≤ 0.879 |
| FR-03 | Biological R² > 0.85 for all | Minimum **0.869** (TBR); the rest 0.93–0.98 | ✅ (small margin for TBR) | TBR ≤ 0.882 |

\* Theoretical ceiling = R² of the best possible model that cannot predict the synthetic measurement noise (`noise_ceiling_r2`). **The 0.85 threshold for loading is only ≈0.03 below the data ceiling**; whether NFR-03 holds in this benchmark depends more than anything on the noise level chosen in data generation. On the independent frozen reference set (n=500) the value was 0.862; the 0.006 difference is within sampling error. Therefore NFR-03 is "satisfied but not with strong statistical confidence".

PDI (R² 0.92) and loading content (0.79; ceiling 0.81) are not among the R² > 0.85 requirements.

## Performance (real run on the machine above, without concurrent load)

| Requirement | Target | Measurement | Status |
|---|---|---|---|
| NFR-04 | 100k structures in < 10 minutes | **60.9 s**, 100,000 unique, RDKit validity 100% | ✅ |
| NFR-05 | < 100 ms per structure | **5.4 ms** (both models, ensemble of 3) | ✅ |
| NFR-06 | Optimization of 1000 candidates < 1 hour | **19.5 s** for 1,981 candidates (without MD) | ✅ |
| NFR-08 | ≥ 1M structures | Template-based space of 2,540,780 | ✅ |
| NFR-10 | Seed ≥ 1M reference structures from public databases | 1,000,000 built (556 s) but **0 structures from PubChem/ZINC** | ⚠️ Size yes, public origin no |
| NFR-07 | Availability 99.9% | Not provable by code (infrastructure) | — |
| NFR-09 | Quasi-quantum < 1 kcal/mol | Only an xtb adapter; accuracy not measured | ❌ Not validated |

The NFR-06 measurement **does not include MD** (real MD has not been run).

## Benchmark on real public data (important negative finding)

Dataset: LANTERN/AGILE HeLa — 1100 ionizable lipids with **experimental** transfection efficiency (`data/AGILE.csv`, [MIT repo](https://github.com/AsalMehradfar/LANTERN); the file is not shipped with the repo and is downloaded into `data/external/`). LNP-622 has no direct public file and LANCE is given only on request from the authors; therefore neither was run. 3 random 80/20 splits:

| Model | RMSE | R² |
|---|---|---|
| Constant mean | 3.305 ± 0.102 | −0.008 ± 0.004 |
| RandomForest + Morgan | 2.364 ± 0.070 | **0.482 ± 0.054** |
| LANTERN paper: MLP (Morgan+Expert) | — | 0.8161 |
| Paper: AGILE | — | 0.2655 |
(The paper's split protocol and dataset cleaning are not the same as here; the comparison with the paper is only approximate.)

(The paper's split protocol and dataset cleaning are not the same as here; the comparison with the paper is only approximate.)

**Honest interpretation:** This dataset does not measure the platform's own properties, but it shows that an architecture achieving R²≈0.97 on synthetic data is **weaker than a simple RandomForest baseline** on small real experimental data. Probable reasons (hypotheses, untested): 880 training samples are too few for a graph network; the platform's global features were designed for the chemistry of synthetic lipids and do not discriminate between these similar lipids; lack of substructure fingerprints. Therefore the high synthetic accuracy numbers should not be generalized to real accuracy. For this reason the R2 criterion in `trl.py` is "platform R² ≥ simple baseline" and it **is not satisfied**.

Reproduction: `python -m ipind2.benchmarking.public` (output: `docs/public_benchmark.json` and recorded in `benchmarks/history.json`).

## Requirements without full validation

- **FR-05 (MD ≥ 100 ns, CHARMM36):** The MD engine was not installed. Trajectory analysis (Rg, SASA, S₂, MM-GBSA) was tested with an analytical answer; RDKit conformer sampling is labeled `fidelity="conformer_ensemble"` and never sets `md_complete` to `True`. The OpenMM adapter has not been run.
- **FR-12 comparison with papers:** `published_results.json` must be filled with real citations; we have not supplied any number of our own.

## Validation findings that changed the code

| Finding | Root cause | Fix |
|---|---|---|
| The main data generator produced random descriptors for **all** samples (~150 distinct SMILES out of 100k) | `Descriptors.FractionCsp3` does not exist + a blanket `except` | Rewritten; regression test |
| `generate_for_gnn` assigned the first row's target to all graphs | Bug | Per-sample target; test |
| Quaternary ammonium was almost never generated | Template weight ∝ space size ⇒ one template ≈95% of lipids | Uniform weight + 3 cationic templates |
| NFR-04 failed on first measurement: only 79,200 of 100,000 | Saturation of small templates | Adaptive weighting; regression test that **fails** with the old sampler (confirmed) |
| Loading a saved model raised an error | `load` ignored the subclass signature | Construct from base initialization |
| A concurrent API request could corrupt batches | `GraphEncoder` has per-call state | Lock + concurrency test |
| Impossible constraints ⇒ `KeyError` | Empty table columns | Uniform schema |
| Log and failed-login counter were lost on rollback | Error inside the transaction | Commit before the 401 response + test |

## Negative finding: active learning

Under severe distribution shift (model trained on lipids, new data polymer, 30 results): fine-tuning reduces normalized size error from ≈3.5 to ≈0.9. But **uncertainty-based sampling was not better than random selection** (size: random ≈0.9, pure uncertainty ≈1.7); outliers were selected. The `hybrid` strategy (half random) was ≈ random on size/loading and better and more stable on zeta (1.34 vs. 2.22, 3 seeds). The default is `hybrid`; superiority of pure uncertainty is not claimed.

## How to reproduce

```bash
python -m ipind2.training.train --profile release --out models/v1     # ~35 minutes on CPU
python -m ipind2.training.validate --model-dir models/v1 --out docs/validation_report.json --seed-library
python -m ipind2.trl --tests-passed yes
pytest -q                                                              # 327 tests
```

## Trusting the tests

The security tests (account lockout, RBAC) and the template-saturation regression were checked with a **mutation check**: when the logic was deliberately broken, the related tests failed. (One regression test was initially ineffective without this check and was fixed.)
