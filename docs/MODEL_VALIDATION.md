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

## Benchmark on real public data

Dataset: LANTERN/AGILE HeLa — 1100 ionizable lipids with **experimental** transfection efficiency (`data/AGILE.csv`, [MIT repo](https://github.com/AsalMehradfar/LANTERN); the file is not shipped with the repo, download it into `data/external/`). LNP-622 has no direct public file and LANCE is available only on request from its authors; neither was run.

### Path to improvement (including the failed steps)

1. **First version (no fingerprints):** R² = **0.30** — worse than RandomForest+Morgan (0.48). Negative finding: the 0.97 accuracy on synthetic data did not generalise to real data.
2. **Fix:** count-based Morgan fingerprints (log1p) were appended to the global features (`GraphEncoder(fp_bits=...)`; default 0, so existing models and saved bundles are unchanged and still load).
3. **Config chosen on a validation split** (20% of training, seed 0; the test set was not looked at): fp ∈ {0, 512, 1024, 2048, 4096}, dropout ∈ {0.05 … 0.5}. Validation R²: 0.26 without fingerprints → 0.47–0.54 with them; best `fp=2048, dropout=0.3` (0.54). Differences among the fingerprint configs are noise-sized; the gain comes from *having* fingerprints, not from fine tuning.
4. **Final evaluation on 5 fresh splits** (seeds 1–5, not used for tuning), 3-member ensemble:

| Model | RMSE | R² |
|---|---|---|
| Constant mean | 3.368 ± 0.108 | −0.011 ± 0.013 |
| RandomForest + Morgan | 2.456 ± 0.131 | 0.459 ± 0.069 |
| Platform GNN **without** fingerprints | 2.807 ± 0.103 | 0.297 ± 0.028 |
| **Platform GNN + Morgan(2048)** | **2.345 ± 0.116** | **0.507 ± 0.053** |
| LANTERN paper: MLP (Morgan+Expert) | — | 0.8161 |
| Paper: AGILE | — | 0.2655 |

Paired over the 5 splits: GNN+Morgan beats the RandomForest on **5 of 5** splits (mean R² difference +0.049, sd 0.022, paired t-test p = 0.008) and beats the no-fingerprint GNN by about +0.21 (p = 0.004). n = 5 is small; the advantage over the RandomForest is **small but consistent**.

### What is not claimed

- It is still **far below** the 0.82 reported by the paper. The paper's split/cleaning protocol differs and we did not investigate the gap; parity with the paper is not claimed.
- This dataset does not measure the platform's own properties (size, zeta, IC50, ...); their validity on real data remains unknown. Morgan fingerprints were only enabled in this experiment: the `release` bundle was trained without them (retraining and re-validation are needed).
- The R2 criterion in `trl.py` is only "≥ simple RandomForest baseline" and is now met; that is a low bar.

Reproduce: `python -m ipind2.benchmarking.public` (output: `docs/public_benchmark.json` with per-split results, recorded in `benchmarks/history.json`).

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
pytest -q                                                              # 334 tests
```

## Trusting the tests

The security tests (account lockout, RBAC) and the template-saturation regression were checked with a **mutation check**: when the logic was deliberately broken, the related tests failed. (One regression test was initially ineffective without this check and was fixed.)
