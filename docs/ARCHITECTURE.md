# IPIND² Architecture

This document records the mapping "SRS unit → code → testable behavior" and the design decisions. Accuracy numbers are in [`MODEL_VALIDATION.md`](MODEL_VALIDATION.md).

## Data flow (SRS §3.2)

```
Persian/English query ──► nlp_interface.parse_query ──► TargetParameters
                                                            │
                                   GenerationCondition ◄────┘
                                          │
 generation.ConditionalStructureGenerator (CVAE+CGAN → retrieval from a pool of valid templates)
                                          │ unique, valid structures
        ┌─────────────────────────────────┴───────────────────────┐
        ▼                                                         ▼
 physicochemical.PhysicochemicalPredictor               biological.BiologicalPredictor
 (MPNN + attention readout, 7 targets)                      (MPNN + Transformer, 7 targets)
        └────────────── ensemble ⇒ mean + σ ────────────────────┘
                                          │
 optimization.ParetoGuidedRL (+ Pareto front of generated structures, hard constraints)
                                          │ 10–20 candidates
 md_simulation.validate_candidates  (the simulation level is honestly recorded)
                                          │
 pipeline.DesignResult: 3–5 candidates + prediction + confidence + SHAP + attention
                                          │
 api (FastAPI) ─ database ─ fair.export ─ active_learning ◄── lab_automation
```

## Unit mapping

| Unit | Package | Algorithm | Design note |
|---|---|---|---|
| 1 Structure generation | `generation` | Conditional CVAE + CGAN on feature space, **nearest-neighbor retrieval** from a template pool | 100% chemical validity guaranteed; the model only learns "which region", not SMILES syntax |
| 2 Physicochemical | `physicochemical` | Dense MPNN + attention readout + global features | No torch-geometric; ensemble for uncertainty |
| 3 Biological | `biological` | MPNN + Transformer encoder with padding mask | Toxicity on 3 cell lines |
| 4 Optimization | `optimization` | REINFORCE on a categorical policy (template → blocks), reward = Pareto rank + Chebyshev with Dirichlet weights | SRS convergence criterion: ΔHV < 1% over 100 iterations |
| 5 Simulation | `md_simulation` | GROMACS/OpenMM adapter + trajectory analysis (Rg, SASA, S₂) | **The MD engine was not run in this repo**; see "Limitations" |
| 6 Active learning | `active_learning` | QBC (ensemble variance) + fine-tuning with masked loss and replay | Default strategy `hybrid` based on experiment (see MODEL_VALIDATION) |
| 7 Interpretability | `interpretability` | Atomic attention + SHAP on a RandomForest surrogate model (fidelity is reported) + ensemble confidence | Surrogate explanation ≠ network explanation; that is why fidelity accompanies the output |
| 8 Natural language | `nlp_interface` | Offline rule-based parser (Persian/English) | Deterministic and with no LLM dependency |
| 9 Lab | `lab_automation` | CSV/REST adapter → `experimental_results` | Direct connection to specific equipment: next phase (per SRS) |
| 10 Benchmark | `benchmarking` | Frozen reference set + history + regression/NFR gate | Public datasets via the user's local file |

## Key decisions and reasons

1. **"Template + block" generation instead of character-level SMILES generation.** FR-01 requires validity > 95%. Free-syntax VAEs on SMILES for long, charged molecules such as lipids usually stay below this threshold; building from valid templates guarantees it structurally. Cost: the generable space is limited to the defined templates (17 templates, ≈ 2.5 million combinations) and for entirely new chemistry a template must be added.
2. **Uniform template weighting within each class.** Weighting by space size made one template ≈ 95% of lipids and effectively eliminated quaternary ammoniums (DOTAP-like); the `test_physical_trends` test revealed it. Now every template is equally likely and cationic templates were added.
3. **Ensemble instead of a Bayesian network.** Simple, reliable, and exactly the "variance between models" measure that SRS §4.6 requires.
4. **Surrogate for SHAP.** The graph model's input is a combination of graph and features; direct SHAP is not possible. A RandomForest surrogate + `TreeExplainer` is accurate and fast (0.03 s instead of 40 s with permutation) and `fidelity` (out-of-bag R²) accompanies every explanation.
5. **Deny-by-default security.** Every endpoint requires a named permission; the role is read from the database, not the token (deactivation/role change takes effect immediately); login failure is deliberately uniform; access denials are also recorded in `audit_log`.
6. **Honesty about simulation level.** `Trajectory.fidelity` and `ValidationReport.md_complete` guarantee that conformer sampling is never reported as "MD ≥ 100 ns" (tested).

## Concurrency

`GraphEncoder` has per-call state (`prepare`/`batch`). All training/prediction operations of each predictor are serialized under an `RLock`; the `TestConcurrency` test checks the correctness of concurrent results. For higher throughput in production, use multiple processes (workers) with independent model loading.

## Known limitations

- Real MD (GROMACS/OpenMM, ≥ 100 ns, CHARMM36) has not been run and validated in this repo; only the adapter and trajectory analysis (with an analytical answer) are tested.
- NFR-09 (quasi-quantum < 1 kcal/mol) is merely an xtb adapter and its accuracy has not been measured.
- The "tissue" target (liver, lung, ...) is only recorded; the current objective functions are not tissue-specific.
- Models were trained on **synthetic data**; accuracy on real experimental data is unknown.
- Windows workaround: the `access violation` message from faulthandler is seen in some pytest sessions on torch 2.1 CPU (non-critical, only under pytest, no effect on results).
