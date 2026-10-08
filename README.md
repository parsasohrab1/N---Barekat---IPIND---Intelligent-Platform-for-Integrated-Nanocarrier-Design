# IPIND — Intelligent Platform for Integrated Nanocarrier Design

Integrated platform for intelligent design of drug nanocarriers: combining deep generative models, graph neural networks, multi-objective reinforcement learning and molecular dynamics simulation to shorten the nanocarrier design cycle from 3-5 years to 6-12 months.

Full Software Requirements Specification (SRS): [`docs/SRS.md`](docs/SRS.md)
Technical comparison with international counterparts (NanoForge, Chemistry42, etc.): [`docs/BENCHMARK.md`](docs/BENCHMARK.md)

## Project structure

```
docs/                   Documentation (SRS, international benchmark, architecture)
docs/patent/            Patent-related materials — encrypted, restricted access (see the README inside)
sql/schema.sql           Database schema
src/ipind2/
  generation/             Unit 1 - Structure generation (Conditional VAE/GAN)
  physicochemical/        Unit 2 - Physicochemical prediction (Multi-Task GNN)
  biological/              Unit 3 - Biological prediction (Multi-Task Transformer/GNN)
  optimization/            Unit 4 - Multi-objective optimization (Pareto-Guided RL)
  md_simulation/           Unit 5 - Molecular dynamics simulation (GROMACS/OpenMM)
  active_learning/         Unit 6 - Active learning and lab feedback
  interpretability/        Unit 7 - Interpretability (Attention + SHAP/LIME)
  nlp_interface/           Unit 8 - Natural-language query interface
  lab_automation/          Unit 9 - Automated lab integration (lab-in-the-loop)
  benchmarking/            Unit 10 - Continuous internal benchmarking against public datasets
  database/                Data layer
  api/                     User interface/API layer
  data_generation/         Synthetic data generator for model training
tests/                    Tests
```

## Status

| | |
|---|---|
| **TRL (computed from evidence)** | **4** — TRL 5 requires real data/MD/deployment; [`docs/TRL_ASSESSMENT.md`](docs/TRL_ASSESSMENT.md) states exactly what is missing |
| Units 1–10 | Implemented and tested (345 tests); real MD is only an adapter |
| Accuracy | On **synthetic data** — [`docs/MODEL_VALIDATION.md`](docs/MODEL_VALIDATION.md) |
| Architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · Operations: [`docs/OPERATIONS.md`](docs/OPERATIONS.md) · FAIR: [`docs/FAIR_MIRIBEL.md`](docs/FAIR_MIRIBEL.md) |

## Quick start

```bash
pip install -r requirements-dev.txt
python -m ipind2.training.train --profile smoke --out models/dev          # ~30 seconds; for real accuracy: --profile release
export IPIND_JWT_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"
export IPIND_ENCRYPTION_KEY="$(python -c 'from ipind2.security import generate_key;print(generate_key())')"
export IPIND_MODEL_DIR=models/dev
python -m ipind2.api.manage create-user admin --role admin                 # then TOTP confirmation (docs/OPERATIONS.md)
python -m ipind2.api.manage serve                                          # dashboard: http://127.0.0.1:8000
```

Programmatic use:

```python
from ipind2.training import ModelBundle
from ipind2.pipeline import DesignPipeline
result = DesignPipeline(ModelBundle.load("models/dev")).design("A lipid nanocarrier for tumor, size between 80 to 120 nm")
print(result.final_candidates[0]["predictions"], result.warnings)
```

## Testing

```bash
pytest -q     # ~8 minutes on CPU; smoke models are trained once
```

> The `joblib ... wmic` messages and occasional `access violation` (faulthandler) on Windows are harmless and do not affect results.
