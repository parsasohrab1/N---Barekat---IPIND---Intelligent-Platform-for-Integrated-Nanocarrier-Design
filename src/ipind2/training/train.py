"""
Training, evaluation and registration of model versions (the "CI for models" cycle, FR-12).

Usage::

    python -m ipind2.training.train --profile standard --out models/v1

Profiles:
    smoke     ~2 minutes, only to verify the pipeline is healthy (accuracy does not matter)
    standard  ~20–40 minutes on CPU; reportable accuracy against NFRs
    release   ~1–2 hours on CPU; more data to approach the noise ceiling (recommended for release)
    full      ~hours; more data and epochs

The final accuracy is reported on **held-out synthetic data (holdout)**; this is not proof of accuracy on real
experimental data (docs/MODEL_VALIDATION.md).
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..biological import BIO_TARGET_COLUMNS, BiologicalPredictor
from ..data_generation.synthetic_data_generator import SyntheticDataGenerator
from ..generation import CONDITION_COLUMNS, ConditionalStructureGenerator
from ..physicochemical import PHYSICO_TARGET_COLUMNS, PhysicochemicalPredictor
from .bundle import ModelBundle


@dataclass(frozen=True)
class Profile:
    n_train: int
    n_holdout: int
    epochs: int
    n_ensemble: int
    gen_epochs: int
    pool_size: int


PROFILES: Dict[str, Profile] = {
    "smoke": Profile(n_train=300, n_holdout=100, epochs=4, n_ensemble=2, gen_epochs=8, pool_size=1500),
    "standard": Profile(n_train=4000, n_holdout=800, epochs=30, n_ensemble=3, gen_epochs=80, pool_size=20000),
    "release": Profile(n_train=10000, n_holdout=1500, epochs=50, n_ensemble=3, gen_epochs=100, pool_size=40000),
    "full": Profile(n_train=15000, n_holdout=2000, epochs=60, n_ensemble=5, gen_epochs=150, pool_size=60000),
}

# NFR targets measured from model accuracy: (column, metric, threshold, direction)
NFR_CHECKS = (
    ("NFR-01", "phys_size_nm", "rmse", 5.0, "lt"),
    ("NFR-02", "phys_zeta_potential_mV", "rmse", 2.0, "lt"),
    ("NFR-03", "phys_drug_loading_efficiency_percent", "r2", 0.85, "gt"),
)
BIO_R2_TARGET = 0.85  # SRS §4.3: R² > 0.85 for all biological properties


def evaluate_nfr(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    """Compare holdout metrics with NFR/SRS thresholds; the result can be saved in the manifest."""
    report: Dict[str, Any] = {}
    for nfr, column, metric, threshold, direction in NFR_CHECKS:
        value = metrics["physico"][column][metric]
        passed = value < threshold if direction == "lt" else value > threshold
        report[nfr] = {"target": column, "metric": metric, "value": value, "threshold": threshold, "passed": bool(passed)}
    bio = {c: metrics["bio"][c]["r2"] for c in BIO_TARGET_COLUMNS}
    report["FR-03"] = {
        "metric": "r2",
        "threshold": BIO_R2_TARGET,
        "per_target": bio,
        "passed": bool(all(v > BIO_R2_TARGET for v in bio.values())),
    }
    return report


def noise_ceiling_r2(seed: int = 0, n: int = 1500) -> Dict[str, float]:
    """
    Approximate R² ceiling of each synthetic target.

    For identical structures, the noise-free and noisy targets are built and ``R²(noisy,
    noise-free)`` is reported: the best possible R² for any model that cannot predict the measurement
    noise. If the model's R² is close to this ceiling, the model has fully learned the structure
    and the R² shortfall is due to data noise, not the model. (For biological targets that come from noisy
    physicochemical values, the ceiling is approximate.)
    """
    from ..benchmarking.metrics import r_squared
    from ..data_generation.properties import biological_truth, physicochemical_truth
    from ..featurization import extended_dict, parse_smiles
    from ..generation import CombinatorialLibrary

    structures, _ = CombinatorialLibrary(seed=seed).generate(n)
    rng = np.random.default_rng(seed)
    clean_rows, noisy_rows = [], []
    for structure in structures:
        features = extended_dict(parse_smiles(structure.smiles))
        for noise, rows in ((0.0, clean_rows), (1.0, noisy_rows)):
            physico = physicochemical_truth(structure.scaffold_type, features, rng, noise)
            bio = biological_truth(structure.scaffold_type, features, physico, rng, noise)
            rows.append(
                {**{f"phys_{k}": v for k, v in physico.items()}, **{f"bio_{k}": v for k, v in bio.items()}}
            )
    ceilings = {}
    for column in (*PHYSICO_TARGET_COLUMNS, *BIO_TARGET_COLUMNS):
        truth = np.array([row[column] for row in clean_rows])
        observed = np.array([row[column] for row in noisy_rows])
        ceilings[column] = float(r_squared(observed, truth))
    return ceilings


def train_bundle(profile: str = "standard", seed: int = 2024, verbose: bool = True) -> ModelBundle:
    """Full training of the three models and evaluation on the holdout. Version = hash of profile+seed+time."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile!r} (allowed: {', '.join(PROFILES)})")
    cfg = PROFILES[profile]
    started = time.time()

    def log(message: str) -> None:
        if verbose:
            print(f"[{time.time() - started:7.1f}s] {message}", flush=True)

    data = SyntheticDataGenerator(seed).generate_dataset(cfg.n_train + cfg.n_holdout, include_pareto_labels=False)
    train, holdout = data.iloc[: cfg.n_train], data.iloc[cfg.n_train :]
    log(f"Data: {len(train)} train / {len(holdout)} holdout")

    physico = PhysicochemicalPredictor(n_ensemble=cfg.n_ensemble)
    physico.fit(train.smiles.tolist(), train[list(PHYSICO_TARGET_COLUMNS)].to_numpy(), epochs=cfg.epochs, seed=seed)
    log("Unit 2 (GNN) trained")

    bio = BiologicalPredictor(n_ensemble=cfg.n_ensemble)
    bio.fit(train.smiles.tolist(), train[list(BIO_TARGET_COLUMNS)].to_numpy(), epochs=cfg.epochs, seed=seed)
    log("Unit 3 (Transformer+GNN) trained")

    generator = ConditionalStructureGenerator(seed=seed).fit(
        train, pool_size=cfg.pool_size, epochs=cfg.gen_epochs
    )
    log("Unit 1 (CVAE+CGAN) trained")

    metrics = {
        "physico": physico.evaluate(holdout.smiles.tolist(), holdout[list(PHYSICO_TARGET_COLUMNS)].to_numpy()).to_dict("index"),
        "bio": bio.evaluate(holdout.smiles.tolist(), holdout[list(BIO_TARGET_COLUMNS)].to_numpy()).to_dict("index"),
    }
    nfr = evaluate_nfr(metrics)
    metrics["nfr"] = nfr
    metrics["noise_ceiling_r2"] = noise_ceiling_r2(seed)
    fingerprint = hashlib.sha256(f"{profile}:{seed}:{cfg}".encode()).hexdigest()[:8]
    version = f"{time.strftime('%Y%m%d')}-{profile}-{fingerprint}"
    log(f"Version {version}; NFR: " + ", ".join(f"{k}={'✓' if v['passed'] else '✗'}" for k, v in nfr.items()))

    return ModelBundle(
        physico=physico,
        bio=bio,
        generator=generator,
        version=version,
        metrics=metrics,
        info={
            "profile": profile,
            "seed": seed,
            "n_train": cfg.n_train,
            "n_holdout": cfg.n_holdout,
            "data": "synthetic (ipind2.data_generation)",
            "training_seconds": round(time.time() - started, 1),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="IPIND² model training")
    parser.add_argument("--profile", default="standard", choices=sorted(PROFILES))
    parser.add_argument("--out", required=True, help="Output folder of the model bundle")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    bundle = train_bundle(args.profile, args.seed)
    bundle.save(args.out)
    print(json.dumps(bundle.metrics["nfr"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
