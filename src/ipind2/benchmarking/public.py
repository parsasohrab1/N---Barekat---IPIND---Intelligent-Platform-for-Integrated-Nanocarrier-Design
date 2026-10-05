"""
Benchmark on a real public dataset (FR-12, the R2 criterion in ``trl.py``).

Registered dataset: **LANTERN/AGILE HeLa** — 1100 ionizable lipids (SMILES) with continuous experimental
transfection efficiency (``Target``); the file ``data/AGILE.csv`` from the MIT repo
https://github.com/AsalMehradfar/LANTERN. The file is not shipped with the repo (download it into ``data/external/``).

⚠️ This is **not the platform's own 14 properties** (size, zeta, ...). It only shows that the learning architecture
(multi-task GNN + global features) gives meaningful regression on real experimental data too and is
comparable with simple baselines and with the numbers reported in the paper. Validation of the platform's own properties
on real data (LNP-622/LANCE/lab) is still required.

Paper numbers (LANTERN, arXiv 2507.03209, abstract): MLP with Morgan+Expert: R²=0.8161, AGILE:
R²=0.2655. The paper's split protocol is not the same as here; the comparison is only **approximate**.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

from ..featurization import extended_matrix
from ..nn.predictor import EnsemblePropertyPredictor
from .datasets import load_reference_dataset
from .metrics import r_squared, rmse
from .runner import BenchmarkHistory, BenchmarkResult

DATASET_NAME = "lantern-hela"
PUBLISHED = {
    "citation": "LANTERN, arXiv:2507.03209",
    "mlp_morgan_expert_r2": 0.8161,
    "agile_r2": 0.2655,
    "caveat": "The split protocol is not the same as here; the comparison is approximate",
}


def _morgan(smiles, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32))
    return np.vstack(rows)


@dataclass
class PublicBenchmarkReport:
    dataset: str
    n_samples: int
    seeds: List[int]
    per_model: Dict[str, Dict[str, List[float]]]

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            model: {
                "rmse_mean": float(np.mean(m["rmse"])), "rmse_sd": float(np.std(m["rmse"])),
                "r2_mean": float(np.mean(m["r2"])), "r2_sd": float(np.std(m["r2"])),
            }
            for model, m in self.per_model.items()
        }

    def to_markdown(self) -> str:
        lines = ["| Model | RMSE | R² |", "|---|---|---|"]
        for model, s in self.summary().items():
            lines.append(f"| {model} | {s['rmse_mean']:.3f} ± {s['rmse_sd']:.3f} | {s['r2_mean']:.3f} ± {s['r2_sd']:.3f} |")
        lines.append(f"| Paper: MLP (Morgan+Expert) | — | {PUBLISHED['mlp_morgan_expert_r2']} |")
        lines.append(f"| Paper: AGILE | — | {PUBLISHED['agile_r2']} |")
        return "\n".join(lines)


def run_public_benchmark(
    csv_path: str,
    seeds=(0, 1, 2),
    test_fraction: float = 0.2,
    n_ensemble: int = 3,
    epochs: int = 80,
    history: Optional[BenchmarkHistory] = None,
    model_version: str = "ipind2-gnn",
) -> PublicBenchmarkReport:
    """Three models on several random splits: constant mean (R²≈0), RandomForest+Morgan, and the platform GNN."""
    frame = load_reference_dataset(DATASET_NAME, csv_path).rename(columns={"SMILES": "smiles"})
    smiles = frame["smiles"].tolist()
    target = frame["Target"].to_numpy(dtype=np.float32)
    fingerprints = _morgan(smiles)
    extended, kept = extended_matrix(smiles)
    if len(kept) != len(smiles):
        raise ValueError("Invalid SMILES in dataset")

    per_model: Dict[str, Dict[str, List[float]]] = {
        name: {"rmse": [], "r2": []} for name in ("mean-baseline", "random-forest (Morgan)", "ipind2-gnn")
    }
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(smiles))
        n_test = int(round(test_fraction * len(smiles)))
        test, train = order[:n_test], order[n_test:]

        baseline = np.full(len(test), target[train].mean())
        forest = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1).fit(fingerprints[train], target[train])
        gnn = EnsemblePropertyPredictor(["Target"], "gnn", n_ensemble=n_ensemble, model_kwargs={"hidden_dim": 64}, max_atoms=96)
        gnn.fit([smiles[i] for i in train], target[train, None], epochs=epochs, seed=seed)
        gnn_pred = gnn.predict([smiles[i] for i in test])["Target"].to_numpy()

        for name, prediction in (
            ("mean-baseline", baseline), ("random-forest (Morgan)", forest.predict(fingerprints[test])), ("ipind2-gnn", gnn_pred),
        ):
            per_model[name]["rmse"].append(rmse(target[test], prediction))
            per_model[name]["r2"].append(r_squared(target[test], prediction))

    if history is not None:
        gnn_scores = per_model["ipind2-gnn"]
        history.append(
            BenchmarkResult(
                dataset=DATASET_NAME, model_version=model_version, n_samples=len(smiles),
                rmse=float(np.mean(gnn_scores["rmse"])), r2=float(np.mean(gnn_scores["r2"])),
                timestamp=pd.Timestamp.now(tz="UTC").isoformat(),
            )
        )
    return PublicBenchmarkReport(DATASET_NAME, len(smiles), list(seeds), per_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="IPIND² public-dataset benchmark")
    parser.add_argument("--csv", default="data/external/AGILE.csv")
    parser.add_argument("--history", default="benchmarks/history.json")
    parser.add_argument("--out", default="docs/public_benchmark.json")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    report = run_public_benchmark(args.csv, epochs=args.epochs, history=BenchmarkHistory(args.history))
    print(report.to_markdown())
    Path(args.out).write_text(
        json.dumps({"dataset": report.dataset, "n": report.n_samples, "seeds": report.seeds,
                    "summary": report.summary(), "published": PUBLISHED}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
