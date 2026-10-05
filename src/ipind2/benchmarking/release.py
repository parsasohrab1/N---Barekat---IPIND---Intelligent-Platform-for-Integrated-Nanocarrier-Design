"""
Benchmark of each model release + CI gate (FR-12).

* **Frozen internal reference set** — deterministic synthetic data (fixed seed) that can run in CI without any
  download. Its content fingerprint is recorded in the dataset name; if the data generation rules change, the history
  automatically starts over (comparison with a changed reference is meaningless).
* **Public datasets** (LNP-622, LANCE) — via ``load_reference_dataset`` with a local file.
* **Published papers** — fill ``published_results.json`` yourself with exact citations; this
  module never invents any "published" number of its own.

The ``gate`` gives a non-zero exit code if accuracy drops relative to the previous version **or** NFR thresholds are violated
(for the model release pipeline)::

    python -m ipind2.benchmarking.release --model-dir models/v1 --history benchmarks/history.json
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..biological import BIO_TARGET_COLUMNS
from ..physicochemical import PHYSICO_TARGET_COLUMNS
from .runner import BenchmarkHistory, BenchmarkResult, RegressionReport, check_regression, run_benchmark

FROZEN_SEED = 20240601
FROZEN_SIZE = 500

# NFR thresholds (SRS §2.2) for the release gate
NFR_GATES = {
    "phys_size_nm": ("rmse", 5.0),
    "phys_zeta_potential_mV": ("rmse", 2.0),
    "phys_drug_loading_efficiency_percent": ("r2", 0.85),
}


def frozen_reference_dataset(n: int = FROZEN_SIZE, seed: int = FROZEN_SEED) -> pd.DataFrame:
    """Deterministic reference set (with a seed separate from any training seed)."""
    from ..data_generation.synthetic_data_generator import SyntheticDataGenerator

    return SyntheticDataGenerator(seed).generate_dataset(n, include_pareto_labels=False)


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Content fingerprint of the reference (SMILES + target values) — versions the dataset name."""
    columns = ["smiles", *PHYSICO_TARGET_COLUMNS, *BIO_TARGET_COLUMNS]
    payload = frame[columns].round(6).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


@dataclass
class ReleaseReport:
    results: List[BenchmarkResult]
    regressions: List[RegressionReport]
    nfr_violations: List[str]
    published: Dict[str, dict]

    @property
    def passed(self) -> bool:
        return not self.nfr_violations and not any(r.regressed for r in self.regressions)

    def to_markdown(self) -> str:
        lines = ["| Dataset/target | RMSE | R² | Change vs. previous version |", "|---|---|---|---|"]
        for result, regression in zip(self.results, self.regressions):
            lines.append(f"| {result.dataset} | {result.rmse:.4f} | {result.r2:.4f} | {regression.message} |")
        if self.nfr_violations:
            lines += ["", "**NFR violations:**"] + [f"- {v}" for v in self.nfr_violations]
        return "\n".join(lines)


def load_published_results(path: Optional[str]) -> Dict[str, dict]:
    """Load ``published_results.json`` (key = target, value = {metric, value, citation})."""
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, entry in data.items():
        if "citation" not in entry or "value" not in entry:
            raise ValueError(f"Published entry '{key}' must have value and citation (not accepted without a citation)")
    return data


def benchmark_release(
    bundle,
    history: BenchmarkHistory,
    reference: Optional[pd.DataFrame] = None,
    rel_rmse_tolerance: float = 0.05,
    r2_tolerance: float = 0.02,
    published_path: Optional[str] = None,
    record: bool = True,
) -> ReleaseReport:
    """
    Measure a model version on the reference set, compare with the previous version and check the NFR thresholds.

    Degradation tolerance: RMSE up to ``rel_rmse_tolerance`` (relative) higher and R² up to ``r2_tolerance`` (absolute) lower
    is allowed — the random training noise of the models should not be counted as "regression".
    """
    reference = frozen_reference_dataset() if reference is None else reference
    tag = dataset_fingerprint(reference)
    predictors = {**{c: bundle.physico for c in PHYSICO_TARGET_COLUMNS}, **{c: bundle.bio for c in BIO_TARGET_COLUMNS}}

    results: List[BenchmarkResult] = []
    regressions: List[RegressionReport] = []
    violations: List[str] = []
    for target, predictor in predictors.items():
        column = predictor.target_names.index(target)

        def predict(frame: pd.DataFrame, _p=predictor, _c=column) -> np.ndarray:
            mean = _p.predict(frame["smiles"].tolist()).reindex(range(len(frame)))
            return mean.iloc[:, _c].to_numpy()

        result = run_benchmark(
            predict, reference, target, feature_columns=["smiles"],
            model_version=bundle.version, dataset_name=f"{target}@{tag}",
        )
        previous = history.latest_for(result.dataset)
        tolerance = rel_rmse_tolerance * previous.rmse if previous else 0.0
        regressions.append(check_regression(history, result, rmse_tolerance=tolerance, r2_tolerance=r2_tolerance))
        results.append(result)

        if target in NFR_GATES:
            metric, threshold = NFR_GATES[target]
            value = result.rmse if metric == "rmse" else result.r2
            ok = value < threshold if metric == "rmse" else value > threshold
            if not ok:
                violations.append(f"{target}: {metric}={value:.4f} (threshold {'<' if metric == 'rmse' else '>'} {threshold})")

    if record:
        for result in results:
            history.append(result)
    return ReleaseReport(results, regressions, violations, load_published_results(published_path))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="IPIND² release benchmark gate (FR-12)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--published", default=None)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args(argv)

    from ..training import ModelBundle

    report = benchmark_release(
        ModelBundle.load(args.model_dir), BenchmarkHistory(args.history),
        published_path=args.published, record=not args.no_record,
    )
    print(report.to_markdown())
    print("\nGate result:", "PASS" if report.passed else "FAIL")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
