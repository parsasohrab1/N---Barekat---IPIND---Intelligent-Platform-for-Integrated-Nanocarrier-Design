"""
Benchmark execution and history tracking to detect accuracy regression between model versions.

Per FR-12: "automatic measurement of model accuracy against public reference datasets at every model release".
This module is designed to run in CI — ``assert_no_regression`` can be used in a pipeline
with a non-zero exit code (exception) to prevent releasing a model with lower accuracy.

See docs/SRS.md §4.10 (FR-12).
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import r_squared, rmse


@dataclass
class BenchmarkResult:
    """Result of one benchmark run for a given reference dataset."""

    dataset: str
    model_version: str
    n_samples: int
    rmse: float
    r2: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_benchmark(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    dataset: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[Sequence[str]] = None,
    model_version: str = "unversioned",
    dataset_name: str = "unnamed",
) -> BenchmarkResult:
    """Run a model on a reference dataset and compute RMSE/R²."""
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset")

    feature_columns = list(feature_columns) if feature_columns else [
        c for c in dataset.columns if c != target_column
    ]
    y_true = dataset[target_column].to_numpy(dtype=float)
    y_pred = np.asarray(predict_fn(dataset[feature_columns])).ravel()

    return BenchmarkResult(
        dataset=dataset_name,
        model_version=model_version,
        n_samples=len(dataset),
        rmse=rmse(y_true, y_pred),
        r2=r_squared(y_true, y_pred),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


class BenchmarkHistory:
    """Simple JSON-file-based storage for benchmark run history."""

    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> List[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def append(self, result: BenchmarkResult) -> None:
        history = self.load()
        history.append(result.to_dict())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    def latest_for(self, dataset_name: str) -> Optional[BenchmarkResult]:
        matching = [row for row in self.load() if row["dataset"] == dataset_name]
        if not matching:
            return None
        latest = max(matching, key=lambda row: row["timestamp"])
        return BenchmarkResult(**latest)


@dataclass
class RegressionReport:
    """Result of comparing a new run with the last previous run for the same dataset."""

    dataset: str
    regressed: bool
    delta_rmse: Optional[float]
    delta_r2: Optional[float]
    message: str


def check_regression(
    history: BenchmarkHistory,
    result: BenchmarkResult,
    rmse_tolerance: float = 0.0,
    r2_tolerance: float = 0.0,
) -> RegressionReport:
    """
    Compare ``result`` with the last result recorded in ``history`` for the same dataset.

    Args:
        rmse_tolerance: maximum allowed RMSE increase without flagging as regression.
        r2_tolerance: maximum allowed R² decrease without flagging as regression.
    """
    previous = history.latest_for(result.dataset)
    if previous is None:
        return RegressionReport(
            dataset=result.dataset,
            regressed=False,
            delta_rmse=None,
            delta_r2=None,
            message="No previous run available for comparison (first benchmark for this dataset)",
        )

    delta_rmse = result.rmse - previous.rmse  # positive means worse
    delta_r2 = result.r2 - previous.r2  # negative means worse
    regressed = delta_rmse > rmse_tolerance or delta_r2 < -r2_tolerance

    message = (
        f"RMSE: {previous.rmse:.4f} -> {result.rmse:.4f} (Δ={delta_rmse:+.4f}); "
        f"R2: {previous.r2:.4f} -> {result.r2:.4f} (Δ={delta_r2:+.4f})"
    )
    return RegressionReport(
        dataset=result.dataset,
        regressed=regressed,
        delta_rmse=delta_rmse,
        delta_r2=delta_r2,
        message=message,
    )


def assert_no_regression(
    history: BenchmarkHistory,
    result: BenchmarkResult,
    rmse_tolerance: float = 0.0,
    r2_tolerance: float = 0.0,
) -> RegressionReport:
    """Like ``check_regression`` but raises ``AssertionError`` on accuracy drop (for CI)."""
    report = check_regression(history, result, rmse_tolerance, r2_tolerance)
    if report.regressed:
        raise AssertionError(f"Model accuracy drop on dataset '{result.dataset}': {report.message}")
    return report
