"""Continuous benchmarking of trained models against public reference datasets (e.g. LNP-622, LANCE) on every model release. See docs/SRS.md §4.10 (FR-12)."""

from .datasets import REGISTRY, ReferenceDataset, list_reference_datasets, load_reference_dataset
from .metrics import r_squared, rmse
from .runner import (
    BenchmarkHistory,
    BenchmarkResult,
    RegressionReport,
    assert_no_regression,
    check_regression,
    run_benchmark,
)

__all__ = [
    "rmse",
    "r_squared",
    "ReferenceDataset",
    "REGISTRY",
    "list_reference_datasets",
    "load_reference_dataset",
    "BenchmarkResult",
    "BenchmarkHistory",
    "RegressionReport",
    "run_benchmark",
    "check_regression",
    "assert_no_regression",
]
