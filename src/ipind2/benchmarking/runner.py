"""
اجرای بنچمارک و ردیابی تاریخچه برای شناسایی افت دقت (regression) بین نسخه‌های مدل.

طبق FR-12: «سنجش خودکار دقت مدل‌ها در برابر دیتاست‌های عمومی مرجع در هر انتشار مدل».
این ماژول برای اجرا در CI طراحی شده — ``assert_no_regression`` می‌تواند در یک pipeline
با کد خروج غیرصفر (exception) استفاده شود تا از انتشار مدلی با دقت پایین‌تر جلوگیری کند.

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
    """نتیجه یک اجرای بنچمارک برای یک دیتاست مرجع مشخص."""

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
    """اجرای یک مدل روی یک دیتاست مرجع و محاسبه RMSE/R²."""
    if target_column not in dataset.columns:
        raise ValueError(f"ستون هدف '{target_column}' در دیتاست موجود نیست")

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
    """ذخیره‌سازی ساده مبتنی بر فایل JSON برای تاریخچه اجرای بنچمارک‌ها."""

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
    """نتیجه مقایسه یک اجرای جدید با آخرین اجرای قبلی برای همان دیتاست."""

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
    مقایسه ``result`` با آخرین نتیجه ثبت‌شده در ``history`` برای همان دیتاست.

    Args:
        rmse_tolerance: حداکثر افزایش مجاز RMSE بدون علامت‌گذاری به‌عنوان regression.
        r2_tolerance: حداکثر کاهش مجاز R² بدون علامت‌گذاری به‌عنوان regression.
    """
    previous = history.latest_for(result.dataset)
    if previous is None:
        return RegressionReport(
            dataset=result.dataset,
            regressed=False,
            delta_rmse=None,
            delta_r2=None,
            message="اجرای قبلی برای مقایسه موجود نیست (اولین بنچمارک این دیتاست)",
        )

    delta_rmse = result.rmse - previous.rmse  # مثبت یعنی بدتر شدن
    delta_r2 = result.r2 - previous.r2  # منفی یعنی بدتر شدن
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
    """مثل ``check_regression`` اما در صورت افت دقت، ``AssertionError`` می‌اندازد (برای CI)."""
    report = check_regression(history, result, rmse_tolerance, r2_tolerance)
    if report.regressed:
        raise AssertionError(f"افت دقت مدل روی دیتاست '{result.dataset}': {report.message}")
    return report
