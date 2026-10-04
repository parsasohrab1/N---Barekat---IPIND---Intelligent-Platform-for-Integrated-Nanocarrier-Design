"""
توابع هدف FR-04: تبدیل ویژگی‌های پیش‌بینی‌شده به ماتریس اهداف (همه حداکثرسازی).

اهداف مطابق SRS §4.4: حداکثرسازی کارایی بارگذاری و نفوذ سلولی، حداقل‌سازی سمیت (یعنی
حداکثرسازی IC50) و اندازه در بازه مطلوب، و حداکثرسازی پایداری.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

OBJECTIVE_NAMES: Tuple[str, ...] = (
    "loading_efficiency",
    "cellular_uptake",
    "safety_ic50",
    "size_fit",
    "stability",
)

# نگاشت هر هدف به ستون دیتافریم
OBJECTIVE_COLUMNS: Dict[str, str] = {
    "loading_efficiency": "phys_drug_loading_efficiency_percent",
    "cellular_uptake": "bio_cellular_uptake_efficiency_percent",
    "safety_ic50": "bio_cytotoxicity_ic50_ug_ml",
    "size_fit": "phys_size_nm",
    "stability": "phys_colloidal_stability_hours",
}

DEFAULT_SIZE_RANGE_NM = (80.0, 120.0)


def size_fit_score(size_nm, size_range: Tuple[float, float] = DEFAULT_SIZE_RANGE_NM):
    """امتیاز ۰..۱: ۱ داخل بازه مطلوب، با افت گاوسی بیرون آن."""
    size = np.asarray(size_nm, dtype=float)
    low, high = size_range
    width = max((high - low) / 2.0, 1.0)
    outside = np.where(size < low, low - size, np.where(size > high, size - high, 0.0))
    return np.exp(-((outside / width) ** 2))


def objective_matrix(
    values: Dict[str, Sequence[float]],
    size_range: Tuple[float, float] = DEFAULT_SIZE_RANGE_NM,
) -> np.ndarray:
    """ساخت ماتریس (n, 5) اهداف از دیکشنری ستون‌ها. همه ستون‌ها حداکثرسازی‌اند."""
    columns = []
    for name in OBJECTIVE_NAMES:
        raw = np.asarray(values[name], dtype=float)
        columns.append(size_fit_score(raw, size_range) if name == "size_fit" else raw)
    return np.column_stack(columns)


def default_objective_matrix(
    df: pd.DataFrame, size_range: Tuple[float, float] = DEFAULT_SIZE_RANGE_NM
) -> np.ndarray:
    """ماتریس اهداف از دیتافریم با ستون‌های استاندارد."""
    return objective_matrix({n: df[c].to_numpy() for n, c in OBJECTIVE_COLUMNS.items()}, size_range)
