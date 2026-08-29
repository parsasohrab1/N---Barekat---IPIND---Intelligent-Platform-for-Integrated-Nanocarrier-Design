"""
تخمین اطمینان مبتنی بر ensemble (Ensemble-Based Confidence Estimation)

معیار عدم‌قطعیت را از واریانس پیش‌بینی چند مدل (ensemble) محاسبه می‌کند و آن را به
یک نمره اطمینان نرمال‌شده (0..1) تبدیل می‌کند. همان معیاری که واحد یادگیری فعال
(ipind2.active_learning) برای Uncertainty-Aware Sampling استفاده می‌کند، اینجا برای
همراه‌کردن هر پیش‌بینی با یک درجه اطمینان قابل‌گزارش بازاستفاده می‌شود.

See docs/SRS.md §4.7 (FR-09) and §4.6 (FR-06).
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ConfidenceScore:
    """نتیجه تخمین اطمینان برای یک نمونه."""

    mean: float
    std: float
    confidence: float  # در بازه [0, 1]؛ هرچه بالاتر، عدم‌قطعیت کمتر

    def is_high_uncertainty(self, threshold: float = 0.5) -> bool:
        """آیا این نمونه کاندیدای مناسبی برای نمونه‌برداری یادگیری فعال است."""
        return self.confidence < threshold


def ensemble_confidence(
    predictions: Sequence[float],
    scale: float = 1.0,
) -> ConfidenceScore:
    """
    محاسبه اطمینان برای یک نمونه از روی پیش‌بینی چند مدل ensemble.

    Args:
        predictions: پیش‌بینی هر یک از مدل‌های ensemble برای یک نمونه (حداقل ۲ مدل).
        scale: مقیاس مورد انتظار انحراف‌معیار برای نرمال‌سازی confidence؛ باید متناسب
            با دامنه مقدار هدف تنظیم شود (مثلاً nm برای اندازه، mV برای زتا).

    Returns:
        ConfidenceScore با میانگین، انحراف‌معیار و نمره اطمینان نرمال‌شده.
    """
    if len(predictions) < 2:
        raise ValueError("ensemble_confidence needs at least 2 model predictions")
    if scale <= 0:
        raise ValueError("scale must be positive")

    arr = np.asarray(predictions, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    # نگاشت انحراف‌معیار به بازه (0, 1] با یک تابع نمایی نزولی: std=0 -> confidence=1
    confidence = float(np.exp(-std / scale))
    return ConfidenceScore(mean=mean, std=std, confidence=confidence)


def batch_ensemble_confidence(
    predictions: np.ndarray,
    scale: float = 1.0,
) -> list:
    """
    نسخه دسته‌ای (batch) از ``ensemble_confidence``.

    Args:
        predictions: آرایه به شکل (n_models, n_samples).
        scale: مقیاس نرمال‌سازی، مطابق ``ensemble_confidence``.

    Returns:
        فهرستی از ConfidenceScore به طول n_samples.
    """
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2:
        raise ValueError("predictions must be a 2D array of shape (n_models, n_samples)")
    return [
        ensemble_confidence(predictions[:, i], scale=scale)
        for i in range(predictions.shape[1])
    ]
