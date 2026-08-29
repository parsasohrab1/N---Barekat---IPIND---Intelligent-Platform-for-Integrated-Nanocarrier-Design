"""معیارهای دقت مطابق NFR-01..NFR-03 در docs/SRS.md."""

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true{y_true.shape} vs y_pred{y_pred.shape}")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true, y_pred) -> float:
    """ضریب تعیین R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true{y_true.shape} vs y_pred{y_pred.shape}")

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot
