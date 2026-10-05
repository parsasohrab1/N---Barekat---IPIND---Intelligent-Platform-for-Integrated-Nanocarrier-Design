"""
Ensemble-Based Confidence Estimation

Computes the uncertainty measure from the prediction variance of several models (ensemble) and converts it to
a normalized confidence score (0..1). The same measure that the active learning unit
(ipind2.active_learning) uses for Uncertainty-Aware Sampling is reused here to
attach a reportable confidence level to every prediction.

See docs/SRS.md §4.7 (FR-09) and §4.6 (FR-06).
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ConfidenceScore:
    """Confidence estimation result for one sample."""

    mean: float
    std: float
    confidence: float  # in the range [0, 1]; higher means lower uncertainty

    def is_high_uncertainty(self, threshold: float = 0.5) -> bool:
        """Whether this sample is a good candidate for active learning sampling."""
        return self.confidence < threshold


def ensemble_confidence(
    predictions: Sequence[float],
    scale: float = 1.0,
) -> ConfidenceScore:
    """
    Compute confidence for one sample from the predictions of several ensemble models.

    Args:
        predictions: prediction of each ensemble model for one sample (at least 2 models).
        scale: expected scale of the standard deviation for normalizing confidence; should be
            tuned to the range of the target value (e.g., nm for size, mV for zeta).

    Returns:
        ConfidenceScore with the mean, standard deviation and normalized confidence score.
    """
    if len(predictions) < 2:
        raise ValueError("ensemble_confidence needs at least 2 model predictions")
    if scale <= 0:
        raise ValueError("scale must be positive")

    arr = np.asarray(predictions, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    # Map the standard deviation to the range (0, 1] with a decreasing exponential function: std=0 -> confidence=1
    confidence = float(np.exp(-std / scale))
    return ConfidenceScore(mean=mean, std=std, confidence=confidence)


def batch_ensemble_confidence(
    predictions: np.ndarray,
    scale: float = 1.0,
) -> list:
    """
    Batch version of ``ensemble_confidence``.

    Args:
        predictions: array of shape (n_models, n_samples).
        scale: normalization scale, as in ``ensemble_confidence``.

    Returns:
        List of ConfidenceScore of length n_samples.
    """
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2:
        raise ValueError("predictions must be a 2D array of shape (n_models, n_samples)")
    return [
        ensemble_confidence(predictions[:, i], scale=scale)
        for i in range(predictions.shape[1])
    ]
