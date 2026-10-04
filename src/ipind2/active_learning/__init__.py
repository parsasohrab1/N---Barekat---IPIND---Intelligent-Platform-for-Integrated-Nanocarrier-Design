"""Unit 6: active learning & lab feedback loop (Uncertainty-Aware Sampling). See docs/SRS.md §4.6 (FR-06)."""

from .loop import EXPERIMENTAL_TO_TARGET, ActiveLearningLoop, RoundReport
from .sampler import committee_uncertainty, select_samples

__all__ = [
    "EXPERIMENTAL_TO_TARGET",
    "ActiveLearningLoop",
    "RoundReport",
    "committee_uncertainty",
    "select_samples",
]
