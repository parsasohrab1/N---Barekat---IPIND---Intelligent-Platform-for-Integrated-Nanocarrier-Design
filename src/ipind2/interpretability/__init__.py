"""Prediction interpretability: attention weights, SHAP explanations, confidence scores. See docs/SRS.md §4.7 (FR-09)."""

from .confidence import ConfidenceScore, batch_ensemble_confidence, ensemble_confidence
from .shap_explainer import ExplanationResult, FeatureAttribution, SHAPExplainer, explain_multi_output

__all__ = [
    "ConfidenceScore",
    "ensemble_confidence",
    "batch_ensemble_confidence",
    "ExplanationResult",
    "FeatureAttribution",
    "SHAPExplainer",
    "explain_multi_output",
]


def __getattr__(name):
    # AttentionExtractor نیاز به torch دارد؛ import تنبل تا وارد کردن این پکیج بدون
    # torch نصب‌شده (مثلاً فقط برای تفسیر SHAP روی مدل‌های sklearn) شکست نخورد.
    if name == "AttentionExtractor":
        from .attention import AttentionExtractor

        return AttentionExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
