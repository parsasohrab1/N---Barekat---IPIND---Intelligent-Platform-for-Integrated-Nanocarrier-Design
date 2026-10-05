"""Prediction interpretability: attention weights, SHAP explanations, confidence scores. See docs/SRS.md §4.7 (FR-09)."""

from .confidence import ConfidenceScore, batch_ensemble_confidence, ensemble_confidence
from .shap_explainer import ExplanationResult, FeatureAttribution, SHAPExplainer, explain_multi_output
from .surrogate import SurrogateExplainer, SurrogateExplanation

__all__ = [
    "ConfidenceScore",
    "ensemble_confidence",
    "batch_ensemble_confidence",
    "ExplanationResult",
    "FeatureAttribution",
    "SHAPExplainer",
    "explain_multi_output",
    "SurrogateExplainer",
    "SurrogateExplanation",
]


def __getattr__(name):
    # AttentionExtractor requires torch; lazy import so importing this package without
    # torch installed (e.g. only for SHAP interpretation on sklearn models) does not fail.
    if name == "AttentionExtractor":
        from .attention import AttentionExtractor

        return AttentionExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
