"""
SHAP-based interpretability (SHapley Additive exPlanations)

A thin, model-agnostic layer over the ``shap`` library that converts its output to simple
data structures (dataclass) so it can be used independently of the underlying model type (GNN, Transformer,
classical sklearn models) in the physicochemical/biological prediction units (FR-02, FR-03)
.

See docs/SRS.md §4.7 (FR-09).
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:  # pragma: no cover - exercised only when shap is missing
    raise ImportError(
        "Package 'shap' is not installed. Install it with «pip install shap» or via requirements.txt."
    ) from exc


@dataclass
class FeatureAttribution:
    """Contribution of one input feature to a specific prediction."""

    feature: str
    shap_value: float


@dataclass
class ExplanationResult:
    """Full interpretation of a prediction: base value + contribution of each feature."""

    prediction: float
    base_value: float
    attributions: List[FeatureAttribution] = field(default_factory=list)

    def top_features(self, n: int = 5) -> List[FeatureAttribution]:
        """Return the n features with the largest absolute contribution (most important)."""
        return sorted(self.attributions, key=lambda a: abs(a.shap_value), reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "base_value": self.base_value,
            "attributions": {a.feature: a.shap_value for a in self.attributions},
        }


class SHAPExplainer:
    """
    SHAP explainer for a model with scalar output (one target property).

    For models that predict several properties simultaneously (like the multi-task GNN of Unit 2),
    build a separate instance for each output property or use ``explain_multi_output``.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: pd.DataFrame,
        feature_names: Optional[Sequence[str]] = None,
        algorithm: str = "auto",
        max_background_samples: int = 100,
    ):
        """
        Args:
            predict_fn: a function that takes an array/DataFrame (n_samples, n_features) and
                returns a prediction vector (n_samples,).
            background_data: reference data for estimating the base value; for large
                models a small sample (<=100 rows) is recommended.
            feature_names: column names; if not given, read from the columns of background_data.
            algorithm: shap algorithm ('auto', 'permutation', 'exact', ...).
            max_background_samples: maximum number of background samples to control computational cost.
        """
        if len(background_data) == 0:
            raise ValueError("background_data must not be empty")

        self.feature_names = list(feature_names or background_data.columns)
        self._predict_fn = predict_fn
        background = background_data
        if len(background) > max_background_samples:
            background = shap.sample(background, max_background_samples)

        self._explainer = shap.Explainer(predict_fn, background, algorithm=algorithm)

    def explain(self, X: pd.DataFrame) -> List[ExplanationResult]:
        """Compute the SHAP explanation for each row in X."""
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]

        shap_values = self._explainer(X)
        predictions = np.asarray(self._predict_fn(X)).ravel()

        results = []
        for i in range(len(X)):
            attributions = [
                FeatureAttribution(feature=name, shap_value=float(val))
                for name, val in zip(self.feature_names, np.asarray(shap_values.values[i]).ravel())
            ]
            base_value = float(np.asarray(shap_values.base_values[i]).ravel()[0])
            results.append(
                ExplanationResult(
                    prediction=float(predictions[i]),
                    base_value=base_value,
                    attributions=attributions,
                )
            )
        return results


def explain_multi_output(
    predict_fn_per_target: dict,
    background_data: pd.DataFrame,
    X: pd.DataFrame,
    feature_names: Optional[Sequence[str]] = None,
    algorithm: str = "auto",
) -> dict:
    """
    SHAP explanation for a multi-task model by building one explainer for each
    output property — per the need of the physicochemical prediction unit (FR-02) which simultaneously predicts
    ≥7 properties.

    Args:
        predict_fn_per_target: mapping target property name -> scalar prediction function for that property.
        background_data: background data shared among all properties.
        X: samples to be explained.
        feature_names: input column names.
        algorithm: shap algorithm.

    Returns:
        Mapping target property name -> list of ExplanationResult (one per row in X).
    """
    results = {}
    for target_name, predict_fn in predict_fn_per_target.items():
        explainer = SHAPExplainer(
            predict_fn=predict_fn,
            background_data=background_data,
            feature_names=feature_names,
            algorithm=algorithm,
        )
        results[target_name] = explainer.explain(X)
    return results
