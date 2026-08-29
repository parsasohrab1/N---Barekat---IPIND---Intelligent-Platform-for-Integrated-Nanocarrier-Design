"""
تفسیرپذیری مبتنی بر SHAP (SHapley Additive exPlanations)

یک لایه نازک و model-agnostic روی کتابخانه ``shap`` که خروجی آن را به ساختارهای
داده‌ای ساده (dataclass) تبدیل می‌کند تا مستقل از نوع مدل زیرین (GNN، Transformer،
مدل‌های کلاسیک sklearn) در واحدهای پیش‌بینی فیزیکوشیمیایی/زیستی (FR-02، FR-03)
قابل استفاده باشد.

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
        "پکیج 'shap' نصب نیست. با «pip install shap» یا از طریق requirements.txt نصب کنید."
    ) from exc


@dataclass
class FeatureAttribution:
    """سهم یک ویژگی ورودی در یک پیش‌بینی مشخص."""

    feature: str
    shap_value: float


@dataclass
class ExplanationResult:
    """تفسیر کامل یک پیش‌بینی: مقدار پایه + سهم هر ویژگی."""

    prediction: float
    base_value: float
    attributions: List[FeatureAttribution] = field(default_factory=list)

    def top_features(self, n: int = 5) -> List[FeatureAttribution]:
        """بازگرداندن n ویژگی با بیشترین قدر مطلق سهم (مهم‌ترین‌ها)."""
        return sorted(self.attributions, key=lambda a: abs(a.shap_value), reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "base_value": self.base_value,
            "attributions": {a.feature: a.shap_value for a in self.attributions},
        }


class SHAPExplainer:
    """
    تفسیرگر SHAP برای یک مدل با خروجی اسکالر (یک ویژگی هدف).

    برای مدل‌هایی که هم‌زمان چند ویژگی پیش‌بینی می‌کنند (مثل GNN چندوظیفه‌ای واحد ۲)،
    یک نمونه جدا برای هر ویژگی خروجی بسازید یا از ``explain_multi_output`` استفاده کنید.
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
            predict_fn: تابعی که یک آرایه/DataFrame (n_samples, n_features) می‌گیرد و
                بردار پیش‌بینی (n_samples,) برمی‌گرداند.
            background_data: داده مرجع برای تخمین مقدار پایه (base value)؛ برای مدل‌های
                بزرگ توصیه می‌شود یک نمونه کوچک (<=100 ردیف) داده شود.
            feature_names: نام ستون‌ها؛ اگر ندهید از ستون‌های background_data خوانده می‌شود.
            algorithm: الگوریتم shap ('auto', 'permutation', 'exact', ...).
            max_background_samples: حداکثر تعداد نمونه پس‌زمینه برای کنترل هزینه محاسباتی.
        """
        if len(background_data) == 0:
            raise ValueError("background_data نباید خالی باشد")

        self.feature_names = list(feature_names or background_data.columns)
        self._predict_fn = predict_fn
        background = background_data
        if len(background) > max_background_samples:
            background = shap.sample(background, max_background_samples)

        self._explainer = shap.Explainer(predict_fn, background, algorithm=algorithm)

    def explain(self, X: pd.DataFrame) -> List[ExplanationResult]:
        """محاسبه تفسیر SHAP برای هر ردیف در X."""
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
    تفسیر SHAP برای یک مدل چندوظیفه‌ای (Multi-Task) با ساختن یک explainer برای هر
    ویژگی خروجی — مطابق نیاز واحد پیش‌بینی فیزیکوشیمیایی (FR-02) که به‌طور هم‌زمان
    ≥۷ ویژگی پیش‌بینی می‌کند.

    Args:
        predict_fn_per_target: نگاشت نام ویژگی هدف -> تابع پیش‌بینی اسکالر برای همان ویژگی.
        background_data: داده پس‌زمینه مشترک بین همه ویژگی‌ها.
        X: نمونه‌هایی که باید تفسیر شوند.
        feature_names: نام ستون‌های ورودی.
        algorithm: الگوریتم shap.

    Returns:
        نگاشت نام ویژگی هدف -> فهرست ExplanationResult (یکی به ازای هر ردیف در X).
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
