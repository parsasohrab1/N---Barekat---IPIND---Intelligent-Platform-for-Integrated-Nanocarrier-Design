"""
تفسیر SHAP برای مدل‌های گرافی واحد ۲/۳ از طریق مدل جانشین (surrogate).

ورودی مدل‌های گرافی ترکیبی از گراف و ویژگی‌های global است و SHAP روی آن‌ها مستقیم قابل‌
اجرا نیست. روش: یک RandomForest روی ویژگی‌های توصیفی (``EXTENDED_NAMES``) آموزش می‌بیند
تا **خروجی خود مدل گرافی** را تقلید کند و SHAP روی آن محاسبه می‌شود.

⚠️ تفسیر، تفسیرِ جانشین است نه خود شبکه. برای همین ``fidelity`` (R² خارج‌از‌کیسه،
OOB) همراه هر تفسیر گزارش می‌شود؛ تفسیر با fidelity پایین نباید جدی گرفته شود.
وزن‌های attention اتمی شبکه (``EnsemblePropertyPredictor.atom_attention``) مکمل
مستقیم و بدون جانشین‌اند.

See docs/SRS.md §4.7 (FR-09).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from ..featurization import EXTENDED_NAMES, extended_matrix, extended_vector
from .shap_explainer import ExplanationResult, FeatureAttribution


@dataclass
class SurrogateExplanation:
    """تفسیر یک پیش‌بینی همراه با کیفیت مدل جانشین."""

    target: str
    explanation: ExplanationResult
    fidelity: float  # R² خارج‌از‌کیسه جانشین نسبت به مدل اصلی

    def to_dict(self, top: int = 5) -> dict:
        return {
            "target": self.target,
            "prediction": self.explanation.prediction,
            "base_value": self.explanation.base_value,
            "fidelity": self.fidelity,
            "top_features": [
                {"feature": a.feature, "shap_value": a.shap_value}
                for a in self.explanation.top_features(top)
            ],
        }


class SurrogateExplainer:
    """
    تفسیرگر SHAP روی جانشین RandomForest برای یک پیش‌بین چندوظیفه‌ای.

    Args:
        predictor: ``EnsemblePropertyPredictor`` آموزش‌دیده.
        pool_smiles: ساختارهای نماینده برای آموزش جانشین (چند صد تا کافی است).
        targets: اهداف مدنظر؛ ``None`` یعنی همه اهداف پیش‌بین.
    """

    def __init__(
        self,
        predictor,
        pool_smiles: Sequence[str],
        targets: Optional[Sequence[str]] = None,
        n_background: int = 40,
        seed: int = 0,
    ):
        features, kept = extended_matrix(list(pool_smiles))
        if len(kept) < 30:
            raise ValueError("برای ساخت جانشین حداقل ۳۰ ساختار معتبر لازم است")
        self.predictor = predictor
        self.targets = list(targets or predictor.target_names)
        self._frame = pd.DataFrame(features, columns=EXTENDED_NAMES)

        predictions = predictor.predict([pool_smiles[i] for i in kept])
        rng = np.random.default_rng(seed)
        background = self._frame.iloc[rng.choice(len(self._frame), size=min(n_background, len(self._frame)), replace=False)]

        self.surrogates: Dict[str, RandomForestRegressor] = {}
        self.fidelity: Dict[str, float] = {}
        self._explainers: Dict[str, "shap.TreeExplainer"] = {}
        for target in self.targets:
            forest = RandomForestRegressor(
                n_estimators=120, max_depth=10, min_samples_leaf=2, oob_score=True,
                random_state=seed, n_jobs=1,
            )
            forest.fit(self._frame.to_numpy(), predictions[target].to_numpy())
            self.surrogates[target] = forest
            self.fidelity[target] = float(forest.oob_score_)
            # جانشین جنگل تصادفی است ⇒ TreeExplainer دقیق و بسیار سریع‌تر از permutation
            self._explainers[target] = shap.TreeExplainer(forest, data=background.to_numpy())

    def explain(self, smiles: str, target: str) -> Optional[SurrogateExplanation]:
        """تفسیر پیش‌بینی ``target`` برای یک SMILES؛ ``None`` اگر SMILES نامعتبر باشد."""
        if target not in self._explainers:
            raise KeyError(f"هدف ناشناخته: {target!r}")
        vector = extended_vector(smiles)
        if vector is None:
            return None
        explainer = self._explainers[target]
        values = np.asarray(explainer.shap_values(vector[None, :], check_additivity=False)).reshape(-1)
        base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        prediction = float(self.surrogates[target].predict(vector[None, :])[0])
        result = ExplanationResult(
            prediction=prediction,
            base_value=base,
            attributions=[
                FeatureAttribution(feature=name, shap_value=float(v)) for name, v in zip(EXTENDED_NAMES, values)
            ],
        )
        return SurrogateExplanation(target=target, explanation=result, fidelity=self.fidelity[target])

    def explain_many(self, smiles: str, targets: Optional[Sequence[str]] = None) -> List[SurrogateExplanation]:
        out = []
        for target in targets or self.targets:
            explanation = self.explain(smiles, target)
            if explanation is not None:
                out.append(explanation)
        return out
