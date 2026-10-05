"""
SHAP interpretation for the graph models of Units 2/3 via a surrogate model.

The input of the hybrid graph models is a combination of graph and global features, and SHAP cannot be run on them
directly. Method: a RandomForest is trained on the descriptive features (``EXTENDED_NAMES``)
to mimic the **output of the graph model itself**, and SHAP is computed on it.

⚠️ The interpretation is that of the surrogate, not the network itself. For this reason ``fidelity`` (out-of-bag R²,
OOB) is reported with every interpretation; an interpretation with low fidelity should not be taken seriously.
The network's atomic attention weights (``EnsemblePropertyPredictor.atom_attention``) are a direct
complement without a surrogate.

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
    """Interpretation of a prediction together with the quality of the surrogate model."""

    target: str
    explanation: ExplanationResult
    fidelity: float  # out-of-bag R² of the surrogate relative to the main model

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
    SHAP explainer on a RandomForest surrogate for a multi-task predictor.

    Args:
        predictor: trained ``EnsemblePropertyPredictor``.
        pool_smiles: representative structures for training the surrogate (a few hundred suffice).
        targets: targets of interest; ``None`` means all predictor targets.
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
            raise ValueError("At least 30 valid structures are required to build the surrogate")
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
            # The surrogate is a random forest ⇒ TreeExplainer is exact and much faster than permutation
            self._explainers[target] = shap.TreeExplainer(forest, data=background.to_numpy())

    def explain(self, smiles: str, target: str) -> Optional[SurrogateExplanation]:
        """Interpretation of the ``target`` prediction for one SMILES; ``None`` if the SMILES is invalid."""
        if target not in self._explainers:
            raise KeyError(f"Unknown target: {target!r}")
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
