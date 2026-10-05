"""
Uncertainty-aware sampling + Query-by-Committee (FR-06).

Uncertainty measure = prediction variance among ensemble members (SRS §4.6). To avoid proposing several very
similar structures for experiment at the same time (the real cost of experiments!), the final selection uses
farthest-first in feature space, which also accounts for diversity.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..featurization import extended_matrix
from ..nn.predictor import EnsemblePropertyPredictor


def committee_uncertainty(
    predictors: Sequence[EnsemblePropertyPredictor], smiles: Sequence[str]
) -> Tuple[np.ndarray, List[int]]:
    """
    Uncertainty score of each structure: mean normalized standard deviation among ensemble members.

    Each target's standard deviation is divided by that target's training standard deviation so that targets with
    different scales (nm vs. PDI) have an equal share.

    Returns:
        (scores of length equal to the number of valid structures, indices of valid structures in the input)
    """
    if not predictors:
        raise ValueError("At least one predictor is required")
    per_predictor: List[np.ndarray] = []
    kept: Optional[List[int]] = None
    for predictor in predictors:
        members, valid = predictor.member_predictions(smiles)
        if len(predictor.models) < 2:
            raise ValueError("Query-by-Committee requires an ensemble with at least 2 members")
        spread = members.std(axis=0, ddof=1) / predictor.scaler.std[None, :]
        per_predictor.append(spread.mean(axis=1))
        kept = valid
    return np.mean(per_predictor, axis=0), kept or []


def _farthest_first(features: np.ndarray, scores: np.ndarray, n: int) -> List[int]:
    """Select n points: start from the most uncertain, then the farthest from those already selected."""
    chosen = [int(np.argmax(scores))]
    distances = np.linalg.norm(features - features[chosen[0]], axis=1)
    while len(chosen) < min(n, len(features)):
        # Combine distance and uncertainty; avoid a zero score so a duplicate point is not returned
        priority = distances * (scores + 1e-9)
        priority[chosen] = -np.inf
        nxt = int(np.argmax(priority))
        chosen.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(features - features[nxt], axis=1))
    return chosen


def select_samples(
    smiles: Sequence[str],
    predictors: Sequence[EnsemblePropertyPredictor],
    n: int,
    strategy: str = "hybrid",
    candidate_factor: int = 5,
    seed: int = 0,
) -> List[int]:
    """
    Select n structures for the next experiment.

    Args:
        strategy: ``'hybrid'`` (default: half random for distribution coverage + half
            diverse-uncertainty), ``'diverse_uncertainty'``, ``'uncertainty'`` or
            ``'random'``. In the severe distribution shift experiment (model trained on lipids, new data
            polymer), pure uncertainty was worse than random (outlier samples); therefore
            the default is hybrid. See docs/MODEL_VALIDATION.md.
        candidate_factor: in diverse mode, first the ``n × factor`` most uncertain are selected
            and then n are chosen from among them with farthest-first.

    Returns:
        Selected indices in the input ``smiles`` list.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if strategy not in ("uncertainty", "diverse_uncertainty", "random", "hybrid"):
        raise ValueError(f"Unknown strategy: {strategy!r}")

    if strategy == "random":
        _, valid = extended_matrix(smiles)
        rng = np.random.default_rng(seed)
        return [int(i) for i in rng.choice(valid, size=min(n, len(valid)), replace=False)]

    if strategy == "hybrid":
        n_random = n // 2
        _, all_valid = extended_matrix(smiles)
        rng = np.random.default_rng(seed)
        random_pick = [int(i) for i in rng.choice(all_valid, size=min(n_random, len(all_valid)), replace=False)]
        remaining = [i for i in all_valid if i not in set(random_pick)]
        sub = [smiles[i] for i in remaining]
        informed = select_samples(
            sub, predictors, n - len(random_pick), "diverse_uncertainty", candidate_factor, seed
        )
        return random_pick + [remaining[i] for i in informed]

    scores, valid = committee_uncertainty(predictors, smiles)
    if not valid:
        return []
    order = np.argsort(-scores)
    if strategy == "uncertainty":
        return [valid[i] for i in order[:n]]

    top = order[: max(n * candidate_factor, n)]
    features, _ = extended_matrix([smiles[valid[i]] for i in top])
    mean, std = features.mean(axis=0), features.std(axis=0)
    features = (features - mean) / np.where(std < 1e-6, 1.0, std)
    picks = _farthest_first(features, scores[top], n)
    return [valid[int(top[p])] for p in picks]
