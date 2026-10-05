"""
Pareto tools: dominance, non-dominated sorting, crowding distance and hypervolume.

All objectives are treated as **maximization**; minimization objectives (toxicity, undesirable
size) must be negated/transformed before entering these functions (see
``objectives.py``).

See docs/SRS.md §4.4 (FR-04).
"""

from typing import List

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether ``a`` dominates ``b`` (≥ in all objectives and > in at least one)."""
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_sort(objectives: np.ndarray) -> List[np.ndarray]:
    """
    Fast non-dominated sorting (NSGA-II). ``objectives`` has shape (n, m); all maximization.

    Returns:
        List of fronts; front 0 is the optimal Pareto front.
    """
    objectives = np.asarray(objectives, dtype=float)
    n = objectives.shape[0]
    if n == 0:
        return []

    # Dominance matrix with broadcasting: dominated_by[i, j] = i dominates j
    ge = np.all(objectives[:, None, :] >= objectives[None, :, :], axis=2)
    gt = np.any(objectives[:, None, :] > objectives[None, :, :], axis=2)
    dominated_by = ge & gt
    domination_count = dominated_by.sum(axis=0)

    fronts: List[np.ndarray] = []
    remaining = np.ones(n, dtype=bool)
    counts = domination_count.copy()
    while remaining.any():
        current = np.where(remaining & (counts == 0))[0]
        if current.size == 0:  # guard (should not happen)
            current = np.where(remaining)[0]
        fronts.append(current)
        remaining[current] = False
        counts = counts - dominated_by[current].sum(axis=0)
    return fronts


def pareto_ranks(objectives: np.ndarray) -> np.ndarray:
    """Pareto rank of each sample (1 = first front)."""
    objectives = np.asarray(objectives, dtype=float)
    ranks = np.zeros(objectives.shape[0], dtype=int)
    for rank, front in enumerate(non_dominated_sort(objectives), start=1):
        ranks[front] = rank
    return ranks


def pareto_front_mask(objectives: np.ndarray) -> np.ndarray:
    """Boolean mask of first-front members."""
    return pareto_ranks(objectives) == 1


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """Crowding distance (for selecting diverse candidates on the front)."""
    objectives = np.asarray(objectives, dtype=float)
    n, m = objectives.shape
    distance = np.zeros(n)
    if n <= 2:
        distance[:] = np.inf
        return distance
    for j in range(m):
        order = np.argsort(objectives[:, j])
        distance[order[0]] = distance[order[-1]] = np.inf
        span = objectives[order[-1], j] - objectives[order[0], j]
        if span <= 0:
            continue
        gaps = (objectives[order[2:], j] - objectives[order[:-2], j]) / span
        distance[order[1:-1]] += gaps
    return distance


def hypervolume(front: np.ndarray, reference: np.ndarray, n_samples: int = 20000, seed: int = 0) -> float:
    """
    Hypervolume estimate of the front relative to a reference point (Monte-Carlo; all objectives maximization).

    Sufficient for the FR-04 convergence criterion ("change < 1% over 100 iterations"); exact accuracy is not
    needed because only the *relative change* between iterations is compared (with a fixed seed, noise is eliminated).
    """
    front = np.asarray(front, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if front.size == 0:
        return 0.0
    upper = front.max(axis=0)
    box = upper - reference
    if np.any(box <= 0):
        return 0.0
    rng = np.random.default_rng(seed)
    samples = reference + rng.random((n_samples, front.shape[1])) * box
    covered = np.zeros(n_samples, dtype=bool)
    for point in front:
        covered |= np.all(samples <= point, axis=1)
    return float(covered.mean() * np.prod(box))
