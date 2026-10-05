"""
FR-04 objective functions: converting predicted properties into an objective matrix (all maximization).

Objectives per SRS §4.4: maximize loading efficiency and cellular uptake, minimize toxicity (i.e.,
maximize IC50) and size within the desired range, and maximize stability.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

OBJECTIVE_NAMES: Tuple[str, ...] = (
    "loading_efficiency",
    "cellular_uptake",
    "safety_ic50",
    "size_fit",
    "stability",
)

# Mapping of each objective to a dataframe column
OBJECTIVE_COLUMNS: Dict[str, str] = {
    "loading_efficiency": "phys_drug_loading_efficiency_percent",
    "cellular_uptake": "bio_cellular_uptake_efficiency_percent",
    "safety_ic50": "bio_cytotoxicity_ic50_ug_ml",
    "size_fit": "phys_size_nm",
    "stability": "phys_colloidal_stability_hours",
}

DEFAULT_SIZE_RANGE_NM = (80.0, 120.0)


def size_fit_score(size_nm, size_range: Tuple[float, float] = DEFAULT_SIZE_RANGE_NM):
    """Score 0..1: 1 inside the desired range, with Gaussian falloff outside it."""
    size = np.asarray(size_nm, dtype=float)
    low, high = size_range
    width = max((high - low) / 2.0, 1.0)
    outside = np.where(size < low, low - size, np.where(size > high, size - high, 0.0))
    return np.exp(-((outside / width) ** 2))


def objective_matrix(
    values: Dict[str, Sequence[float]],
    size_range: Tuple[float, float] = DEFAULT_SIZE_RANGE_NM,
) -> np.ndarray:
    """Build the (n, 5) objective matrix from a dictionary of columns. All columns are maximization."""
    columns = []
    for name in OBJECTIVE_NAMES:
        raw = np.asarray(values[name], dtype=float)
        columns.append(size_fit_score(raw, size_range) if name == "size_fit" else raw)
    return np.column_stack(columns)


def default_objective_matrix(
    df: pd.DataFrame, size_range: Tuple[float, float] = DEFAULT_SIZE_RANGE_NM
) -> np.ndarray:
    """Objective matrix from a dataframe with the standard columns."""
    return objective_matrix({n: df[c].to_numpy() for n, c in OBJECTIVE_COLUMNS.items()}, size_range)
