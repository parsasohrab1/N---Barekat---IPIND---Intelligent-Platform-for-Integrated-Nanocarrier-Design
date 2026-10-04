"""Unit 4: multi-objective optimization (Pareto-Guided RL). See docs/SRS.md §4.4 (FR-04)."""

from .evaluators import PredictorObjective, oracle_objective
from .objectives import (
    OBJECTIVE_COLUMNS,
    OBJECTIVE_NAMES,
    default_objective_matrix,
    objective_matrix,
    size_fit_score,
)
from .pareto import (
    crowding_distance,
    dominates,
    hypervolume,
    non_dominated_sort,
    pareto_front_mask,
    pareto_ranks,
)
from .pgrl import Constraints, OptimizationResult, ParetoGuidedRL, random_search_front

__all__ = [
    "PredictorObjective",
    "oracle_objective",
    "OBJECTIVE_COLUMNS",
    "OBJECTIVE_NAMES",
    "default_objective_matrix",
    "objective_matrix",
    "size_fit_score",
    "crowding_distance",
    "dominates",
    "hypervolume",
    "non_dominated_sort",
    "pareto_front_mask",
    "pareto_ranks",
    "Constraints",
    "OptimizationResult",
    "ParetoGuidedRL",
    "random_search_front",
]
