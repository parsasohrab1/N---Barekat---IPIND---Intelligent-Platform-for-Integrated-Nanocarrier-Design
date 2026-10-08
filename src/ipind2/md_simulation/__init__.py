"""Unit 5: molecular dynamics simulation & validation (GROMACS/OpenMM). See docs/SRS.md §4.5 (FR-05)."""

from .analysis import (
    block_average_error,
    mmgbsa_delta_g,
    order_parameter_p2,
    radius_of_gyration,
    sasa_shrake_rupley,
    sasa_trajectory,
)
from .engines import (
    CondaOpenMMEngine,
    ConformerEnsembleEngine,
    GromacsEngine,
    MDEngine,
    MDEngineUnavailable,
    OpenMMEngine,
    Trajectory,
    available_engines,
)
from .validation import (
    REQUIRED_MD_NS,
    MDValidation,
    ValidationReport,
    analyze_trajectory,
    validate_candidates,
)

__all__ = [
    "radius_of_gyration",
    "sasa_shrake_rupley",
    "sasa_trajectory",
    "order_parameter_p2",
    "mmgbsa_delta_g",
    "block_average_error",
    "Trajectory",
    "MDEngine",
    "MDEngineUnavailable",
    "CondaOpenMMEngine",
    "ConformerEnsembleEngine",
    "GromacsEngine",
    "OpenMMEngine",
    "available_engines",
    "REQUIRED_MD_NS",
    "MDValidation",
    "ValidationReport",
    "analyze_trajectory",
    "validate_candidates",
]
