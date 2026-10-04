"""Unit 1: molecular structure generation (Conditional VAE/GAN ensemble). See docs/SRS.md §4.1 (FR-01)."""

from .building_blocks import SCAFFOLD_TYPES, TEMPLATES, theoretical_library_size
from .conditional import (
    CONDITION_COLUMNS,
    ConditionalStructureGenerator,
    GenerationCondition,
)
from .library import (
    CombinatorialLibrary,
    GeneratedStructure,
    LibraryStats,
    generate_library,
    structures_to_dataframe,
)
from .models import ConditionalGAN, ConditionalVAE

__all__ = [
    "SCAFFOLD_TYPES",
    "TEMPLATES",
    "theoretical_library_size",
    "CONDITION_COLUMNS",
    "ConditionalStructureGenerator",
    "GenerationCondition",
    "CombinatorialLibrary",
    "GeneratedStructure",
    "LibraryStats",
    "generate_library",
    "structures_to_dataframe",
    "ConditionalVAE",
    "ConditionalGAN",
]
