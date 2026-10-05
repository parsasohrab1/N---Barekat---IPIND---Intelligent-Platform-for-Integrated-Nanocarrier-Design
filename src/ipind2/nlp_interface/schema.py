"""Data structure of target parameters extracted from the user query. See docs/SRS.md §4.8 (FR-10)."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TargetParameters:
    """
    Nanocarrier design target parameters, equivalent to the input of Unit 1 (structure generation, FR-01).

    ``None`` fields mean that constraint was not mentioned in the query and the default value/value from the user
    must be used; ``unresolved_terms`` shows phrases the parser could not map to any
    known parameter (useful for user feedback or manual review).
    """

    scaffold_type: Optional[str] = None  # 'lipid' | 'polymer' | 'metal'
    target_tissue: Optional[str] = None
    size_range_nm: Optional[Tuple[float, float]] = None
    max_toxicity_ic50: Optional[float] = None
    min_loading_efficiency: Optional[float] = None
    raw_query: str = ""
    unresolved_terms: List[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Whether at least the scaffold type and target tissue are specified (minimum required for the generation unit)."""
        return self.scaffold_type is not None and self.target_tissue is not None

    def to_dict(self) -> dict:
        return {
            "scaffold_type": self.scaffold_type,
            "target_tissue": self.target_tissue,
            "size_range_nm": self.size_range_nm,
            "max_toxicity_ic50": self.max_toxicity_ic50,
            "min_loading_efficiency": self.min_loading_efficiency,
            "unresolved_terms": list(self.unresolved_terms),
        }
