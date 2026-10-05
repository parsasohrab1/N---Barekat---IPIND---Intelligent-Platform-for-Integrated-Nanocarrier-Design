"""
Nanoparticle record schema conforming to FAIR principles and aligned with the MIRIBEL three-category classification (FR-13).

MIRIBEL (Minimum Information Reporting in Bio–Nano Experimental Literature) requires reporting of the minimum
information in three categories: **material characterization**, **biological characterization** and **experimental protocol**.
The fields of this module are the project's reading of those three categories and must be reconciled with the official
published checklist (Faria et al., Nature Nanotechnology 2018) by a specialist; therefore
the claim is "aligned with MIRIBEL", not "certified/officially compliant" (see docs/FAIR_MIRIBEL.md).

Reusability principle (R): every value explicitly has a ``provenance`` — ``predicted`` (model and
its version) or ``measured`` (measurement method) — so that a prediction is never mistaken for a
measurement; this is vital for interaction with regulators.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# MIRIBEL categories and required fields of each category
MATERIAL_FIELDS = (
    "smiles",
    "scaffold_type",
    "hydrodynamic_size_nm",
    "zeta_potential_mV",
    "pdi",
    "drug_loading_efficiency_percent",
)
BIOLOGICAL_FIELDS = (
    "cell_line",
    "cytotoxicity_ic50_ug_ml",
    "cellular_uptake_efficiency_percent",
)
PROTOCOL_FIELDS = (
    "characterization_method",  # e.g., DLS
    "medium",
    "temperature_c",
    "exposure_time_h",
    "replicates",
    "operator",
    "date",
)
MIRIBEL_FIELDS = {
    "material": MATERIAL_FIELDS,
    "biological": BIOLOGICAL_FIELDS,
    "protocol": PROTOCOL_FIELDS,
}

# Units (QUDT reference) so the output is machine-readable and unambiguous
UNITS = {
    "hydrodynamic_size_nm": "http://qudt.org/vocab/unit/NanoM",
    "zeta_potential_mV": "http://qudt.org/vocab/unit/MilliV",
    "temperature_c": "http://qudt.org/vocab/unit/DEG_C",
    "exposure_time_h": "http://qudt.org/vocab/unit/HR",
    "cytotoxicity_ic50_ug_ml": "http://qudt.org/vocab/unit/MicroGM-PER-MilliL",
    "drug_loading_efficiency_percent": "http://qudt.org/vocab/unit/PERCENT",
    "cellular_uptake_efficiency_percent": "http://qudt.org/vocab/unit/PERCENT",
    "pdi": "http://qudt.org/vocab/unit/UNITLESS",
}

PROVENANCE_VALUES = ("predicted", "measured")


@dataclass
class Observation:
    """A value together with its provenance."""

    value: Any
    provenance: str  # 'predicted' | 'measured'
    method: Optional[str] = None  # measurement method or model name
    model_version: Optional[str] = None
    uncertainty: Optional[float] = None  # ensemble standard deviation or measurement error

    def __post_init__(self):
        if self.provenance not in PROVENANCE_VALUES:
            raise ValueError(f"provenance must be one of {PROVENANCE_VALUES}, not {self.provenance!r}")
        if self.provenance == "predicted" and not self.model_version:
            raise ValueError("A predicted value must have a model_version (traceability)")


@dataclass
class NanoparticleRecord:
    """Nanoparticle/nanocarrier record with the three MIRIBEL categories."""

    identifier: str  # persistent identifier (urn:uuid:...)
    material: Dict[str, Observation] = field(default_factory=dict)
    biological: Dict[str, Observation] = field(default_factory=dict)
    protocol: Dict[str, Observation] = field(default_factory=dict)
    license: str = "CC-BY-4.0"
    created: Optional[str] = None
    notes: List[str] = field(default_factory=list)
