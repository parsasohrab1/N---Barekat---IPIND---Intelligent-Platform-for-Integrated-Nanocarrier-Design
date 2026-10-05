"""
Final validation of candidates with simulation (FR-05).

For each candidate a simulation trajectory is obtained and the four SRS §4.5 properties are extracted:
energy (MM-GBSA or MMFF proxy), radius of gyration, SASA and order parameter.

``MDValidation.fidelity`` always records the real simulation level. The final report
(``ValidationReport.md_complete``) is ``True`` only when **all** candidates have been simulated with a real
MD engine and for at least ``min_ns`` (default 100 ns, per the SRS).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem

from .analysis import order_parameter_p2, radius_of_gyration, sasa_trajectory
from .engines import ConformerEnsembleEngine, MDEngine, MDEngineUnavailable, Trajectory

REQUIRED_MD_NS = 100.0

_PT = Chem.GetPeriodicTable()


def _vdw_radii(elements: Sequence[str]) -> np.ndarray:
    return np.array([_PT.GetRvdw(_PT.GetAtomicNumber(e)) for e in elements], dtype=float)


def _masses(elements: Sequence[str]) -> np.ndarray:
    return np.array([_PT.GetAtomicWeight(_PT.GetAtomicNumber(e)) for e in elements], dtype=float)


@dataclass
class MDValidation:
    """Validation result of one candidate."""

    smiles: str
    engine: str
    fidelity: str
    simulated_ns: float
    n_frames: int
    rg_mean: float
    rg_std: float
    sasa_mean: float
    sasa_std: float
    order_parameter: Optional[float]
    energy_mean_kcal: Optional[float]
    energy_label: str
    stable: bool
    notes: List[str] = field(default_factory=list)

    @property
    def is_real_md(self) -> bool:
        return self.fidelity == "md"

    @property
    def meets_sim_time(self) -> bool:
        return self.is_real_md and self.simulated_ns >= REQUIRED_MD_NS

    def to_dict(self) -> Dict:
        data = dict(self.__dict__)
        data["notes"] = list(self.notes)
        data["is_real_md"] = self.is_real_md
        data["meets_sim_time"] = self.meets_sim_time
        return data


@dataclass
class ValidationReport:
    results: List[MDValidation]
    skipped: List[str] = field(default_factory=list)

    @property
    def md_complete(self) -> bool:
        """Whether all candidates were validated with real MD and ≥ 100 ns (SRS requirement)."""
        return bool(self.results) and not self.skipped and all(r.meets_sim_time for r in self.results)

    def stable(self) -> List[MDValidation]:
        return [r for r in self.results if r.stable]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([r.to_dict() for r in self.results])


def analyze_trajectory(
    trajectory: Trajectory, max_rg_cv: float = 0.35
) -> MDValidation:
    """Extract Rg, SASA, order parameter and energy from one trajectory.

    ``stable`` means the coefficient of variation of Rg is less than ``max_rg_cv`` — a simple screening criterion for
    "the structure did not collapse/unfold during the simulation", not a calibrated chemical threshold.
    """
    elements = trajectory.elements
    radii = _vdw_radii(elements)
    masses = _masses(elements)

    rg = radius_of_gyration(trajectory.coords, masses)
    sasa = sasa_trajectory(trajectory.coords, radii, n_points=120)

    carbon = {i for i, e in enumerate(elements) if e == "C"}
    vectors = [
        trajectory.coords[:, j, :] - trajectory.coords[:, i, :]
        for i, j in trajectory.bonds
        if i in carbon and j in carbon
    ]
    order = (
        order_parameter_p2(np.concatenate(vectors, axis=0)) if vectors else None
    )

    energies = trajectory.energies_kcal
    energy_mean = None
    if energies is not None and np.isfinite(energies).any():
        energy_mean = float(np.nanmean(energies))
    energy_label = (
        "mmff_conformer_energy" if trajectory.fidelity == "conformer_ensemble" else "potential_energy"
    )

    rg_mean = float(rg.mean())
    rg_cv = float(rg.std() / rg_mean) if rg_mean > 0 else float("inf")
    notes: List[str] = []
    if trajectory.fidelity != "md":
        notes.append("The result is from conformer sampling, not MD; insufficient for the FR-05 requirement.")
    elif trajectory.simulated_ns < REQUIRED_MD_NS:
        notes.append(f"Simulation duration {trajectory.simulated_ns} ns < {REQUIRED_MD_NS} ns (SRS requirement).")

    return MDValidation(
        smiles=trajectory.smiles,
        engine=trajectory.engine,
        fidelity=trajectory.fidelity,
        simulated_ns=trajectory.simulated_ns,
        n_frames=trajectory.n_frames,
        rg_mean=rg_mean,
        rg_std=float(rg.std()),
        sasa_mean=float(sasa.mean()),
        sasa_std=float(sasa.std()),
        order_parameter=order,
        energy_mean_kcal=energy_mean,
        energy_label=energy_label,
        stable=rg_cv < max_rg_cv,
        notes=notes,
    )


def validate_candidates(
    smiles_list: Sequence[str],
    engine: Optional[MDEngine] = None,
    duration_ns: float = REQUIRED_MD_NS,
    max_candidates: int = 10,
    fallback_to_conformers: bool = False,
) -> ValidationReport:
    """
    Validation of the top 5–10 candidates (SRS §4.5).

    Args:
        engine: simulation engine; ``None`` means ``ConformerEnsembleEngine``.
        fallback_to_conformers: if the MD engine is unavailable, use conformer sampling
            instead of failing. The result still has ``fidelity='conformer_ensemble'`` and does not
            set ``ValidationReport.md_complete`` to ``True``.
    """
    engine = engine or ConformerEnsembleEngine()
    fallback = ConformerEnsembleEngine()
    results: List[MDValidation] = []
    skipped: List[str] = []

    for smiles in list(smiles_list)[:max_candidates]:
        try:
            trajectory = engine.simulate(smiles, duration_ns=duration_ns)
        except MDEngineUnavailable:
            if not fallback_to_conformers:
                raise
            trajectory = fallback.simulate(smiles)
        except (ValueError, RuntimeError):
            skipped.append(smiles)
            continue
        results.append(analyze_trajectory(trajectory))
    return ValidationReport(results=results, skipped=skipped)
