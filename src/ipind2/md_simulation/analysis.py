"""
Trajectory analysis for FR-05 properties: radius of gyration, SASA, order parameter and MM-GBSA.

All functions work on pure numpy arrays (no dependence on a simulation engine) so
they can be validated on the output of any engine (GROMACS/OpenMM/conformer set) and with analytical tests
(e.g., the SASA of a single sphere).

Coordinates: array (n_frames, n_atoms, 3) in angstroms.
"""

from typing import Optional, Sequence

import numpy as np


def radius_of_gyration(coords: np.ndarray, masses: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Radius of gyration of each frame (Å): ``sqrt(Σ m_i |r_i - r_cm|² / Σ m_i)``.

    Args:
        coords: (T, N, 3) or (N, 3).
        masses: mass of each atom; ``None`` means equal mass.
    """
    coords = np.asarray(coords, dtype=float)
    single = coords.ndim == 2
    if single:
        coords = coords[None]
    n_atoms = coords.shape[1]
    weights = np.ones(n_atoms) if masses is None else np.asarray(masses, dtype=float)
    if weights.shape != (n_atoms,):
        raise ValueError("Length of masses must equal the number of atoms")
    total = weights.sum()
    center = (coords * weights[None, :, None]).sum(axis=1) / total
    squared = ((coords - center[:, None, :]) ** 2).sum(axis=2)
    rg = np.sqrt((squared * weights[None, :]).sum(axis=1) / total)
    return rg[0:1] if single else rg


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    """Approximately uniform points on the unit sphere (for Shrake-Rupley)."""
    indices = np.arange(n_points) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def sasa_shrake_rupley(
    coords: np.ndarray,
    radii: Sequence[float],
    probe_radius: float = 1.4,
    n_points: int = 480,
) -> float:
    """
    Solvent-accessible surface area (Å²) of one frame using the Shrake–Rupley method.

    Args:
        coords: (N, 3).
        radii: vdW radius of each atom (Å).
        probe_radius: water probe radius (1.4 Å).
    """
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float) + probe_radius
    n_atoms = coords.shape[0]
    if radii.shape != (n_atoms,):
        raise ValueError("Length of radii must equal the number of atoms")

    unit = _fibonacci_sphere(n_points)
    total = 0.0
    for i in range(n_atoms):
        points = coords[i] + radii[i] * unit  # (P, 3)
        distance_sq = ((points[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)  # (P, N)
        inside = distance_sq < (radii**2)[None, :]
        inside[:, i] = False
        accessible = ~inside.any(axis=1)
        total += 4.0 * np.pi * radii[i] ** 2 * accessible.mean()
    return float(total)


def sasa_trajectory(
    coords: np.ndarray, radii: Sequence[float], probe_radius: float = 1.4, n_points: int = 240
) -> np.ndarray:
    """SASA of each frame (Å²)."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 2:
        coords = coords[None]
    return np.array([sasa_shrake_rupley(frame, radii, probe_radius, n_points) for frame in coords])


def order_parameter_p2(bond_vectors: np.ndarray, director: Optional[np.ndarray] = None) -> float:
    """
    Order parameter ``S = <(3cos²θ − 1)/2>`` (−0.5 to 1).

    Args:
        bond_vectors: (M, 3) bond vectors (e.g., C–C of alkyl tails) from all frames.
        director: reference axis; ``None`` means the principal axis (largest eigenvalue of the Q tensor)
            so that ``S`` is computed independent of the overall sample orientation.
    """
    vectors = np.asarray(bond_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) == 0:
        raise ValueError("bond_vectors must be a non-empty (M, 3) array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1.0, norms)

    if director is None:
        q_tensor = 1.5 * (unit.T @ unit) / len(unit) - 0.5 * np.eye(3)
        return float(np.linalg.eigvalsh(q_tensor)[-1])

    axis = np.asarray(director, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cos_theta = unit @ axis
    return float(np.mean(1.5 * cos_theta**2 - 0.5))


def mmgbsa_delta_g(
    complex_energy: float,
    receptor_energy: float,
    ligand_energy: float,
    entropy_term: float = 0.0,
) -> float:
    """
    MM-GBSA binding free energy: ``ΔG = G_complex − G_receptor − G_ligand − TΔS`` (kcal/mol).

    The inputs are averages of ``E_MM + G_solv`` from the simulation trajectory (MD engine output); this
    function only performs the final arithmetic. ``entropy_term`` is the same ``−TΔS``.
    """
    return float(complex_energy - receptor_energy - ligand_energy + entropy_term)


def block_average_error(series: Sequence[float], n_blocks: int = 5) -> float:
    """Standard error of the mean using block averaging (for reporting MM-GBSA uncertainty)."""
    values = np.asarray(series, dtype=float)
    if len(values) < n_blocks * 2:
        n_blocks = max(len(values) // 2, 1)
    blocks = np.array_split(values, n_blocks)
    means = np.array([b.mean() for b in blocks])
    return float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.0
