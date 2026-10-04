"""
تحلیل مسیر (trajectory) برای خواص FR-05: شعاع ژیراسیون، SASA، پارامتر ترازوی سفارش و MM-GBSA.

همه توابع روی آرایه‌های numpy خالص کار می‌کنند (بدون وابستگی به موتور شبیه‌سازی) تا
بتوان آن‌ها را روی خروجی هر موتور (GROMACS/OpenMM/مجموعه کانفورمر) و با تست تحلیلی
(مثلاً SASA یک کره منفرد) اعتبارسنجی کرد.

مختصات: آرایه (n_frames, n_atoms, 3) بر حسب آنگستروم.
"""

from typing import Optional, Sequence

import numpy as np


def radius_of_gyration(coords: np.ndarray, masses: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    شعاع ژیراسیون هر فریم (Å): ``sqrt(Σ m_i |r_i - r_cm|² / Σ m_i)``.

    Args:
        coords: (T, N, 3) یا (N, 3).
        masses: جرم هر اتم؛ ``None`` یعنی جرم یکسان.
    """
    coords = np.asarray(coords, dtype=float)
    single = coords.ndim == 2
    if single:
        coords = coords[None]
    n_atoms = coords.shape[1]
    weights = np.ones(n_atoms) if masses is None else np.asarray(masses, dtype=float)
    if weights.shape != (n_atoms,):
        raise ValueError("طول masses باید برابر تعداد اتم‌ها باشد")
    total = weights.sum()
    center = (coords * weights[None, :, None]).sum(axis=1) / total
    squared = ((coords - center[:, None, :]) ** 2).sum(axis=2)
    rg = np.sqrt((squared * weights[None, :]).sum(axis=1) / total)
    return rg[0:1] if single else rg


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    """نقاط تقریباً یکنواخت روی کره واحد (برای Shrake-Rupley)."""
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
    سطح قابل‌دسترس حلال (Å²) یک فریم به روش Shrake–Rupley.

    Args:
        coords: (N, 3).
        radii: شعاع vdW هر اتم (Å).
        probe_radius: شعاع پروب آب (۱٫۴ Å).
    """
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float) + probe_radius
    n_atoms = coords.shape[0]
    if radii.shape != (n_atoms,):
        raise ValueError("طول radii باید برابر تعداد اتم‌ها باشد")

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
    """SASA هر فریم (Å²)."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 2:
        coords = coords[None]
    return np.array([sasa_shrake_rupley(frame, radii, probe_radius, n_points) for frame in coords])


def order_parameter_p2(bond_vectors: np.ndarray, director: Optional[np.ndarray] = None) -> float:
    """
    پارامتر ترازوی سفارش ``S = <(3cos²θ − 1)/2>`` (−۰٫۵ تا ۱).

    Args:
        bond_vectors: (M, 3) بردارهای پیوند (مثلاً C–C دم‌های آلکیلی) از همه فریم‌ها.
        director: محور مرجع؛ ``None`` یعنی محور اصلی (بزرگ‌ترین مقدار ویژه تانسور Q)
            تا ``S`` مستقل از جهت‌گیری کلی نمونه محاسبه شود.
    """
    vectors = np.asarray(bond_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) == 0:
        raise ValueError("bond_vectors باید آرایه (M, 3) غیرخالی باشد")
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
    انرژی آزاد اتصال MM-GBSA: ``ΔG = G_complex − G_receptor − G_ligand − TΔS`` (kcal/mol).

    ورودی‌ها میانگین‌های ``E_MM + G_solv`` از مسیر شبیه‌سازی (خروجی موتور MD) هستند؛ این
    تابع فقط آریتمتیک نهایی را انجام می‌دهد. ``entropy_term`` همان ``−TΔS`` است.
    """
    return float(complex_energy - receptor_energy - ligand_energy + entropy_term)


def block_average_error(series: Sequence[float], n_blocks: int = 5) -> float:
    """خطای استاندارد میانگین با روش block averaging (برای گزارش عدم‌قطعیت MM-GBSA)."""
    values = np.asarray(series, dtype=float)
    if len(values) < n_blocks * 2:
        n_blocks = max(len(values) // 2, 1)
    blocks = np.array_split(values, n_blocks)
    means = np.array([b.mean() for b in blocks])
    return float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.0
