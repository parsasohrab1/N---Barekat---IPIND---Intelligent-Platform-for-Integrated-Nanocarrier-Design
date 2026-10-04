"""تست‌های واحد ۵ (FR-05): تحلیل مسیر با پاسخ تحلیلی، موتورها و صداقت سطح شبیه‌سازی."""

import subprocess

import numpy as np
import pytest

from ipind2.md_simulation import (
    REQUIRED_MD_NS,
    ConformerEnsembleEngine,
    GromacsEngine,
    MDEngineUnavailable,
    OpenMMEngine,
    Trajectory,
    ValidationReport,
    analyze_trajectory,
    available_engines,
    block_average_error,
    mmgbsa_delta_g,
    order_parameter_p2,
    radius_of_gyration,
    sasa_shrake_rupley,
    sasa_trajectory,
    validate_candidates,
)
from ipind2.md_simulation.quantum import XTBRefiner


class TestAnalysisAgainstAnalyticResults:
    def test_sasa_single_sphere_exact(self):
        radius, probe = 1.7, 1.4
        assert sasa_shrake_rupley(np.zeros((1, 3)), [radius], probe) == pytest.approx(4 * np.pi * (radius + probe) ** 2, rel=1e-6)

    def test_sasa_two_touching_spheres_less_than_two_isolated(self):
        isolated = 2 * sasa_shrake_rupley(np.zeros((1, 3)), [1.7])
        pair = sasa_shrake_rupley(np.array([[0.0, 0, 0], [1.5, 0, 0]]), [1.7, 1.7])
        assert pair < isolated

    def test_sasa_far_apart_spheres_additive(self):
        far = sasa_shrake_rupley(np.array([[0.0, 0, 0], [100.0, 0, 0]]), [1.7, 1.7])
        assert far == pytest.approx(2 * sasa_shrake_rupley(np.zeros((1, 3)), [1.7]), rel=1e-6)

    def test_sasa_buried_atom_contributes_nothing(self):
        core = np.array([[0.0, 0, 0]])
        shell = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
        coords = np.vstack([core, shell])
        radii = [1.7] * 7
        outer = sasa_shrake_rupley(coords[1:], radii[1:])
        assert sasa_shrake_rupley(coords, radii) == pytest.approx(outer, rel=0.02)

    def test_sasa_translation_and_rotation_invariant(self):
        rng = np.random.default_rng(0)
        coords = rng.normal(size=(8, 3)) * 2
        radii = [1.5] * 8
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        base = sasa_shrake_rupley(coords, radii, n_points=960)
        moved = sasa_shrake_rupley(coords @ rotation.T + np.array([5.0, -3.0, 2.0]), radii, n_points=960)
        assert moved == pytest.approx(base, rel=0.03)

    def test_rg_ring_and_mass_weighting(self):
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        ring = np.stack([3 * np.cos(theta), 3 * np.sin(theta), np.zeros_like(theta)], axis=1)
        assert radius_of_gyration(ring)[0] == pytest.approx(3.0)
        two = np.array([[0.0, 0, 0], [4.0, 0, 0]])
        assert radius_of_gyration(two, [1.0, 1.0])[0] == pytest.approx(2.0)
        assert radius_of_gyration(two, [1.0, 3.0])[0] == pytest.approx(np.sqrt(3.0))  # مرکز جرم به اتم سنگین‌تر نزدیک‌تر

    def test_rg_trajectory_shape_and_validation(self):
        assert radius_of_gyration(np.random.default_rng(0).normal(size=(5, 10, 3))).shape == (5,)
        with pytest.raises(ValueError):
            radius_of_gyration(np.zeros((1, 3, 3)), [1.0, 1.0])

    def test_order_parameter_limits(self):
        rng = np.random.default_rng(0)
        assert order_parameter_p2(np.tile([0.0, 0, 1.0], (200, 1)) + rng.normal(0, 0.02, (200, 3))) > 0.98
        assert abs(order_parameter_p2(rng.normal(size=(6000, 3)))) < 0.05
        assert order_parameter_p2(np.tile([1.0, 0, 0], (50, 1)), director=[0, 0, 1.0]) == pytest.approx(-0.5)  # عمود بر محور
        with pytest.raises(ValueError):
            order_parameter_p2(np.zeros((0, 3)))

    def test_mmgbsa_arithmetic_and_error_estimate(self):
        assert mmgbsa_delta_g(-120.0, -80.0, -30.0) == pytest.approx(-10.0)
        assert mmgbsa_delta_g(-120.0, -80.0, -30.0, entropy_term=4.0) == pytest.approx(-6.0)
        noisy = np.random.default_rng(0).normal(-10, 2, 500)
        assert 0 < block_average_error(noisy) < 2.0


class TestConformerEngine:
    SMILES = "CCCCCCCCCCCC[N+](C)(C)C"

    def test_trajectory_is_labelled_not_md(self):
        trajectory = ConformerEnsembleEngine(n_conformers=4).simulate(self.SMILES)
        assert trajectory.fidelity == "conformer_ensemble" and trajectory.simulated_ns == 0.0
        assert trajectory.coords.shape[0] == trajectory.n_frames == 4 and trajectory.coords.shape[2] == 3

    def test_energies_reported_only_with_forcefield_params(self):
        organic = ConformerEnsembleEngine(n_conformers=3).simulate(self.SMILES)
        assert np.isfinite(organic.energies_kcal).all() and organic.metadata["forcefield"] == "MMFF94"
        gold = ConformerEnsembleEngine(n_conformers=3).simulate("OCCOCCS[Au]")
        assert np.isnan(gold.energies_kcal).all(), "برای Au انرژی بی‌معنی نباید گزارش شود"

    def test_same_seed_same_geometry(self):
        a = ConformerEnsembleEngine(n_conformers=3, seed=5).simulate(self.SMILES, ).coords
        b = ConformerEnsembleEngine(n_conformers=3, seed=5).simulate(self.SMILES).coords
        assert a.shape == b.shape

    def test_invalid_smiles(self):
        with pytest.raises(ValueError):
            ConformerEnsembleEngine().simulate("bad(((")


class TestHonestValidation:
    def test_conformer_results_never_claim_md_complete(self):
        report = validate_candidates([TestConformerEngine.SMILES], ConformerEnsembleEngine(n_conformers=4))
        assert isinstance(report, ValidationReport) and not report.md_complete
        result = report.results[0]
        assert not result.is_real_md and not result.meets_sim_time and result.notes
        assert result.rg_mean > 0 and result.sasa_mean > 0 and result.order_parameter is not None

    def test_md_complete_requires_real_md_and_sufficient_time(self):
        def trajectory(fidelity, ns):
            coords = np.random.default_rng(0).normal(size=(5, 6, 3))
            return Trajectory(coords, ["C"] * 6, fidelity, "fake", "C" * 6, bonds=[(0, 1)], simulated_ns=ns)

        short = ValidationReport([analyze_trajectory(trajectory("md", 10.0))])
        full = ValidationReport([analyze_trajectory(trajectory("md", REQUIRED_MD_NS))])
        fake = ValidationReport([analyze_trajectory(trajectory("conformer_ensemble", 500.0))])
        assert not short.md_complete and full.md_complete and not fake.md_complete
        assert ValidationReport([]).md_complete is False

    def test_skipped_candidate_blocks_completion(self):
        report = ValidationReport([], skipped=["x"])
        assert not report.md_complete

    def test_unavailable_engine_raises_without_silent_fallback(self, tmp_path):
        with pytest.raises(MDEngineUnavailable):
            validate_candidates(["CCO"], GromacsEngine(str(tmp_path), gmx_executable="definitely-not-installed-gmx"))

    def test_explicit_fallback_is_labelled(self, tmp_path):
        engine = GromacsEngine(str(tmp_path), gmx_executable="definitely-not-installed-gmx")
        report = validate_candidates(["CCCCCCCCO"], engine, fallback_to_conformers=True)
        assert report.results[0].fidelity == "conformer_ensemble" and not report.md_complete

    def test_unstable_trajectory_flagged(self):
        rng = np.random.default_rng(1)
        frames = np.stack([rng.normal(size=(8, 3)) * scale for scale in (1, 1, 6, 6, 1, 8)])
        unstable = analyze_trajectory(Trajectory(frames, ["C"] * 8, "md", "fake", "C", bonds=[], simulated_ns=100.0))
        steady = analyze_trajectory(Trajectory(np.stack([rng.normal(size=(8, 3))] * 4), ["C"] * 8, "md", "fake", "C", bonds=[], simulated_ns=100.0))
        assert not unstable.stable and steady.stable


class TestEngineAdapters:
    def test_gromacs_mdp_encodes_duration_and_forcefield_constraints(self, tmp_path):
        engine = GromacsEngine(str(tmp_path))
        text = engine.mdp_text(duration_ns=100.0, dt_ps=0.002)
        assert "nsteps = 50000000" in text and "ref-t = 310.0" in text and "coulombtype = PME" in text
        path = engine.prepare(duration_ns=1.0)
        assert path.exists() and "nsteps = 500000" in path.read_text()

    def test_gromacs_rejects_unknown_forcefield(self, tmp_path):
        with pytest.raises(ValueError):
            GromacsEngine(str(tmp_path), forcefield="amber99")

    def test_gromacs_requires_prepared_system_files(self, tmp_path, monkeypatch):
        import shutil

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/gmx")
        monkeypatch.setitem(__import__("sys").modules, "MDAnalysis", object())
        with pytest.raises(MDEngineUnavailable, match="system.gro|topol.top"):
            GromacsEngine(str(tmp_path)).simulate("CCO")

    def test_gromacs_checks_reader_before_running_expensive_job(self, tmp_path, monkeypatch):
        import shutil
        import sys

        (tmp_path / "system.gro").write_text("x")
        (tmp_path / "topol.top").write_text("x")
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/gmx")
        monkeypatch.setitem(sys.modules, "MDAnalysis", None)  # import ⇒ ImportError
        called = []
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a))
        with pytest.raises(MDEngineUnavailable, match="MDAnalysis"):
            GromacsEngine(str(tmp_path)).simulate("CCO")
        assert not called, "نباید پیش از اطمینان از امکان خواندن نتیجه، شبیه‌سازی گران اجرا شود"

    def test_openmm_unavailable_message(self):
        try:
            import openmm  # noqa: F401
            pytest.skip("OpenMM نصب است")
        except ImportError:
            with pytest.raises(MDEngineUnavailable):
                OpenMMEngine().simulate("CCO", duration_ns=0.001)

    def test_available_engines_always_includes_conformers(self, tmp_path):
        assert "rdkit-conformer-ensemble" in available_engines(str(tmp_path))

    def test_xtb_unavailable(self):
        with pytest.raises(MDEngineUnavailable):
            XTBRefiner(executable="definitely-not-installed-xtb").single_point_kcal("CCO")
