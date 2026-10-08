"""Unit 5 tests (FR-05): trajectory analysis with an analytical answer, engines and honesty about simulation level."""

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
        assert radius_of_gyration(two, [1.0, 3.0])[0] == pytest.approx(np.sqrt(3.0))  # center of mass is closer to the heavier atom

    def test_rg_trajectory_shape_and_validation(self):
        assert radius_of_gyration(np.random.default_rng(0).normal(size=(5, 10, 3))).shape == (5,)
        with pytest.raises(ValueError):
            radius_of_gyration(np.zeros((1, 3, 3)), [1.0, 1.0])

    def test_order_parameter_limits(self):
        rng = np.random.default_rng(0)
        assert order_parameter_p2(np.tile([0.0, 0, 1.0], (200, 1)) + rng.normal(0, 0.02, (200, 3))) > 0.98
        assert abs(order_parameter_p2(rng.normal(size=(6000, 3)))) < 0.05
        assert order_parameter_p2(np.tile([1.0, 0, 0], (50, 1)), director=[0, 0, 1.0]) == pytest.approx(-0.5)  # perpendicular to the axis
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
        assert np.isnan(gold.energies_kcal).all(), "A meaningless energy must not be reported for Au"

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
        assert not called, "The expensive simulation must not run before it is certain the result can be read"

    def test_openmm_unavailable_message(self):
        try:
            import openmm  # noqa: F401
            pytest.skip("OpenMM is installed")
        except ImportError:
            with pytest.raises(MDEngineUnavailable):
                OpenMMEngine().simulate("CCO", duration_ns=0.001)

    def test_available_engines_always_includes_conformers(self, tmp_path):
        assert "rdkit-conformer-ensemble" in available_engines(str(tmp_path))

    def test_xtb_unavailable(self):
        with pytest.raises(MDEngineUnavailable):
            XTBRefiner(executable="definitely-not-installed-xtb").single_point_kcal("CCO")


class TestCondaOpenMMEngine:
    """Real-MD engine via a separate conda env. Unit tests mock the runner; one integration test runs it for real."""

    MD_PYTHON = "C:/Users/asus/miniforge3/envs/ipind-md/python.exe"

    def test_unavailable_without_env(self, monkeypatch):
        from ipind2.md_simulation import CondaOpenMMEngine

        monkeypatch.delenv("IPIND_MD_PYTHON", raising=False)
        engine = CondaOpenMMEngine()
        assert not engine.available()
        with pytest.raises(MDEngineUnavailable, match="IPIND_MD_PYTHON"):
            engine.simulate("CCO", duration_ns=0.01)

    def _fake_run(self, tmp_path, returncode=0, stdout="", ns=0.5, finite=True):
        import json as _json
        from types import SimpleNamespace

        def run(command, **kwargs):
            if returncode == 0:
                out = command[command.index("--out") + 1]
                coords = np.random.default_rng(0).normal(size=(4, 3, 3)).astype(np.float32)
                meta = {"simulated_ns": ns, "finite": finite, "ns_per_day": 100.0}
                np.savez_compressed(out, coords=coords, energies=np.array([1.0, 2.0, 3.0, 4.0]),
                                    elements=np.array(["C", "C", "O"]), bonds=np.array([[0, 1], [1, 2]], dtype=np.int32),
                                    meta=_json.dumps(meta))
            return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="boom")

        return run

    def test_parses_trajectory_and_reports_actual_simulated_time(self, tmp_path, monkeypatch):
        from ipind2.md_simulation import CondaOpenMMEngine

        python = tmp_path / "python.exe"
        python.write_text("x")
        monkeypatch.setattr(subprocess, "run", self._fake_run(tmp_path, ns=0.5))
        trajectory = CondaOpenMMEngine(python=str(python)).simulate("CCO", duration_ns=100.0)
        assert trajectory.fidelity == "md" and trajectory.simulated_ns == 0.5  # actual, not requested
        assert trajectory.coords.shape == (4, 3, 3) and trajectory.bonds == [(0, 1), (1, 2)]
        result = analyze_trajectory(trajectory)
        assert result.is_real_md and not result.meets_sim_time, "0.5 ns of real MD must not satisfy the 100 ns requirement"

    def test_full_length_run_satisfies_requirement(self, tmp_path, monkeypatch):
        from ipind2.md_simulation import CondaOpenMMEngine

        python = tmp_path / "python.exe"
        python.write_text("x")
        monkeypatch.setattr(subprocess, "run", self._fake_run(tmp_path, ns=100.0))
        trajectory = CondaOpenMMEngine(python=str(python)).simulate("CCO")
        assert analyze_trajectory(trajectory).meets_sim_time

    def test_unsupported_chemistry_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        from ipind2.md_simulation import CondaOpenMMEngine

        python = tmp_path / "python.exe"
        python.write_text("x")
        monkeypatch.setattr(subprocess, "run", self._fake_run(tmp_path, returncode=3, stdout='{"error": "elements without Sage parameters: [\'Au\']"}'))
        report = validate_candidates(["OCCS[Au]"], CondaOpenMMEngine(python=str(python)))
        assert report.skipped == ["OCCS[Au]"] and not report.md_complete

    def test_runner_crash_surfaces_as_error(self, tmp_path, monkeypatch):
        from ipind2.md_simulation import CondaOpenMMEngine

        python = tmp_path / "python.exe"
        python.write_text("x")
        monkeypatch.setattr(subprocess, "run", self._fake_run(tmp_path, returncode=1))
        with pytest.raises(RuntimeError, match="MD runner failed"):
            CondaOpenMMEngine(python=str(python)).simulate("CCO", duration_ns=0.01)

    def test_blown_up_simulation_rejected(self, tmp_path, monkeypatch):
        from ipind2.md_simulation import CondaOpenMMEngine

        python = tmp_path / "python.exe"
        python.write_text("x")
        monkeypatch.setattr(subprocess, "run", self._fake_run(tmp_path, finite=False))
        with pytest.raises(RuntimeError, match="non-finite"):
            CondaOpenMMEngine(python=str(python)).simulate("CCO", duration_ns=0.01)

    @pytest.mark.skipif(not __import__("os").path.exists(MD_PYTHON), reason="MD conda env (ipind-md) not installed")
    def test_integration_real_openmm_run(self):
        """Actually runs OpenMM (≈0.02 ns): finite energies, plausible geometry, honest time accounting."""
        from ipind2.md_simulation import CondaOpenMMEngine

        engine = CondaOpenMMEngine(python=self.MD_PYTHON, platform="CPU", report_ps=5.0, equil_ps=5.0)
        trajectory = engine.simulate("CCCCCCCCCCCC[N+](C)(C)C", duration_ns=0.02)
        assert trajectory.fidelity == "md" and trajectory.simulated_ns == pytest.approx(0.02, abs=0.006)
        assert np.isfinite(trajectory.coords).all() and np.isfinite(trajectory.energies_kcal).all()
        # bonded geometry stays physical: C-C bonds ~1.5 Å in every frame
        carbon_bonds = [(i, j) for i, j in trajectory.bonds if trajectory.elements[i] == "C" and trajectory.elements[j] == "C"]
        lengths = np.linalg.norm(trajectory.coords[:, [j for _, j in carbon_bonds]] - trajectory.coords[:, [i for i, _ in carbon_bonds]], axis=2)
        assert 1.3 < lengths.mean() < 1.7 and lengths.max() < 2.0
        report = analyze_trajectory(trajectory)
        assert report.is_real_md and not report.meets_sim_time and report.rg_mean > 0

    def test_unsupported_element_real_runner(self):
        import os

        if not os.path.exists(self.MD_PYTHON):
            pytest.skip("MD conda env (ipind-md) not installed")
        from ipind2.md_simulation import CondaOpenMMEngine

        with pytest.raises(ValueError, match="Sage"):
            CondaOpenMMEngine(python=self.MD_PYTHON, platform="CPU").simulate("OCCS[Au]", duration_ns=0.01)
