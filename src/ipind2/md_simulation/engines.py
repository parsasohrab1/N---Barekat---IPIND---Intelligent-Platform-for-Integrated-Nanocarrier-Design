"""
Simulation engines (FR-05): GROMACS, OpenMM and the RDKit conformer set.

Each engine returns a ``Trajectory`` whose ``fidelity`` field honestly shows the real simulation
level:

* ``"md"`` — real molecular dynamics simulation (GROMACS / OpenMM);
* ``"conformer_ensemble"`` — RDKit conformer sampling (ETKDG + MMFF), **not MD**;
  for fast screening and testing the pipeline in environments where no MD engine is installed.

MD engines raise ``MDEngineUnavailable`` if the software is missing and never silently fall back to a
lower-fidelity mode — choosing an alternative is the caller's job (see
``validation.validate_candidates``).
"""

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")  # UFFTYPER warns every time for metals; we report the missing parameters ourselves


class MDEngineUnavailable(RuntimeError):
    """The requested engine (or its dependencies) is not installed on this machine."""


@dataclass
class Trajectory:
    """Simulation trajectory: coordinates (T, N, 3) in Å, elements and metadata."""

    coords: np.ndarray
    elements: List[str]
    fidelity: str  # 'md' | 'conformer_ensemble'
    engine: str
    smiles: str = ""
    energies_kcal: Optional[np.ndarray] = None
    bonds: List[tuple] = field(default_factory=list)  # (i, j) atom indices
    simulated_ns: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.coords.shape[0])


class MDEngine(Protocol):
    name: str

    def simulate(self, smiles: str, duration_ns: float = 100.0, **kwargs) -> Trajectory: ...


class ConformerEnsembleEngine:
    """
    Fast conformer sampling (ETKDG + MMFF/UFF optimization). **Not MD.**

    ``duration_ns`` is ignored (and ``simulated_ns=0`` is reported); the number of frames is set by
    ``n_conformers``.
    """

    name = "rdkit-conformer-ensemble"

    def __init__(self, n_conformers: int = 16, seed: int = 7, n_threads: int = 0):
        self.n_conformers = n_conformers
        self.seed = seed
        self.n_threads = n_threads

    def simulate(self, smiles: str, duration_ns: float = 0.0, **kwargs) -> Trajectory:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = self.seed
        params.useRandomCoords = True  # more stable for long/hydrophobic chains
        params.maxIterations = 200
        params.numThreads = self.n_threads  # 0 = all cores
        conformer_ids = list(AllChem.EmbedMultipleConfs(mol, self.n_conformers, params))
        if not conformer_ids:
            raise RuntimeError(f"Conformer generation failed for {smiles!r}")

        energies: List[float] = [float("nan")] * len(conformer_ids)
        method = "none (no force-field parameters)"
        # Energy is reported only when all atoms have parameters; otherwise a meaningless number
        # (e.g., Au in UFF) would enter the output instead of NaN.
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500, numThreads=self.n_threads)
                method = "MMFF94"
                energies = [float(e) for _, e in results]
            elif AllChem.UFFHasAllMoleculeParams(mol):
                results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=500, numThreads=self.n_threads)
                method = "UFF"
                energies = [float(e) for _, e in results]
        except Exception:  # pragma: no cover - unexpected optimizer error
            energies = [float("nan")] * len(conformer_ids)

        coords = np.stack([mol.GetConformer(cid).GetPositions() for cid in conformer_ids])
        elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        return Trajectory(
            coords=coords,
            elements=elements,
            fidelity="conformer_ensemble",
            engine=self.name,
            smiles=smiles,
            energies_kcal=np.asarray(energies, dtype=float),
            bonds=bonds,
            simulated_ns=0.0,
            metadata={"forcefield": method, "n_conformers": len(conformer_ids)},
        )


class GromacsEngine:
    """
    GROMACS adapter. The system inputs (.gro structure and .top topology with a CHARMM36/OPLS-AA
    force field) must already be built — nanocarrier molecules need dedicated
    parameterization (CGenFF/LigParGen) which is outside the scope of this adapter.

    ``prepare`` writes the ``.mdp`` files (no GROMACS needed); ``simulate`` requires
    ``gmx`` on PATH.
    """

    name = "gromacs"

    def __init__(self, work_dir: str, forcefield: str = "charmm36", gmx_executable: str = "gmx"):
        if forcefield not in ("charmm36", "oplsaa"):
            raise ValueError("forcefield must be charmm36 or oplsaa")
        self.work_dir = Path(work_dir)
        self.forcefield = forcefield
        self.gmx = gmx_executable

    def mdp_text(self, duration_ns: float, temperature_k: float = 310.0, dt_ps: float = 0.002) -> str:
        n_steps = int(round(duration_ns * 1000.0 / dt_ps))
        return "\n".join(
            [
                "; generated by ipind2.md_simulation.GromacsEngine",
                "integrator = md",
                f"dt = {dt_ps}",
                f"nsteps = {n_steps}",
                "nstxout-compressed = 5000",
                "nstenergy = 5000",
                "cutoff-scheme = Verlet",
                "coulombtype = PME",
                "rcoulomb = 1.2",
                "rvdw = 1.2",
                "tcoupl = V-rescale",
                "tc-grps = System",
                "tau-t = 0.1",
                f"ref-t = {temperature_k}",
                "pcoupl = C-rescale",
                "pcoupltype = isotropic",
                "tau-p = 2.0",
                "ref-p = 1.0",
                "compressibility = 4.5e-5",
                "constraints = h-bonds",
                "constraint-algorithm = lincs",
                "",
            ]
        )

    def prepare(self, duration_ns: float = 100.0, temperature_k: float = 310.0) -> Path:
        """Write ``production.mdp``; returns the file path."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        mdp = self.work_dir / "production.mdp"
        mdp.write_text(self.mdp_text(duration_ns, temperature_k), encoding="utf-8")
        return mdp

    def available(self) -> bool:
        return shutil.which(self.gmx) is not None

    def simulate(self, smiles: str, duration_ns: float = 100.0, **kwargs) -> Trajectory:
        if not self.available():
            raise MDEngineUnavailable(
                f"Executable '{self.gmx}' not found on PATH; install GROMACS or "
                "use OpenMMEngine/ConformerEnsembleEngine."
            )
        system = self.work_dir / "system.gro"
        topology = self.work_dir / "topol.top"
        if not system.exists() or not topology.exists():
            raise MDEngineUnavailable(
                f"System files are not ready ({system.name}, {topology.name}) — force-field parameterization "
                "for this structure must be done before running."
            )
        try:
            import MDAnalysis as mda
        except ImportError as exc:  # we check before the expensive run, not after it
            raise MDEngineUnavailable(
                "MDAnalysis is required to read the xtc trajectory (pip install MDAnalysis)"
            ) from exc

        mdp = self.prepare(duration_ns, kwargs.get("temperature_k", 310.0))

        def run(*args):
            subprocess.run([self.gmx, *args], cwd=self.work_dir, check=True, capture_output=True, text=True)

        run("grompp", "-f", mdp.name, "-c", system.name, "-p", topology.name, "-o", "production.tpr")
        run("mdrun", "-deffnm", "production")

        universe = mda.Universe(str(system), str(self.work_dir / "production.xtc"))
        frames = np.stack([universe.atoms.positions.copy() for _ in universe.trajectory])
        elements = [
            (getattr(a, "element", "") or a.name[:1]).strip().capitalize() for a in universe.atoms
        ]
        return Trajectory(
            coords=frames,
            elements=elements,
            fidelity="md",
            engine=self.name,
            smiles=smiles,
            bonds=[tuple(int(i) for i in b) for b in universe.bonds.indices] if hasattr(universe, "bonds") else [],
            simulated_ns=duration_ns,
            metadata={"forcefield": self.forcefield},
        )


class OpenMMEngine:
    """
    OpenMM adapter (GBn2 implicit solvent, Langevin, 310 K).

    ⚠️ Not run in this repo's CI (OpenMM/openmmforcefields are not installed); the code path is
    only activated by a real import on a machine that has OpenMM.
    """

    name = "openmm"

    def __init__(self, temperature_k: float = 310.0, report_every_ps: float = 10.0):
        self.temperature_k = temperature_k
        self.report_every_ps = report_every_ps

    def simulate(self, smiles: str, duration_ns: float = 100.0, **kwargs) -> Trajectory:
        try:
            import openmm
            from openmm import app, unit
            from openmmforcefields.generators import SystemGenerator
            from openff.toolkit.topology import Molecule
        except ImportError as exc:
            raise MDEngineUnavailable(
                "OpenMM/openmmforcefields/openff-toolkit are not installed "
                "(pip install openmm openmmforcefields openff-toolkit)"
            ) from exc

        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        molecule.generate_conformers(n_conformers=1)
        topology = molecule.to_topology().to_openmm()
        positions = molecule.conformers[0].to_openmm()

        generator = SystemGenerator(
            forcefields=["amber/ff14SB.xml", "implicit/gbn2.xml"],
            small_molecule_forcefield="gaff-2.11",
            molecules=[molecule],
            forcefield_kwargs={"constraints": app.HBonds},
        )
        system = generator.create_system(topology)
        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature_k * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        )
        simulation = app.Simulation(topology, system, integrator)
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy()

        steps_total = int(duration_ns * 1000.0 / 0.002)
        steps_per_report = int(self.report_every_ps / 0.002)
        frames: List[np.ndarray] = []
        energies: List[float] = []
        for _ in range(max(steps_total // steps_per_report, 1)):
            simulation.step(steps_per_report)
            state = simulation.context.getState(getPositions=True, getEnergy=True)
            frames.append(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
            energies.append(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

        elements = [atom.element.symbol for atom in topology.atoms()]
        bonds = [(b.atom1.index, b.atom2.index) for b in topology.bonds()]
        return Trajectory(
            coords=np.stack(frames),
            elements=elements,
            fidelity="md",
            engine=self.name,
            smiles=smiles,
            energies_kcal=np.asarray(energies),
            bonds=bonds,
            simulated_ns=duration_ns,
            metadata={"forcefield": "gaff-2.11 + GBn2", "temperature_k": self.temperature_k},
        )


def available_engines(work_dir: str = ".") -> Sequence[str]:
    """Names of engines usable on this machine (the conformer set is always available)."""
    names = [ConformerEnsembleEngine.name]
    if GromacsEngine(work_dir).available():
        names.append(GromacsEngine.name)
    try:
        import openmm  # noqa: F401

        names.append(OpenMMEngine.name)
    except ImportError:
        pass
    return names
