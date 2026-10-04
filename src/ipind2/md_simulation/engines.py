"""
موتورهای شبیه‌سازی (FR-05): GROMACS، OpenMM و مجموعه کانفورمر RDKit.

هر موتور یک ``Trajectory`` برمی‌گرداند که فیلد ``fidelity`` صادقانه سطح واقعی شبیه‌سازی را
نشان می‌دهد:

* ``"md"`` — شبیه‌سازی دینامیک مولکولی واقعی (GROMACS / OpenMM)؛
* ``"conformer_ensemble"`` — نمونه‌برداری کانفورمری RDKit (ETKDG + MMFF)، **MD نیست**؛
  برای غربالگری سریع و تست خط لوله در محیط‌هایی که موتور MD نصب نیست.

موتورهای MD در صورت نبودِ نرم‌افزار ``MDEngineUnavailable`` می‌اندازند و هرگز بی‌صدا به
حالت کم‌دقت‌تر برنمی‌گردند — انتخاب جایگزین کار فراخوان است (نگاه کنید به
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

RDLogger.DisableLog("rdApp.*")  # UFFTYPER برای فلزات هر بار هشدار می‌دهد؛ نبود پارامتر را خودمان گزارش می‌کنیم


class MDEngineUnavailable(RuntimeError):
    """موتور درخواستی (یا وابستگی‌های آن) روی این ماشین نصب نیست."""


@dataclass
class Trajectory:
    """مسیر شبیه‌سازی: مختصات (T, N, 3) بر حسب Å، عناصر و فراداده."""

    coords: np.ndarray
    elements: List[str]
    fidelity: str  # 'md' | 'conformer_ensemble'
    engine: str
    smiles: str = ""
    energies_kcal: Optional[np.ndarray] = None
    bonds: List[tuple] = field(default_factory=list)  # (i, j) اندیس اتم‌ها
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
    نمونه‌برداری کانفورمری سریع (ETKDG + بهینه‌سازی MMFF/UFF). **MD نیست.**

    ``duration_ns`` نادیده گرفته می‌شود (و ``simulated_ns=0`` گزارش می‌شود)؛ تعداد فریم‌ها با
    ``n_conformers`` تعیین می‌شود.
    """

    name = "rdkit-conformer-ensemble"

    def __init__(self, n_conformers: int = 16, seed: int = 7, n_threads: int = 0):
        self.n_conformers = n_conformers
        self.seed = seed
        self.n_threads = n_threads

    def simulate(self, smiles: str, duration_ns: float = 0.0, **kwargs) -> Trajectory:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"SMILES نامعتبر: {smiles!r}")
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = self.seed
        params.useRandomCoords = True  # برای زنجیره‌های بلند/آبگریز پایدارتر است
        params.maxIterations = 200
        params.numThreads = self.n_threads  # ۰ = همه هسته‌ها
        conformer_ids = list(AllChem.EmbedMultipleConfs(mol, self.n_conformers, params))
        if not conformer_ids:
            raise RuntimeError(f"تولید کانفورمر برای {smiles!r} ممکن نشد")

        energies: List[float] = [float("nan")] * len(conformer_ids)
        method = "none (no force-field parameters)"
        # فقط وقتی همه اتم‌ها پارامتر دارند انرژی گزارش می‌شود؛ وگرنه عدد بی‌معنی
        # (مثلاً Au در UFF) به‌جای NaN وارد خروجی می‌شد.
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500, numThreads=self.n_threads)
                method = "MMFF94"
                energies = [float(e) for _, e in results]
            elif AllChem.UFFHasAllMoleculeParams(mol):
                results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=500, numThreads=self.n_threads)
                method = "UFF"
                energies = [float(e) for _, e in results]
        except Exception:  # pragma: no cover - خطای غیرمنتظره بهینه‌ساز
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
    آداپتور GROMACS. ورودی‌های سیستم (ساختار .gro و توپولوژی .top با میدان نیروی
    CHARMM36/OPLS-AA) باید از قبل ساخته شده باشند — مولکول‌های نانوحامل به پارامترسازی
    اختصاصی (CGenFF/LigParGen) نیاز دارند که خارج از دامنه این آداپتور است.

    ``prepare`` فایل‌های ``.mdp`` را می‌نویسد (بدون نیاز به GROMACS)؛ ``simulate`` به
    ``gmx`` روی PATH نیاز دارد.
    """

    name = "gromacs"

    def __init__(self, work_dir: str, forcefield: str = "charmm36", gmx_executable: str = "gmx"):
        if forcefield not in ("charmm36", "oplsaa"):
            raise ValueError("forcefield باید charmm36 یا oplsaa باشد")
        self.work_dir = Path(work_dir)
        self.forcefield = forcefield
        self.gmx = gmx_executable

    def mdp_text(self, duration_ns: float, temperature_k: float = 310.0, dt_ps: float = 0.002) -> str:
        n_steps = int(round(duration_ns * 1000.0 / dt_ps))
        return "\n".join(
            [
                "; تولیدشده توسط ipind2.md_simulation.GromacsEngine",
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
        """نوشتن ``production.mdp``؛ مسیر فایل را برمی‌گرداند."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        mdp = self.work_dir / "production.mdp"
        mdp.write_text(self.mdp_text(duration_ns, temperature_k), encoding="utf-8")
        return mdp

    def available(self) -> bool:
        return shutil.which(self.gmx) is not None

    def simulate(self, smiles: str, duration_ns: float = 100.0, **kwargs) -> Trajectory:
        if not self.available():
            raise MDEngineUnavailable(
                f"اجرایی «{self.gmx}» روی PATH یافت نشد؛ GROMACS را نصب کنید یا "
                "از OpenMMEngine/ConformerEnsembleEngine استفاده کنید."
            )
        system = self.work_dir / "system.gro"
        topology = self.work_dir / "topol.top"
        if not system.exists() or not topology.exists():
            raise MDEngineUnavailable(
                f"فایل‌های سیستم آماده نیست ({system.name}, {topology.name}) — پارامترسازی میدان "
                "نیرو برای این ساختار باید پیش از اجرا انجام شود."
            )
        try:
            import MDAnalysis as mda
        except ImportError as exc:  # پیش از اجرای گران بررسی می‌کنیم، نه بعد از آن
            raise MDEngineUnavailable(
                "MDAnalysis برای خواندن مسیر xtc لازم است (pip install MDAnalysis)"
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
    آداپتور OpenMM (حلال ضمنی GBn2، Langevin، ۳۱۰ K).

    ⚠️ در CI این مخزن اجرا نمی‌شود (OpenMM/openmmforcefields نصب نیست)؛ مسیر کد فقط با
    import واقعی روی ماشین دارای OpenMM فعال می‌شود.
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
                "OpenMM/openmmforcefields/openff-toolkit نصب نیست "
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
    """نام موتورهای قابل‌استفاده روی این ماشین (مجموعه کانفورمر همیشه هست)."""
    names = [ConformerEnsembleEngine.name]
    if GromacsEngine(work_dir).available():
        names.append(GromacsEngine.name)
    try:
        import openmm  # noqa: F401

        names.append(OpenMMEngine.name)
    except ImportError:
        pass
    return names
