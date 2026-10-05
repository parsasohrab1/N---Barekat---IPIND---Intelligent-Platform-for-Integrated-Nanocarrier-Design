"""
Optional quasi-quantum-level refinement (NFR-09) with xtb (GFN2-xTB).

NFR-09 in the SRS is "optional/complementary to MM-GBSA" and targets "error < 1 kcal/mol". This module is only
an xtb execution adapter for machines that have it installed; **an accuracy of 1 kcal/mol is not claimed or
validated** — that requires comparison with reference data (e.g., DFT or experimental).
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from .engines import MDEngineUnavailable

HARTREE_TO_KCAL = 627.5094740631
_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")


class XTBRefiner:
    """GFN2-xTB single-point energy (Eh → kcal/mol) for a structure."""

    def __init__(self, executable: str = "xtb", charge_aware: bool = True, timeout_s: int = 300):
        self.executable = executable
        self.charge_aware = charge_aware
        self.timeout_s = timeout_s

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def single_point_kcal(self, smiles: str, seed: int = 7) -> float:
        if not self.available():
            raise MDEngineUnavailable(f"Executable '{self.executable}' (xtb) is not on PATH")
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
            raise RuntimeError(f"3D structure generation failed for {smiles!r}")
        charge = Chem.GetFormalCharge(mol) if self.charge_aware else 0
        with tempfile.TemporaryDirectory() as directory:
            xyz = Path(directory) / "mol.xyz"
            Chem.MolToXYZFile(mol, str(xyz))
            result = subprocess.run(
                [self.executable, xyz.name, "--gfn", "2", "--chrg", str(charge)],
                cwd=directory, capture_output=True, text=True, timeout=self.timeout_s, check=True,
            )
        match = _ENERGY_RE.search(result.stdout)
        if match is None:
            raise RuntimeError("The xtb output did not contain a total energy")
        return float(match.group(1)) * HARTREE_TO_KCAL
