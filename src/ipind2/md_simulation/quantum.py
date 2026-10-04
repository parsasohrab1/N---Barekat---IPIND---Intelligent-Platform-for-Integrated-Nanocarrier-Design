"""
پالایش سطح شبه‌کوانتومی اختیاری (NFR-09) با xtb (GFN2-xTB).

NFR-09 در SRS «اختیاری/مکمل MM-GBSA» است و هدف «خطای < ۱ kcal/mol» دارد. این ماژول فقط
آداپتور اجرای xtb روی ماشین‌هایی است که آن را نصب دارند؛ **دقت ۱ kcal/mol ادعا یا
اعتبارسنجی نمی‌شود** — آن به مقایسه با داده مرجع (مثلاً DFT یا تجربی) نیاز دارد.
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
    """انرژی تک‌نقطه‌ای GFN2-xTB (Eh → kcal/mol) برای یک ساختار."""

    def __init__(self, executable: str = "xtb", charge_aware: bool = True, timeout_s: int = 300):
        self.executable = executable
        self.charge_aware = charge_aware
        self.timeout_s = timeout_s

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def single_point_kcal(self, smiles: str, seed: int = 7) -> float:
        if not self.available():
            raise MDEngineUnavailable(f"اجرایی «{self.executable}» (xtb) روی PATH نیست")
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
            raise RuntimeError(f"تولید ساختار سه‌بعدی برای {smiles!r} ممکن نشد")
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
            raise RuntimeError("خروجی xtb شامل انرژی کل نبود")
        return float(match.group(1)) * HARTREE_TO_KCAL
