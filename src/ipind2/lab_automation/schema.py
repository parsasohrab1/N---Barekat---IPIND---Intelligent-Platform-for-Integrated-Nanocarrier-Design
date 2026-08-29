"""ساختار داده نتیجه آزمایشگاهی، منطبق بر جدول experimental_results در sql/schema.sql."""

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional


@dataclass
class ExperimentalResult:
    """یک رکورد بازخورد آزمایشگاهی، معادل یک سطر از جدول ``experimental_results``."""

    molecule_id: int
    experimental_size_nm: Optional[float] = None
    experimental_zeta_potential: Optional[float] = None
    experimental_loading_efficiency: Optional[float] = None
    experimental_cytotoxicity: Optional[float] = None
    experimental_date: Optional[date] = None
    lab_technician: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentalResult":
        if "molecule_id" not in data or data["molecule_id"] in (None, ""):
            raise ValueError("رکورد آزمایشگاهی بدون molecule_id قابل قبول نیست")

        exp_date = data.get("experimental_date")
        if isinstance(exp_date, str) and exp_date:
            exp_date = date.fromisoformat(exp_date)
        elif not exp_date:
            exp_date = None

        def _to_float(value):
            return float(value) if value not in (None, "") else None

        return cls(
            molecule_id=int(data["molecule_id"]),
            experimental_size_nm=_to_float(data.get("experimental_size_nm")),
            experimental_zeta_potential=_to_float(data.get("experimental_zeta_potential")),
            experimental_loading_efficiency=_to_float(data.get("experimental_loading_efficiency")),
            experimental_cytotoxicity=_to_float(data.get("experimental_cytotoxicity")),
            experimental_date=exp_date,
            lab_technician=data.get("lab_technician") or None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "molecule_id": self.molecule_id,
            "experimental_size_nm": self.experimental_size_nm,
            "experimental_zeta_potential": self.experimental_zeta_potential,
            "experimental_loading_efficiency": self.experimental_loading_efficiency,
            "experimental_cytotoxicity": self.experimental_cytotoxicity,
            "experimental_date": self.experimental_date.isoformat() if self.experimental_date else None,
            "lab_technician": self.lab_technician,
        }
