"""
شمای رکورد نانوذره منطبق بر اصول FAIR و همسو با دسته‌بندی سه‌گانه MIRIBEL (FR-13).

MIRIBEL (Minimum Information Reporting in Bio–Nano Experimental Literature) گزارش حداقل
اطلاعات را در سه دسته می‌خواهد: **مشخصات ماده**، **مشخصات زیستی** و **پروتکل آزمایشی**.
فیلدهای این ماژول برداشت پروژه از آن سه دسته‌اند و باید با چک‌لیست رسمی منتشرشده
(Faria و همکاران، Nature Nanotechnology 2018) توسط یک متخصص تطبیق داده شوند؛ بنابراین
ادعای «همسو با MIRIBEL» است، نه «تأییدشده/منطبق رسمی» (نگاه کنید به docs/FAIR_MIRIBEL.md).

اصل بازاستفاده‌پذیری (R): هر مقدار صراحتاً ``provenance`` دارد — ``predicted`` (مدل و
نسخه آن) یا ``measured`` (روش اندازه‌گیری) — تا پیش‌بینی هرگز با اندازه‌گیری اشتباه
گرفته نشود؛ این برای تعامل با نهادهای تنظیم‌گر حیاتی است.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# دسته‌های MIRIBEL و فیلدهای لازم هر دسته
MATERIAL_FIELDS = (
    "smiles",
    "scaffold_type",
    "hydrodynamic_size_nm",
    "zeta_potential_mV",
    "pdi",
    "drug_loading_efficiency_percent",
)
BIOLOGICAL_FIELDS = (
    "cell_line",
    "cytotoxicity_ic50_ug_ml",
    "cellular_uptake_efficiency_percent",
)
PROTOCOL_FIELDS = (
    "characterization_method",  # مثلاً DLS
    "medium",
    "temperature_c",
    "exposure_time_h",
    "replicates",
    "operator",
    "date",
)
MIRIBEL_FIELDS = {
    "material": MATERIAL_FIELDS,
    "biological": BIOLOGICAL_FIELDS,
    "protocol": PROTOCOL_FIELDS,
}

# واحدها (مرجع QUDT) تا خروجی ماشین‌خوان و بدون ابهام باشد
UNITS = {
    "hydrodynamic_size_nm": "http://qudt.org/vocab/unit/NanoM",
    "zeta_potential_mV": "http://qudt.org/vocab/unit/MilliV",
    "temperature_c": "http://qudt.org/vocab/unit/DEG_C",
    "exposure_time_h": "http://qudt.org/vocab/unit/HR",
    "cytotoxicity_ic50_ug_ml": "http://qudt.org/vocab/unit/MicroGM-PER-MilliL",
    "drug_loading_efficiency_percent": "http://qudt.org/vocab/unit/PERCENT",
    "cellular_uptake_efficiency_percent": "http://qudt.org/vocab/unit/PERCENT",
    "pdi": "http://qudt.org/vocab/unit/UNITLESS",
}

PROVENANCE_VALUES = ("predicted", "measured")


@dataclass
class Observation:
    """یک مقدار همراه با منشأ."""

    value: Any
    provenance: str  # 'predicted' | 'measured'
    method: Optional[str] = None  # روش اندازه‌گیری یا نام مدل
    model_version: Optional[str] = None
    uncertainty: Optional[float] = None  # انحراف‌معیار ensemble یا خطای اندازه‌گیری

    def __post_init__(self):
        if self.provenance not in PROVENANCE_VALUES:
            raise ValueError(f"provenance باید یکی از {PROVENANCE_VALUES} باشد، نه {self.provenance!r}")
        if self.provenance == "predicted" and not self.model_version:
            raise ValueError("مقدار پیش‌بینی‌شده باید model_version داشته باشد (ردیابی‌پذیری)")


@dataclass
class NanoparticleRecord:
    """رکورد نانوذره/نانوحامل با سه دسته MIRIBEL."""

    identifier: str  # شناسه پایدار (urn:uuid:...)
    material: Dict[str, Observation] = field(default_factory=dict)
    biological: Dict[str, Observation] = field(default_factory=dict)
    protocol: Dict[str, Observation] = field(default_factory=dict)
    license: str = "CC-BY-4.0"
    created: Optional[str] = None
    notes: List[str] = field(default_factory=list)
