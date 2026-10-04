"""
کتابخانه بلوک‌های سازنده و قالب‌های ساختاری نانوحامل (ورودی واحد ۱، FR-01).

رویکرد: ساختارها به‌صورت «قالب + بلوک» ساخته می‌شوند، نه با تولید کاراکتر‌به‌کاراکتر
SMILES. دلیل: FR-01 نرخ اعتبار > ۹۵٪ می‌خواهد؛ قالب‌های اینجا طوری نوشته شده‌اند که
هر ترکیب مجاز بلوک‌ها، SMILES نحویاً و شیمیاً معتبر بدهد (در ``tests/test_generation.py``
نرخ اعتبار هر سه کلاس اسکلت سنجیده می‌شود). مدل مولد (CVAE) روی *انتخاب* بلوک‌ها و
فضای توصیف‌گر کار می‌کند، نه روی نحو SMILES.

ترکیب بلوک‌ها با الحاق رشته‌ای انجام می‌شود؛ هر بلوک میانی طوری نوشته شده که اتم آخرش
ظرفیت پیوند با بلوک بعدی را داشته باشد.

See docs/SRS.md §4.1 (FR-01).
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

SCAFFOLD_TYPES: Tuple[str, ...] = ("lipid", "polymer", "metal")


def _alkyl_tails() -> List[str]:
    """
    دم‌های آبگریز: اشباع، غیراشباع، شاخه‌دار، هیدروکسیله و زیست‌تخریب‌پذیر (استردار).

    تنوع طول/اشباع زنجیره مهم‌ترین اهرم تنظیم اندازه و pKa در لیپیدهای یونیزه‌شونده است،
    بنابراین دامنه آن (۶ تا ۲۴ کربن) پوشش داده می‌شود.
    """
    tails: List[str] = []
    for n in range(6, 25):
        tails.append("C" * n)  # اشباع
    for n in range(12, 25, 2):
        for position in (n // 3, n // 2):
            tails.append("C" * position + "C=C" + "C" * (n - position - 2))  # غیراشباع
    for n in range(9, 22, 3):
        tails.append("C" * (n - 3) + "C(C)C")  # شاخه‌دار انتهایی
    for n in range(10, 22, 3):
        tails.append("C" * (n // 2) + "C(O)" + "C" * (n - n // 2 - 1))  # هیدروکسیله
    for n in range(10, 22, 3):
        tails.append("C" * (n // 2) + "C(=O)O" + "C" * (n - n // 2 - 1))  # استر زیست‌تخریب‌پذیر
    return tails


TAILS: Tuple[str, ...] = tuple(_alkyl_tails())

# لینکرهای قابل‌زیست‌تخریب/پایدار بین دم و سر قطبی
LINKERS: Tuple[str, ...] = (
    "C(=O)O",      # استر
    "OC(=O)",      # استر معکوس
    "C(=O)N",      # آمید
    "NC(=O)",      # آمید معکوس
    "OC(=O)N",     # کاربامات
    "SSC",         # دی‌سولفید (پاسخ‌دهنده به ردوکس)
    "O",           # اتر
    "N",           # آمین ثانویه
    "S",           # تیواتر
    "OCCO",        # اتیلن‌گلیکول کوتاه
    "CC(O)C",      # پروپانول‌دیول
    "C(=O)OCC",    # استر + اسپیسر
)

# سرهای قطبی/یونیزه‌شونده (انتهایی)
POLAR_HEADS: Tuple[str, ...] = (
    "[N+](C)(C)C",                   # آمونیوم چهارتایی (کاتیونی دائم)
    "N(C)C",                         # آمین نوع سوم (یونیزه‌شونده)
    "N(CC)CC",
    "NCCN(C)C",                      # پلی‌آمین
    "N",                             # آمین اولیه
    "O",                             # هیدروکسیل
    "OCCO",
    "OCCOCCOCCO",                    # PEG کوتاه
    "C(=O)O",                        # کربوکسیل (آنیونی)
    "C(N)=O",                        # آمید اولیه
    "N1CCOCC1",                      # مورفولین
    "N1CCCC1",                       # پیرولیدین
    "N1CCN(C)CC1",                   # پیپرازین
    "c1ccncc1",                      # پیریدین
    "OS(=O)(=O)O",                   # سولفات
    "OP(=O)(O)O",                    # فسفات
)

# سرهای گلیسرولی/فسفولیپیدی (برای قالب دو-دمی)
GLYCEROL_HEADS: Tuple[str, ...] = (
    "COP(=O)(O)OCC[N+](C)(C)C",      # فسفوکولین
    "COP(=O)(O)OCCN",                # فسفواتانول‌آمین
    "COP(=O)(O)OCC(N)C(=O)O",        # فسفوسرین
    "COP(=O)(O)OCCOCCOCCO",          # PEG-فسفات
    "CO",                            # دی‌آسیل‌گلیسرول
    "COC(=O)CCC(=O)O",               # سوکسینیل
    "COP(=O)(O)OCC(O)CO",            # فسفوگلیسرول
    "COP(=O)(O)O",                   # فسفاتیدیک‌اسید
    "COCC(O)CO",                     # گلیسریل‌اتر
    "COC(=O)N",                      # کاربامات
    "COS(=O)(=O)O",                  # سولفات
    "COCCN(C)C",                     # آمینواتیل‌اتر
)

# کلاهک انتهایی برای قالب دی‌آلکیل‌آمین
AMINE_CAPS: Tuple[str, ...] = (
    "C",
    "CCO",
    "CC(=O)O",
    "CCN(C)C",
    "CCOCCO",
    "CC(O)CO",
    "CCS",
    "CC(N)=O",
    "CCOCCOCCO",
    "CC1CCOCC1",
    "CCN1CCOCC1",
    "CC(O)C(O)CO",
)

# واحدهای تکرارشونده پلیمری (هر واحد خودبسنده و قابل‌تکرار است)
POLYMER_REPEAT_UNITS: Dict[str, str] = {
    "PLA": "OC(C)C(=O)",
    "PGA": "OCC(=O)",
    "PEG": "OCC",
    "PCL": "OCCCCCC(=O)",
    "PAA": "CC(C(=O)O)",
    "PNIPAM": "CC(C(=O)NC(C)C)",
    "chitosan_like": "OC(CO)C(N)",
    "PBAE": "OCCOC(=O)CCN(C)CCC(=O)",
}

POLYMER_INITIATORS: Tuple[str, ...] = ("CC", "CCCC", "CO", "OCC", "NCC", "c1ccccc1C")
POLYMER_TERMINATORS: Tuple[str, ...] = ("O", "OC", "N", "OCCO", "C(=O)O", "OCCOCCOCCO")

# لنگرهای سطحی نانوذرات فلزی/معدنی (انتهایی)
METAL_ANCHORS: Dict[str, str] = {
    "Au": "S[Au]",
    "Au_dithiol": "SC(S)[Au]",
    "MSN": "[Si](O)(O)O",
    "Fe3O4": "C(=O)O[Fe]",
    "ZnO": "OC(=O)[Zn]",
    "MnO": "OC(=O)[Mn]",
    "Gd_chelate": "N(CC(=O)O)CC(=O)O[Gd]",
    "Si_amino": "[Si](O)(O)OCCN",
}


@dataclass(frozen=True)
class StructureTemplate:
    """
    یک قالب ساختاری: الگوی رشته‌ای با جایگاه‌های نام‌دار و دامنه مجاز هر جایگاه.

    Attributes:
        name: شناسه قالب (در خروجی به‌عنوان ``template`` گزارش می‌شود).
        scaffold_type: 'lipid' | 'polymer' | 'metal'.
        pattern: الگو با جایگاه‌های ``{slot}``.
        slots: نام جایگاه -> فهرست مقادیر مجاز.
    """

    name: str
    scaffold_type: str
    pattern: str
    slots: Tuple[Tuple[str, Tuple[str, ...]], ...]

    def slot_options(self) -> Dict[str, Tuple[str, ...]]:
        return dict(self.slots)

    def build(self, choices: Dict[str, str]) -> str:
        return self.pattern.format(**choices)

    def combination_count(self) -> int:
        total = 1
        for _, options in self.slots:
            total *= len(options)
        return total


_REPEAT_BLOCKS: Tuple[str, ...] = tuple(
    unit * n
    for unit in POLYMER_REPEAT_UNITS.values()
    for n in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
)
_REPEAT_BLOCKS_SHORT: Tuple[str, ...] = tuple(
    unit * n for unit in POLYMER_REPEAT_UNITS.values() for n in (1, 2, 3, 4, 6, 8)
)


TEMPLATES: Tuple[StructureTemplate, ...] = (
    StructureTemplate(
        name="lipid_single_chain",
        scaffold_type="lipid",
        pattern="{tail1}{linker}{head}",
        slots=(("tail1", TAILS), ("linker", LINKERS), ("head", POLAR_HEADS)),
    ),
    StructureTemplate(
        name="lipid_head_first",
        scaffold_type="lipid",
        pattern="C{head}{linker}{tail1}",
        slots=(("head", ("N(C)", "N(CC)", "N(CCO)")), ("linker", LINKERS), ("tail1", TAILS)),
    ),
    StructureTemplate(
        name="phospholipid_two_tail",
        scaffold_type="lipid",
        pattern="{tail1}C(=O)OCC(OC(=O){tail2}){head}",
        slots=(("tail1", TAILS), ("tail2", TAILS), ("head", GLYCEROL_HEADS)),
    ),
    StructureTemplate(
        name="dialkyl_amine",
        scaffold_type="lipid",
        pattern="{tail1}N({tail2}){cap}",
        slots=(("tail1", TAILS), ("tail2", TAILS), ("cap", AMINE_CAPS)),
    ),
    StructureTemplate(
        name="lipidoid_three_tail",
        scaffold_type="lipid",
        pattern="{tail1}N({tail2})CCN({tail3}){cap}",
        slots=(("tail1", TAILS), ("tail2", TAILS), ("tail3", TAILS), ("cap", AMINE_CAPS)),
    ),
    StructureTemplate(
        name="cationic_dialkyl_quat",
        scaffold_type="lipid",
        pattern="{tail1}[N+]({tail2})(C)C",
        slots=(("tail1", TAILS), ("tail2", TAILS)),
    ),
    StructureTemplate(
        name="dotap_like_quat",
        scaffold_type="lipid",
        pattern="{tail1}C(=O)OCC(OC(=O){tail2})C[N+](C)(C)C",
        slots=(("tail1", TAILS), ("tail2", TAILS)),
    ),
    StructureTemplate(
        name="quat_single_chain",
        scaffold_type="lipid",
        pattern="{tail1}{linker}CC[N+]({cap})(C)C",
        slots=(("tail1", TAILS), ("linker", LINKERS), ("cap", ("C", "CC", "CCO", "CCCC", "CC(O)C"))),
    ),
    StructureTemplate(
        name="lipid_diester",
        scaffold_type="lipid",
        pattern="{tail1}OC(=O)CCC(=O)O{tail2}",
        slots=(("tail1", TAILS), ("tail2", TAILS)),
    ),
    StructureTemplate(
        name="polymer_homo",
        scaffold_type="polymer",
        pattern="{initiator}{repeat}{terminator}",
        slots=(
            ("initiator", POLYMER_INITIATORS),
            ("repeat", _REPEAT_BLOCKS),
            ("terminator", POLYMER_TERMINATORS),
        ),
    ),
    StructureTemplate(
        name="polymer_block_copolymer",
        scaffold_type="polymer",
        pattern="{initiator}{repeat_a}{repeat_b}{terminator}",
        slots=(
            ("initiator", POLYMER_INITIATORS),
            ("repeat_a", _REPEAT_BLOCKS_SHORT),
            ("repeat_b", _REPEAT_BLOCKS_SHORT),
            ("terminator", POLYMER_TERMINATORS),
        ),
    ),
    StructureTemplate(
        name="polymer_lipid_conjugate",
        scaffold_type="polymer",
        pattern="{tail1}C(=O)O{repeat}{terminator}",
        slots=(
            ("tail1", TAILS),
            ("repeat", _REPEAT_BLOCKS_SHORT),
            ("terminator", POLYMER_TERMINATORS),
        ),
    ),
    StructureTemplate(
        name="polymer_segmented",
        scaffold_type="polymer",
        pattern="{initiator}{repeat_a}{linker}{repeat_b}{terminator}",
        slots=(
            ("initiator", POLYMER_INITIATORS),
            ("repeat_a", _REPEAT_BLOCKS_SHORT),
            ("linker", LINKERS),
            ("repeat_b", _REPEAT_BLOCKS_SHORT),
            ("terminator", POLYMER_TERMINATORS),
        ),
    ),
    StructureTemplate(
        name="metal_mixed_ligand",
        scaffold_type="metal",
        pattern="{tail1}{linker}N({tail2}){anchor}",
        slots=(
            ("tail1", TAILS),
            ("linker", LINKERS),
            ("tail2", TAILS),
            ("anchor", tuple(METAL_ANCHORS.values())),
        ),
    ),
    StructureTemplate(
        name="metal_ligand",
        scaffold_type="metal",
        pattern="{tail1}{anchor}",
        slots=(("tail1", TAILS), ("anchor", tuple(METAL_ANCHORS.values()))),
    ),
    StructureTemplate(
        name="metal_ligand_linker",
        scaffold_type="metal",
        pattern="{tail1}{linker}{anchor}",
        slots=(
            ("tail1", TAILS),
            ("linker", LINKERS),
            ("anchor", tuple(METAL_ANCHORS.values())),
        ),
    ),
    StructureTemplate(
        name="metal_peg_ligand",
        scaffold_type="metal",
        pattern="{peg}{anchor}",
        slots=(
            ("peg", tuple("OCC" * n + "C" for n in (2, 4, 6, 8, 12, 16))),
            ("anchor", tuple(METAL_ANCHORS.values())),
        ),
    ),
)


def templates_for(scaffold_type: str = None) -> List[StructureTemplate]:
    """قالب‌های یک کلاس اسکلت (یا همه قالب‌ها اگر ``None``)."""
    if scaffold_type is None:
        return list(TEMPLATES)
    if scaffold_type not in SCAFFOLD_TYPES:
        raise ValueError(
            f"scaffold_type نامعتبر: {scaffold_type!r} (مجاز: {', '.join(SCAFFOLD_TYPES)})"
        )
    return [t for t in TEMPLATES if t.scaffold_type == scaffold_type]


def theoretical_library_size(scaffold_type: str = None) -> int:
    """
    اندازه نظری فضای ساختاری قابل‌دسترس (حد بالای کتابخانه مجازی).

    مبنای ادعای NFR-08/FR-01 درباره مقیاس کتابخانه: فضای قابل‌شمارش قالب‌ها.
    """
    return sum(t.combination_count() for t in templates_for(scaffold_type))


def all_building_blocks() -> Dict[str, Sequence[str]]:
    """همه بلوک‌های سازنده، برای مستندسازی و گزارش پوشش شیمیایی."""
    return {
        "tails": TAILS,
        "linkers": LINKERS,
        "polar_heads": POLAR_HEADS,
        "glycerol_heads": GLYCEROL_HEADS,
        "amine_caps": AMINE_CAPS,
        "polymer_repeat_units": tuple(POLYMER_REPEAT_UNITS.values()),
        "polymer_initiators": POLYMER_INITIATORS,
        "polymer_terminators": POLYMER_TERMINATORS,
        "metal_anchors": tuple(METAL_ANCHORS.values()),
    }
