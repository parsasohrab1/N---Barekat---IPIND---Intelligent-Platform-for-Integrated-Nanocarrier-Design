"""
تولید کتابخانه مجازی ساختارها (FR-01).

``CombinatorialLibrary`` قالب‌های ``building_blocks`` را نمونه‌برداری یا شمارش می‌کند و
خروجی را با RDKit اعتبارسنجی و یکتاسازی می‌کند. سنجه‌های FR-01 (تعداد، نرخ اعتبار،
تنوع اسکلت) در ``LibraryStats`` گزارش می‌شوند تا در CI قابل‌سنجش باشند.

See docs/SRS.md §4.1 (FR-01), NFR-04 (زمان تولید)، NFR-08 (مقیاس‌پذیری).
"""

import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..featurization import (
    canonical_smiles,
    descriptor_vector,
    generic_framework,
    parse_smiles,
)
from .building_blocks import (
    SCAFFOLD_TYPES,
    StructureTemplate,
    templates_for,
    theoretical_library_size,
)


@dataclass
class GeneratedStructure:
    """یک ساختار تولیدشده همراه با منشأ (provenance) آن."""

    smiles: str
    scaffold_type: str
    template: str
    slots: Dict[str, str] = field(default_factory=dict)
    descriptors: Optional[np.ndarray] = None
    condition_distance: Optional[float] = None

    def to_dict(self) -> dict:
        row = {
            "smiles": self.smiles,
            "scaffold_type": self.scaffold_type,
            "template": self.template,
        }
        if self.descriptors is not None:
            from ..featurization import DESCRIPTOR_NAMES

            row.update(
                {f"desc_{name}": float(v) for name, v in zip(DESCRIPTOR_NAMES, self.descriptors)}
            )
        if self.condition_distance is not None:
            row["condition_distance"] = self.condition_distance
        return row


@dataclass
class LibraryStats:
    """سنجه‌های یک اجرای تولید کتابخانه — مبنای بررسی انطباق با FR-01/NFR-04."""

    requested: int
    proposed: int
    valid: int
    unique: int
    elapsed_seconds: float
    distinct_skeletons: int = 0
    attempts: int = 0  # تعداد تلاش/بازیابی خام (برای مولد شرطی، شامل تکراری‌ها)

    @property
    def validity_rate(self) -> float:
        """نرخ ساختارهای معتبر RDKit در میان پیشنهادهای مدل (هدف FR-01: > ۰.۹۵)."""
        return self.valid / self.proposed if self.proposed else 0.0

    @property
    def structures_per_second(self) -> float:
        return self.unique / self.elapsed_seconds if self.elapsed_seconds > 0 else float("inf")

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "proposed": self.proposed,
            "valid": self.valid,
            "unique": self.unique,
            "validity_rate": self.validity_rate,
            "distinct_skeletons": self.distinct_skeletons,
            "elapsed_seconds": self.elapsed_seconds,
            "structures_per_second": self.structures_per_second,
        }


class CombinatorialLibrary:
    """
    نمونه‌بردار/شمارنده فضای ساختاری نانوحامل.

    Args:
        scaffold_type: محدودکردن به یک کلاس اسکلت ('lipid'|'polymer'|'metal')؛
            ``None`` یعنی همه کلاس‌ها.
        seed: بذر تصادفی برای بازتولیدپذیری (الزام تکرارپذیری آزمایش‌ها).
    """

    def __init__(
        self,
        scaffold_type: Optional[str] = None,
        seed: int = 0,
        balance_classes: bool = True,
        template_weighting: str = "uniform",
    ):
        if scaffold_type is not None and scaffold_type not in SCAFFOLD_TYPES:
            raise ValueError(f"scaffold_type نامعتبر: {scaffold_type!r}")
        self.scaffold_type = scaffold_type
        self.templates: List[StructureTemplate] = templates_for(scaffold_type)
        if not self.templates:
            raise ValueError("هیچ قالب ساختاری برای این نوع اسکلت یافت نشد")
        self._rng = random.Random(seed)
        self._template_index = {t.name: i for i, t in enumerate(self.templates)}
        self._combinations = [t.combination_count() for t in self.templates]
        # دو لایه وزن‌دهی:
        #  • ``balance_classes``: سهم هر کلاس اسکلت برابر (وگرنه کلاس فلزی با ~۸٪ فضا، کم‌نماینده می‌ماند)؛
        #  • ``template_weighting``: ``'uniform'`` (پیش‌فرض) هر قالبِ یک کلاس را هم‌شانس می‌کند؛
        #    ``'space'`` متناسب با اندازه فضای قالب. با 'space' قالب سه‌دمی lipidoid (~۱٫۲M ترکیب)
        #    ≈۹۵٪ لیپیدها را می‌گرفت و آمونیوم‌های چهارتایی/فسفولیپیدها عملاً نمونه نمی‌شدند.
        if template_weighting not in ("uniform", "space"):
            raise ValueError("template_weighting باید uniform یا space باشد")
        per_class: Dict[str, List[int]] = {}
        for index, t in enumerate(self.templates):
            per_class.setdefault(t.scaffold_type, []).append(index)
        raw = [
            1.0 if template_weighting == "uniform" else float(t.combination_count())
            for t in self.templates
        ]
        class_totals = {c: sum(raw[i] for i in idx) for c, idx in per_class.items()}
        if balance_classes:
            self._weights = [raw[i] / class_totals[t.scaffold_type] for i, t in enumerate(self.templates)]
        else:
            self._weights = raw

    @property
    def space_size(self) -> int:
        """اندازه نظری فضای ساختاری قابل‌دسترس."""
        return theoretical_library_size(self.scaffold_type)

    def _effective_weights(self, used: List[int]) -> List[float]:
        """
        وزن قالب‌ها با کسر «باقی‌مانده فضا». قالب‌های کوچک (۴۸–۳۶۸ ترکیب) با وزن یکنواخت
        زود اشباع می‌شوند و فقط تکراری برمی‌گردانند؛ وزن هر قالب با پر‌شدن فضایش کم و در
        اشباع صفر می‌شود تا درخواست‌های بزرگ (۱۰۰k ساختار یکتا) بن‌بست نخورند.
        """
        return [
            w * max(0.0, 1.0 - u / c) for w, u, c in zip(self._weights, used, self._combinations)
        ]

    def _propose(self, weights: Optional[List[float]] = None) -> Tuple[str, StructureTemplate, Dict[str, str]]:
        template = self._rng.choices(self.templates, weights=weights or self._weights, k=1)[0]
        choices = {
            slot: self._rng.choice(options) for slot, options in template.slots
        }
        return template.build(choices), template, choices

    def propose_many(self, n: int) -> Iterator[Tuple[str, StructureTemplate, Dict[str, str]]]:
        """پیشنهاد خام (بدون اعتبارسنجی) — برای سنجش نرخ اعتبار مدل مولد."""
        for _ in range(n):
            yield self._propose()

    def generate(
        self,
        n: int,
        with_descriptors: bool = False,
        unique: bool = True,
        max_attempt_factor: float = 1.5,
        track_skeletons: bool = False,
    ) -> Tuple[List[GeneratedStructure], LibraryStats]:
        """
        تولید ``n`` ساختار معتبر (و پیش‌فرض: یکتا).

        Args:
            with_descriptors: محاسبه بردار توصیف‌گر RDKit برای هر ساختار (کندتر).
            unique: حذف تکراری‌ها بر پایه SMILES کانونیک.
            max_attempt_factor: حداکثر تلاش = ``n * max_attempt_factor`` (محافظ در برابر
                درخواست بیش از اندازه فضای ساختاری).
            track_skeletons: شمارش اسکلت‌های متمایز (سنجه تنوع FR-01؛ کندتر).
        """
        if n <= 0:
            raise ValueError("n باید مثبت باشد")

        started = time.perf_counter()
        structures: List[GeneratedStructure] = []
        seen: set = set()
        skeletons: set = set()
        proposed = valid = 0
        max_attempts = max(int(n * max_attempt_factor), n + 100)
        used = [0] * len(self.templates)
        refresh_every = max(25, min(500, n // 10))
        weights = list(self._weights)

        while len(structures) < n and proposed < max_attempts:
            if proposed and proposed % refresh_every == 0:
                refreshed = self._effective_weights(used)
                if sum(refreshed) <= 0:
                    break  # کل فضا اشباع شد؛ ساختار یکتای بیشتری وجود ندارد
                weights = refreshed
            smiles, template, choices = self._propose(weights)
            proposed += 1
            mol = parse_smiles(smiles)
            if mol is None:
                continue
            valid += 1
            if unique:
                key = canonical_smiles(smiles)
                if key in seen:
                    continue
                seen.add(key)
            used[self._template_index[template.name]] += 1
            if track_skeletons:
                framework = generic_framework(smiles)
                if framework:
                    skeletons.add(framework)
            structures.append(
                GeneratedStructure(
                    smiles=smiles,
                    scaffold_type=template.scaffold_type,
                    template=template.name,
                    slots=choices,
                    descriptors=descriptor_vector(mol) if with_descriptors else None,
                )
            )

        stats = LibraryStats(
            requested=n,
            proposed=proposed,
            valid=valid,
            unique=len(structures),
            elapsed_seconds=time.perf_counter() - started,
            distinct_skeletons=len(skeletons),
        )
        return structures, stats

    def enumerate_all(self, limit: Optional[int] = None) -> Iterator[GeneratedStructure]:
        """
        شمارش قطعی (deterministic) فضای ساختاری — برای ساخت کتابخانه seed کامل.

        برخلاف ``generate`` که نمونه‌برداری تصادفی می‌کند، این متد فضا را به‌ترتیب و
        بدون تکرار پیمایش می‌کند (مناسب NFR-08: ≥۱ میلیون ساختار).
        """
        produced = 0
        for template in self.templates:
            slot_names = [slot for slot, _ in template.slots]
            option_lists = [options for _, options in template.slots]
            for combination in itertools.product(*option_lists):
                if limit is not None and produced >= limit:
                    return
                choices = dict(zip(slot_names, combination))
                smiles = template.build(choices)
                if parse_smiles(smiles) is None:
                    continue
                produced += 1
                yield GeneratedStructure(
                    smiles=smiles,
                    scaffold_type=template.scaffold_type,
                    template=template.name,
                    slots=choices,
                )


def generate_library(
    n: int = 100_000,
    scaffold_type: Optional[str] = None,
    seed: int = 0,
    with_descriptors: bool = False,
    track_skeletons: bool = False,
) -> Tuple[List[GeneratedStructure], LibraryStats]:
    """تابع راحتی برای یک اجرای تولید کتابخانه مجازی (FR-01)."""
    library = CombinatorialLibrary(scaffold_type=scaffold_type, seed=seed)
    return library.generate(
        n, with_descriptors=with_descriptors, track_skeletons=track_skeletons
    )


def structures_to_dataframe(structures: Sequence[GeneratedStructure]):
    """تبدیل خروجی تولید به ``pandas.DataFrame`` (برای درج در پایگاه داده/خروجی CSV)."""
    import pandas as pd

    return pd.DataFrame([s.to_dict() for s in structures])


def descriptor_matrix_of(structures: Iterable[GeneratedStructure]) -> np.ndarray:
    """ماتریس توصیف‌گر ساختارها؛ در صورت نبودِ توصیف‌گر، محاسبه می‌شود."""
    rows = []
    for structure in structures:
        if structure.descriptors is None:
            structure.descriptors = descriptor_vector(structure.smiles)
        if structure.descriptors is not None:
            rows.append(structure.descriptors)
    if not rows:
        from ..featurization import DESCRIPTOR_DIM

        return np.zeros((0, DESCRIPTOR_DIM), dtype=np.float32)
    return np.vstack(rows)
