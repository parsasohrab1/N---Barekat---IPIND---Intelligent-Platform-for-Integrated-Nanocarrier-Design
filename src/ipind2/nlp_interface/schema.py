"""ساختار داده پارامترهای هدف استخراج‌شده از پرس‌وجوی کاربر. See docs/SRS.md §4.8 (FR-10)."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TargetParameters:
    """
    پارامترهای هدف طراحی نانوحامل، معادل ورودی واحد ۱ (تولید ساختار، FR-01).

    فیلدهای ``None`` یعنی آن قید در پرس‌وجو ذکر نشده و باید مقدار پیش‌فرض/از کاربر
    گرفته شود؛ ``unresolved_terms`` عبارت‌هایی را نشان می‌دهد که پارسر نتوانست به هیچ
    پارامتر شناخته‌شده‌ای نگاشت کند (برای بازخورد به کاربر یا بازبینی دستی مفید است).
    """

    scaffold_type: Optional[str] = None  # 'lipid' | 'polymer' | 'metal'
    target_tissue: Optional[str] = None
    size_range_nm: Optional[Tuple[float, float]] = None
    max_toxicity_ic50: Optional[float] = None
    min_loading_efficiency: Optional[float] = None
    raw_query: str = ""
    unresolved_terms: List[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """آیا حداقل نوع اسکلت و بافت هدف مشخص شده‌اند (کمینه لازم برای واحد تولید)."""
        return self.scaffold_type is not None and self.target_tissue is not None

    def to_dict(self) -> dict:
        return {
            "scaffold_type": self.scaffold_type,
            "target_tissue": self.target_tissue,
            "size_range_nm": self.size_range_nm,
            "max_toxicity_ic50": self.max_toxicity_ic50,
            "min_loading_efficiency": self.min_loading_efficiency,
            "unresolved_terms": list(self.unresolved_terms),
        }
