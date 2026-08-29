"""
رجیستری دیتاست‌های مرجع عمومی برای بنچمارک مستمر (FR-12).

این ماژول خودِ دیتاست‌ها را بسته‌بندی (bundle) نمی‌کند — دیتاست‌هایی مثل LNP-622 یا
LANCE منابع بیرونی با مجوز/توزیع خاص خودشان هستند (نگاه کنید به docs/BENCHMARK.md).
در عوض یک رجیستری از فراداده (metadata) آن‌ها نگه می‌دارد و یک لودر برای فایل CSV
محلی که کاربر از قبل دانلود کرده ارائه می‌دهد.

See docs/SRS.md §4.10 (FR-12).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class ReferenceDataset:
    """فراداده یک دیتاست مرجع عمومی برای بنچمارک."""

    name: str
    description: str
    source_url: str
    target_column: str


REGISTRY: Dict[str, ReferenceDataset] = {
    "lnp-622": ReferenceDataset(
        name="lnp-622",
        description="دیتاست تنظیم‌شده ۶۲۲ نمونه‌ای فرمولاسیون LNP (کارایی ترانسفکشن in-vitro).",
        source_url="https://arxiv.org/abs/2308.01402",
        target_column="transfection_efficiency",
    ),
    "lance": ReferenceDataset(
        name="lance",
        description="دیتاست LANCE، مورد استفاده برای آموزش مدل Transformer چندوظیفه‌ای COMET (کارایی + پایداری LNP).",
        source_url="https://www.nature.com/articles/s41565-025-01975-4",
        target_column="efficacy",
    ),
}


def list_reference_datasets() -> Dict[str, ReferenceDataset]:
    """فهرست دیتاست‌های مرجع شناخته‌شده (فقط فراداده، نه داده)."""
    return dict(REGISTRY)


def load_reference_dataset(name: str, csv_path: str) -> pd.DataFrame:
    """
    خواندن یک دیتاست مرجع از یک فایل CSV محلی (که باید از قبل توسط کاربر طبق منبع
    ذکرشده در ``REGISTRY[name].source_url`` دانلود شده باشد) و اعتبارسنجی وجود ستون هدف.
    """
    if name not in REGISTRY:
        raise KeyError(f"دیتاست مرجع ناشناخته: '{name}'. گزینه‌های موجود: {list(REGISTRY)}")

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"فایل دیتاست یافت نشد: {path}. طبق منبع {REGISTRY[name].source_url} دانلود کنید."
        )

    df = pd.read_csv(path)
    target_column = REGISTRY[name].target_column
    if target_column not in df.columns:
        raise ValueError(
            f"دیتاست '{name}' باید ستون هدف '{target_column}' را داشته باشد؛ "
            f"ستون‌های موجود: {list(df.columns)}"
        )
    return df
