"""
دریافت و آماده‌سازی نتایج آزمایشگاهی برای پایگاه داده و حلقه یادگیری فعال.

جریان: LabAdapter.fetch_new_results() -> ingest_results() -> DataFrame آماده برای
درج در جدول experimental_results و/یا شروع یک دور یادگیری فعال (واحد ۶، FR-06).

See docs/SRS.md §4.9 (FR-11) و §4.6 (FR-06).
"""

from typing import List

import pandas as pd

from .adapters import LabAdapter
from .schema import ExperimentalResult

# طبق SRS §4.6: «دفعات به‌روزرسانی: پس از هر ۱۰-۵۰ داده آزمایشگاهی جدید»
DEFAULT_MIN_RETRAIN_BATCH = 10
DEFAULT_MAX_RETRAIN_BATCH = 50


def ingest_results(adapter: LabAdapter) -> pd.DataFrame:
    """نتایج جدید را از یک آداپتور می‌گیرد و به DataFrame منطبق بر experimental_results تبدیل می‌کند."""
    results: List[ExperimentalResult] = adapter.fetch_new_results()
    if not results:
        return pd.DataFrame(
            columns=[
                "molecule_id",
                "experimental_size_nm",
                "experimental_zeta_potential",
                "experimental_loading_efficiency",
                "experimental_cytotoxicity",
                "experimental_date",
                "lab_technician",
            ]
        )
    return pd.DataFrame([r.to_dict() for r in results])


def should_trigger_retrain(
    new_result_count: int,
    min_batch: int = DEFAULT_MIN_RETRAIN_BATCH,
) -> bool:
    """
    آیا تعداد داده‌های آزمایشگاهی جدید کافی است تا واحد یادگیری فعال یک دور
    fine-tuning جدید را آغاز کند (طبق آستانه SRS §4.6: هر ۱۰-۵۰ رکورد).
    """
    if min_batch <= 0:
        raise ValueError("min_batch باید مثبت باشد")
    return new_result_count >= min_batch
