"""
Ingestion and preparation of lab results for the database and the active learning loop.

Flow: LabAdapter.fetch_new_results() -> ingest_results() -> DataFrame ready for
insertion into the experimental_results table and/or starting an active learning round (Unit 6, FR-06).

See docs/SRS.md §4.9 (FR-11) and §4.6 (FR-06).
"""

from typing import List

import pandas as pd

from .adapters import LabAdapter
from .schema import ExperimentalResult

# Per SRS §4.6: "Update frequency: after every 10-50 new lab data points"
DEFAULT_MIN_RETRAIN_BATCH = 10
DEFAULT_MAX_RETRAIN_BATCH = 50


def ingest_results(adapter: LabAdapter) -> pd.DataFrame:
    """Takes new results from an adapter and converts them to a DataFrame matching experimental_results."""
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
    Whether the number of new lab data points is sufficient for the active learning unit to start a new round of
    fine-tuning (per the SRS §4.6 threshold: every 10-50 records).
    """
    if min_batch <= 0:
        raise ValueError("min_batch must be positive")
    return new_result_count >= min_batch
