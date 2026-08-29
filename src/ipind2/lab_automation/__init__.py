"""Lab-in-the-loop integration: adapters for robotic synthesis/high-throughput screening equipment feeding the active-learning loop. See docs/SRS.md §4.9 (FR-11)."""

from .adapters import CSVLabAdapter, LabAdapter, RESTLabAdapter
from .ingestion import ingest_results, should_trigger_retrain
from .schema import ExperimentalResult

__all__ = [
    "ExperimentalResult",
    "LabAdapter",
    "CSVLabAdapter",
    "RESTLabAdapter",
    "ingest_results",
    "should_trigger_retrain",
]
