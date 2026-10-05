"""
Registry of public reference datasets for continuous benchmarking (FR-12).

This module does not bundle the datasets themselves — datasets such as LNP-622 or
LANCE are external sources with their own licensing/distribution (see docs/BENCHMARK.md).
Instead it keeps a registry of their metadata and provides a loader for a local
CSV file that the user has already downloaded.

See docs/SRS.md §4.10 (FR-12).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class ReferenceDataset:
    """Metadata of a public reference dataset for benchmarking."""

    name: str
    description: str
    source_url: str
    target_column: str


REGISTRY: Dict[str, ReferenceDataset] = {
    "lnp-622": ReferenceDataset(
        name="lnp-622",
        description="Curated 622-sample LNP formulation dataset (in-vitro transfection efficiency).",
        source_url="https://arxiv.org/abs/2308.01402",
        target_column="transfection_efficiency",
    ),
    "lance": ReferenceDataset(
        name="lance",
        description="LANCE dataset, used for training the COMET multi-task Transformer model (LNP efficacy + stability).",
        source_url="https://www.nature.com/articles/s41565-025-01975-4",
        target_column="efficacy",
    ),
    "lantern-hela": ReferenceDataset(
        name="lantern-hela",
        description="۱۱۰۰ لیپید یونیزه‌شونده با کارایی ترانسفکشن تجربی HeLa (LANTERN/AGILE، data/AGILE.csv، مجوز MIT).",
        source_url="https://github.com/AsalMehradfar/LANTERN",
        target_column="Target",
    ),
}


def list_reference_datasets() -> Dict[str, ReferenceDataset]:
    """List of known reference datasets (metadata only, not data)."""
    return dict(REGISTRY)


def load_reference_dataset(name: str, csv_path: str) -> pd.DataFrame:
    """
    Read a reference dataset from a local CSV file (which the user must have already downloaded per the
    source given in ``REGISTRY[name].source_url``) and validate that the target column exists.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown reference dataset: '{name}'. Available options: {list(REGISTRY)}")

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Download it per the source {REGISTRY[name].source_url}."
        )

    df = pd.read_csv(path)
    target_column = REGISTRY[name].target_column
    if target_column not in df.columns:
        raise ValueError(
            f"Dataset '{name}' must have target column '{target_column}'; "
            f"available columns: {list(df.columns)}"
        )
    return df
