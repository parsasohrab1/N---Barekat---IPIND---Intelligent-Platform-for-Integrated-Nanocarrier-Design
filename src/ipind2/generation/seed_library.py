"""
Building the initial seed library (NFR-10: ≥ 1 million reference structures).

Two sources:

* **Combinatorial** — sampling *without replacement* from the entire template space (about 2.5 million
  structures), with no repeated index and with RDKit validation;
* **Public databases** — SMILES/CSV files that the user has downloaded from PubChem/ZINC
  (not shipped with the repo due to license and size).

⚠️ NFR-10 requires the origin to be "public databases". The ``SeedLibraryReport.n_public`` report explicitly
shows how many structures came from public files; if it is zero, the "at least 1 million" condition is
satisfied from the combinatorial source and not from a public database.
"""

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..featurization import canonical_smiles, is_valid_smiles
from .building_blocks import TEMPLATES, StructureTemplate

TARGET_SIZE = 1_000_000


@dataclass
class SeedLibraryReport:
    path: str
    n_total: int
    n_combinatorial: int
    n_public: int
    n_rejected: int
    elapsed_seconds: float

    @property
    def meets_size_requirement(self) -> bool:
        return self.n_total >= TARGET_SIZE

    @property
    def meets_public_source_requirement(self) -> bool:
        return self.n_public >= TARGET_SIZE


def _decode(template: StructureTemplate, index: int) -> str:
    """Map a combined index (mixed radix) to the template SMILES."""
    choices = {}
    for slot, options in reversed(template.slots):
        index, position = divmod(index, len(options))
        choices[slot] = options[position]
    return template.build(choices)


def _sample_indices(n: int, seed: int) -> Iterator[Tuple[StructureTemplate, int]]:
    counts = np.array([t.combination_count() for t in TEMPLATES], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    total = int(offsets[-1])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total, size=min(n, total), replace=False)
    for global_index in chosen:
        template_index = int(np.searchsorted(offsets, global_index, side="right") - 1)
        yield TEMPLATES[template_index], int(global_index - offsets[template_index])


def read_public_smiles(paths: Sequence[str]) -> Iterator[str]:
    """Read SMILES from ``.smi``/``.txt`` files (first column) or ``.csv`` (smiles/SMILES column)."""
    for path in paths:
        file = Path(path)
        if file.suffix.lower() == ".csv":
            with file.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                column = next((c for c in (reader.fieldnames or []) if c.lower() == "smiles"), None)
                if column is None:
                    raise ValueError(f"smiles column not found in {file.name}")
                for row in reader:
                    yield row[column]
        else:
            with file.open(encoding="utf-8") as handle:
                for line in handle:
                    token = line.strip().split()[0] if line.strip() else ""
                    if token and not token.startswith("#"):
                        yield token


def build_seed_library(
    path: str,
    n: int = TARGET_SIZE,
    seed: int = 0,
    public_files: Sequence[str] = (),
    chunk_size: int = 50_000,
    progress: Optional[callable] = None,
) -> SeedLibraryReport:
    """
    Build the seed library and save it as Parquet (columns: ``smiles``, ``source``,
    ``scaffold_type``, ``template``).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.perf_counter()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [("smiles", pa.string()), ("source", pa.string()), ("scaffold_type", pa.string()), ("template", pa.string())]
    )
    seen: set = set()
    rejected = n_public = n_comb = 0
    buffer: List[Tuple[str, str, str, str]] = []

    with pq.ParquetWriter(str(out), schema, compression="zstd") as writer:

        def flush() -> None:
            if not buffer:
                return
            columns = list(zip(*buffer))
            writer.write_table(pa.table(dict(zip(schema.names, map(list, columns))), schema=schema))
            buffer.clear()

        for smiles in read_public_smiles(public_files):
            canonical = canonical_smiles(smiles)
            if canonical is None or canonical in seen:
                rejected += 1
                continue
            seen.add(canonical)
            buffer.append((smiles, "public", "", ""))
            n_public += 1
            if len(buffer) >= chunk_size:
                flush()

        target_comb = max(n - n_public, 0)
        # Slightly oversample so we reach the target after removing invalid/duplicate entries
        for template, index in _sample_indices(int(target_comb * 1.05) + 10, seed):
            if n_comb >= target_comb:
                break
            smiles = _decode(template, index)
            canonical = canonical_smiles(smiles)
            if canonical is None or canonical in seen:
                rejected += 1
                continue
            seen.add(canonical)
            buffer.append((smiles, "combinatorial", template.scaffold_type, template.name))
            n_comb += 1
            if len(buffer) >= chunk_size:
                flush()
                if progress:
                    progress(n_public + n_comb)
        flush()

    return SeedLibraryReport(
        path=str(out),
        n_total=n_public + n_comb,
        n_combinatorial=n_comb,
        n_public=n_public,
        n_rejected=rejected,
        elapsed_seconds=time.perf_counter() - started,
    )
