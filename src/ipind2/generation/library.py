"""
Virtual structure library generation (FR-01).

``CombinatorialLibrary`` samples or enumerates the ``building_blocks`` templates and
validates and deduplicates the output with RDKit. The FR-01 measures (count, validity rate,
scaffold diversity) are reported in ``LibraryStats`` so they can be measured in CI.

See docs/SRS.md §4.1 (FR-01), NFR-04 (generation time), NFR-08 (scalability).
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
    """A generated structure together with its provenance."""

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
    """Measures of one library generation run — the basis for checking conformance with FR-01/NFR-04."""

    requested: int
    proposed: int
    valid: int
    unique: int
    elapsed_seconds: float
    distinct_skeletons: int = 0
    attempts: int = 0  # number of raw attempts/retrievals (for the conditional generator, including duplicates)

    @property
    def validity_rate(self) -> float:
        """Rate of RDKit-valid structures among the model's proposals (FR-01 target: > 0.95)."""
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
    Sampler/enumerator of the nanocarrier structural space.

    Args:
        scaffold_type: restrict to one scaffold class ('lipid'|'polymer'|'metal');
            ``None`` means all classes.
        seed: random seed for reproducibility (experiment repeatability requirement).
    """

    def __init__(
        self,
        scaffold_type: Optional[str] = None,
        seed: int = 0,
        balance_classes: bool = True,
        template_weighting: str = "uniform",
    ):
        if scaffold_type is not None and scaffold_type not in SCAFFOLD_TYPES:
            raise ValueError(f"Invalid scaffold_type: {scaffold_type!r}")
        self.scaffold_type = scaffold_type
        self.templates: List[StructureTemplate] = templates_for(scaffold_type)
        if not self.templates:
            raise ValueError("No structural template was found for this scaffold type")
        self._rng = random.Random(seed)
        self._template_index = {t.name: i for i, t in enumerate(self.templates)}
        self._combinations = [t.combination_count() for t in self.templates]
        # Two weighting layers:
        #  • ``balance_classes``: equal share for each scaffold class (otherwise the metal class, with ~8% of the space, stays under-represented);
        #  • ``template_weighting``: ``'uniform'`` (default) makes every template within a class equally likely;
        #    ``'space'`` is proportional to template space size. With 'space' the three-tail lipidoid template (~1.2M combinations)
        #    took ≈95% of lipids and quaternary ammoniums/phospholipids were effectively never sampled.
        if template_weighting not in ("uniform", "space"):
            raise ValueError("template_weighting must be uniform or space")
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
        """Theoretical size of the reachable structural space."""
        return theoretical_library_size(self.scaffold_type)

    def _effective_weights(self, used: List[int]) -> List[float]:
        """
        Template weights by the "remaining space" fraction. Small templates (48–368 combinations) with uniform weight
        saturate quickly and return only duplicates; each template's weight decreases as its space fills and
        becomes zero at saturation so that large requests (100k unique structures) do not deadlock.
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
        """Raw proposal (without validation) — for measuring the generative model's validity rate."""
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
        Generate ``n`` valid structures (and by default: unique).

        Args:
            with_descriptors: compute the RDKit descriptor vector for each structure (slower).
            unique: remove duplicates based on canonical SMILES.
            max_attempt_factor: maximum attempts = ``n * max_attempt_factor`` (guard against a
                request larger than the structural space).
            track_skeletons: count distinct scaffolds (FR-01 diversity measure; slower).
        """
        if n <= 0:
            raise ValueError("n must be positive")

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
                    break  # the whole space is saturated; no more unique structures exist
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
        Deterministic enumeration of the structural space — for building the full seed library.

        Unlike ``generate`` which samples randomly, this method traverses the space in order and
        without repetition (suitable for NFR-08: ≥1 million structures).
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
    """Convenience function for one virtual library generation run (FR-01)."""
    library = CombinatorialLibrary(scaffold_type=scaffold_type, seed=seed)
    return library.generate(
        n, with_descriptors=with_descriptors, track_skeletons=track_skeletons
    )


def structures_to_dataframe(structures: Sequence[GeneratedStructure]):
    """Convert the generation output to a ``pandas.DataFrame`` (for database insertion/CSV output)."""
    import pandas as pd

    return pd.DataFrame([s.to_dict() for s in structures])


def descriptor_matrix_of(structures: Iterable[GeneratedStructure]) -> np.ndarray:
    """Descriptor matrix of the structures; computed if descriptors are missing."""
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
