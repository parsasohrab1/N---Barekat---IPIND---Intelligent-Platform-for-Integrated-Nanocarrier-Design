"""
Active learning loop: receive lab results → buffer → fine-tuning with Early Stopping (FR-06).

New results enter from ``lab_automation.ingest_results`` (the ``experimental_results`` table). The update
threshold is the same SRS range (10–50 new data points). An experiment usually measures only 4 of
12 properties; fine-tuning is done with a masked loss on those same partial rows
and, to prevent "catastrophic forgetting", is mixed with samples from the initial training data (replay)
.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..lab_automation.ingestion import (
    DEFAULT_MAX_RETRAIN_BATCH,
    DEFAULT_MIN_RETRAIN_BATCH,
    should_trigger_retrain,
)
from ..nn.predictor import EnsemblePropertyPredictor

# Mapping of experimental_results table columns to the models' target columns
EXPERIMENTAL_TO_TARGET: Dict[str, str] = {
    "experimental_size_nm": "phys_size_nm",
    "experimental_zeta_potential": "phys_zeta_potential_mV",
    "experimental_loading_efficiency": "phys_drug_loading_efficiency_percent",
    "experimental_cytotoxicity": "bio_cytotoxicity_ic50_ug_ml",
}


@dataclass
class RoundReport:
    """Report of one model update round."""

    round: int
    n_new: int
    n_replay: int
    predictors_updated: List[str]
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)

    def improved(self) -> bool:
        """Whether the mean normalized RMSE on the holdout decreased (if a holdout was given)."""
        if not self.metrics_before or not self.metrics_after:
            return False
        return np.mean(list(self.metrics_after.values())) < np.mean(list(self.metrics_before.values()))


class ActiveLearningLoop:
    """
    Laboratory feedback coordinator for one or more predictors (Units 2 and 3).

    Args:
        predictors: name → predictor (e.g., ``{"physico": ..., "bio": ...}``).
        molecule_lookup: mapping of ``molecule_id`` to SMILES (usually from the ``molecules`` table).
        replay: initial training data (``smiles`` column + all targets) to prevent forgetting.
        min_batch / max_batch: SRS range for update frequency (10–50).
    """

    def __init__(
        self,
        predictors: Mapping[str, EnsemblePropertyPredictor],
        molecule_lookup: Mapping[int, str],
        replay: Optional[pd.DataFrame] = None,
        replay_ratio: int = 3,
        min_batch: int = DEFAULT_MIN_RETRAIN_BATCH,
        max_batch: int = DEFAULT_MAX_RETRAIN_BATCH,
        seed: int = 0,
    ):
        if not predictors:
            raise ValueError("At least one predictor is required")
        if min_batch > max_batch:
            raise ValueError("min_batch must not be greater than max_batch")
        self.predictors = dict(predictors)
        self.molecule_lookup = molecule_lookup
        self.replay = replay
        self.replay_ratio = replay_ratio
        self.min_batch = min_batch
        self.max_batch = max_batch
        self._rng = np.random.default_rng(seed)
        self._buffer: List[Dict] = []
        self.labeled: List[Dict] = []
        self.reports: List[RoundReport] = []

    # ------------------------------------------------------------------
    @property
    def pending(self) -> int:
        """Number of new lab results that have not yet reached the models."""
        return len(self._buffer)

    def ingest(self, results: pd.DataFrame) -> int:
        """
        Add new results (a DataFrame matching ``experimental_results``) to the buffer.

        Rows without a known ``molecule_id`` or without any usable value are rejected.
        Returns: number of accepted rows.
        """
        accepted = 0
        for _, row in results.iterrows():
            molecule_id = row.get("molecule_id")
            if molecule_id is None or pd.isna(molecule_id) or int(molecule_id) not in self.molecule_lookup:
                continue
            record = {"smiles": self.molecule_lookup[int(molecule_id)]}
            has_value = False
            for source, target in EXPERIMENTAL_TO_TARGET.items():
                value = row.get(source)
                if value is not None and not pd.isna(value):
                    record[target] = float(value)
                    has_value = True
            if has_value:
                self._buffer.append(record)
                accepted += 1
        return accepted

    def ready(self) -> bool:
        """Whether the buffer has reached the update threshold (≥ min_batch)."""
        return should_trigger_retrain(self.pending, self.min_batch)

    # ------------------------------------------------------------------
    def _training_frame(self, predictor: EnsemblePropertyPredictor, batch: pd.DataFrame):
        targets = predictor.target_names
        relevant = batch[[c for c in targets if c in batch.columns]].notna().any(axis=1)
        new = batch[relevant].copy()
        for column in targets:
            if column not in new.columns:
                new[column] = np.nan
        frames = [new[["smiles", *targets]]]
        n_replay = 0
        if self.replay is not None and len(new):
            n_replay = min(len(self.replay), self.replay_ratio * len(new))
            sample = self.replay.sample(n=n_replay, random_state=int(self._rng.integers(0, 2**31 - 1)))
            frames.append(sample[["smiles", *targets]])
        return pd.concat(frames, ignore_index=True), len(new), n_replay

    def retrain(
        self,
        holdout: Optional[pd.DataFrame] = None,
        force: bool = False,
        epochs: int = 20,
        learning_rate: float = 3e-4,
    ) -> Optional[RoundReport]:
        """
        Fine-tuning with Early Stopping on the current buffer (if the threshold is reached or ``force``).

        Args:
            holdout: evaluation data (smiles + targets) for reporting normalized RMSE before/after.
        Returns: ``RoundReport`` or ``None`` if it is still too early.
        """
        if not self._buffer or not (force or self.ready()):
            return None

        batch = pd.DataFrame(self._buffer[: self.max_batch * 4])
        n_batch = len(batch)
        before = self._holdout_metrics(holdout) if holdout is not None else {}

        updated: List[str] = []
        total_new = total_replay = 0
        for name, predictor in self.predictors.items():
            frame, n_new, n_replay = self._training_frame(predictor, batch)
            if n_new < 4:
                continue
            predictor.fine_tune(
                frame["smiles"].tolist(),
                frame[predictor.target_names].to_numpy(dtype=np.float32),
                epochs=epochs,
                learning_rate=learning_rate,
                seed=len(self.reports),
            )
            updated.append(name)
            total_new = max(total_new, n_new)
            total_replay = max(total_replay, n_replay)

        self.labeled.extend(self._buffer[:n_batch])
        self._buffer = self._buffer[n_batch:]
        after = self._holdout_metrics(holdout) if holdout is not None else {}
        report = RoundReport(
            round=len(self.reports) + 1,
            n_new=total_new,
            n_replay=total_replay,
            predictors_updated=updated,
            metrics_before=before,
            metrics_after=after,
        )
        self.reports.append(report)
        return report

    def _holdout_metrics(self, holdout: pd.DataFrame) -> Dict[str, float]:
        """Normalized RMSE (by training standard deviation) of each target present in the holdout."""
        metrics: Dict[str, float] = {}
        for predictor in self.predictors.values():
            columns = [c for c in predictor.target_names if c in holdout.columns]
            if not columns:
                continue
            predictions, kept = predictor.member_predictions(holdout["smiles"].tolist())
            mean = predictions.mean(axis=0)
            for column in columns:
                j = predictor.target_names.index(column)
                truth = holdout[column].to_numpy(dtype=float)[kept]
                valid = ~np.isnan(truth)
                if valid.any():
                    error = np.sqrt(np.mean((truth[valid] - mean[valid, j]) ** 2))
                    metrics[column] = float(error / predictor.scaler.std[j])
        return metrics
