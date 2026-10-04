"""
حلقه یادگیری فعال: دریافت نتایج آزمایشگاهی → بافر → fine-tuning با Early Stopping (FR-06).

نتایج جدید از ``lab_automation.ingest_results`` (جدول ``experimental_results``) وارد
می‌شوند. آستانه به‌روزرسانی همان بازه SRS است (۱۰–۵۰ داده جدید). آزمایش معمولاً فقط ۴ از
۱۲ ویژگی را اندازه می‌گیرد؛ fine-tuning با loss ماسک‌شده روی همان سطرهای جزئی انجام می‌شود
و برای جلوگیری از «فراموشی فاجعه‌بار» با نمونه‌هایی از داده آموزش اولیه (replay) ترکیب
می‌شود.
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

# نگاشت ستون جدول experimental_results به ستون‌های هدف مدل‌ها
EXPERIMENTAL_TO_TARGET: Dict[str, str] = {
    "experimental_size_nm": "phys_size_nm",
    "experimental_zeta_potential": "phys_zeta_potential_mV",
    "experimental_loading_efficiency": "phys_drug_loading_efficiency_percent",
    "experimental_cytotoxicity": "bio_cytotoxicity_ic50_ug_ml",
}


@dataclass
class RoundReport:
    """گزارش یک دور به‌روزرسانی مدل."""

    round: int
    n_new: int
    n_replay: int
    predictors_updated: List[str]
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)

    def improved(self) -> bool:
        """آیا میانگین RMSE نرمال‌شده روی holdout کم شده است (اگر holdout داده شده باشد)."""
        if not self.metrics_before or not self.metrics_after:
            return False
        return np.mean(list(self.metrics_after.values())) < np.mean(list(self.metrics_before.values()))


class ActiveLearningLoop:
    """
    هماهنگ‌کننده بازخورد آزمایشگاهی برای یک یا چند پیش‌بین (واحدهای ۲ و ۳).

    Args:
        predictors: نام → پیش‌بین (مثلاً ``{"physico": ..., "bio": ...}``).
        molecule_lookup: نگاشت ``molecule_id`` به SMILES (معمولاً از جدول ``molecules``).
        replay: داده آموزش اولیه (ستون ``smiles`` + همه اهداف) برای جلوگیری از فراموشی.
        min_batch / max_batch: بازه SRS برای دفعات به‌روزرسانی (۱۰–۵۰).
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
            raise ValueError("حداقل یک پیش‌بین لازم است")
        if min_batch > max_batch:
            raise ValueError("min_batch نباید از max_batch بزرگ‌تر باشد")
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
        """تعداد نتایج آزمایشگاهی جدید که هنوز به مدل‌ها نرسیده‌اند."""
        return len(self._buffer)

    def ingest(self, results: pd.DataFrame) -> int:
        """
        افزودن نتایج جدید (DataFrame منطبق بر ``experimental_results``) به بافر.

        سطرهای بدون ``molecule_id`` شناخته‌شده یا بدون هیچ مقدار قابل‌استفاده رد می‌شوند.
        Returns: تعداد سطر پذیرفته‌شده.
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
        """آیا بافر به آستانه به‌روزرسانی (≥ min_batch) رسیده است."""
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
        Fine-tuning با Early Stopping روی بافر فعلی (اگر به آستانه رسیده یا ``force``).

        Args:
            holdout: داده ارزیابی (smiles + اهداف) برای گزارش RMSE نرمال‌شده قبل/بعد.
        Returns: ``RoundReport`` یا ``None`` اگر هنوز زود است.
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
        """RMSE نرمال‌شده (بر انحراف‌معیار آموزشی) هر هدفِ موجود در holdout."""
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
