"""
پیش‌بین چندوظیفه‌ای مبتنی بر ensemble — پایه مشترک واحدهای ۲ و ۳.

هر عضو ensemble با بذر متفاوت آموزش می‌بیند؛ واریانس پیش‌بینی بین اعضا همان «معیار
عدم‌قطعیت» FR-06 است و به‌طور مستقیم به ``ipind2.interpretability.confidence`` و
نمونه‌برداری یادگیری فعال (Query-by-Committee) خورانده می‌شود.

See docs/SRS.md §4.2, §4.3, §4.6, §4.7.
"""

import json
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..featurization import (
    EXTENDED_DIM,
    MolGraph,
    batch_graphs,
    extended_vector,
    smiles_to_graph,
)
from .training import TargetScaler, TrainingHistory, train_regressor


class GraphEncoder:
    """گراف + بردار ویژگی global هر مولکول را یک‌بار می‌سازد و دسته‌ها را برمی‌گرداند."""

    def __init__(self, max_atoms: int = 96):
        self.max_atoms = max_atoms
        self.global_mean = np.zeros(EXTENDED_DIM, dtype=np.float32)
        self.global_std = np.ones(EXTENDED_DIM, dtype=np.float32)
        self.graphs: List[MolGraph] = []
        self.globals: np.ndarray = np.zeros((0, EXTENDED_DIM), dtype=np.float32)

    def fit_globals(self, matrix: np.ndarray) -> None:
        self.global_mean = matrix.mean(axis=0).astype(np.float32)
        std = matrix.std(axis=0)
        self.global_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def prepare(self, smiles_list: Sequence[str]) -> List[int]:
        """آماده‌سازی ورودی‌ها؛ اندیس نمونه‌های معتبر را برمی‌گرداند."""
        graphs, rows, kept = [], [], []
        for i, smiles in enumerate(smiles_list):
            graph = smiles_to_graph(smiles, self.max_atoms)
            vector = extended_vector(smiles)
            if graph is None or vector is None:
                continue
            graphs.append(graph)
            rows.append(vector)
            kept.append(i)
        self.graphs = graphs
        self.globals = (
            np.vstack(rows) if rows else np.zeros((0, EXTENDED_DIM), dtype=np.float32)
        )
        return kept

    def batch(self, indices: Sequence[int]) -> Tuple[torch.Tensor, ...]:
        indices = [int(i) for i in indices]
        nodes, adjacency, mask = batch_graphs([self.graphs[i] for i in indices])
        scaled = (self.globals[indices] - self.global_mean) / self.global_std
        return (
            torch.from_numpy(nodes),
            torch.from_numpy(adjacency),
            torch.from_numpy(mask),
            torch.from_numpy(scaled.astype(np.float32)),
        )

    def state_dict(self) -> Dict:
        return {
            "max_atoms": self.max_atoms,
            "global_mean": self.global_mean.tolist(),
            "global_std": self.global_std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "GraphEncoder":
        encoder = cls(max_atoms=state["max_atoms"])
        encoder.global_mean = np.asarray(state["global_mean"], dtype=np.float32)
        encoder.global_std = np.asarray(state["global_std"], dtype=np.float32)
        return encoder


class EnsemblePropertyPredictor:
    """
    ensemble از مدل‌های گرافی برای پیش‌بینی چندوظیفه‌ای.

    Args:
        target_names: نام اهداف (ستون‌های خروجی).
        architecture: برچسب معماری ثبت‌شده در ``MODEL_FACTORIES``؛ factory با امضای
            ``(n_tasks, **model_kwargs) -> nn.Module`` که forward آن
            ``(nodes, adjacency, mask, globals) -> (B, n_tasks)`` است.
        n_ensemble: تعداد اعضا (≥۲ برای تخمین عدم‌قطعیت).
    """

    MODEL_FACTORIES: Dict[str, Callable[..., nn.Module]] = {}

    def __init__(
        self,
        target_names: Sequence[str],
        architecture: str,
        n_ensemble: int = 3,
        model_kwargs: Optional[Dict] = None,
        max_atoms: int = 96,
    ):
        if architecture not in self.MODEL_FACTORIES:
            raise ValueError(f"معماری ناشناخته: {architecture!r}")
        if n_ensemble < 1:
            raise ValueError("n_ensemble باید حداقل ۱ باشد")
        self.target_names = list(target_names)
        self.architecture = architecture
        self.n_ensemble = n_ensemble
        self.model_kwargs = dict(model_kwargs or {})
        self.encoder = GraphEncoder(max_atoms=max_atoms)
        self.scaler = TargetScaler()
        self.models: List[nn.Module] = []
        self.histories: List[TrainingHistory] = []
        # GraphEncoder حالت per-call دارد (prepare/batch)؛ برای درخواست‌های هم‌زمان API
        # (چند thread) عملیات پیش‌بینی/آموزش سریالی می‌شوند.
        self._lock = threading.RLock()

    def __getstate__(self):
        # قفل threading قابل pickle/deepcopy نیست؛ هنگام بازیابی دوباره ساخته می‌شود
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def _build(self, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        factory = self.MODEL_FACTORIES[self.architecture]
        return factory(len(self.target_names), **self.model_kwargs)

    def _fit_unlocked(
        self,
        smiles: Sequence[str],
        targets: np.ndarray,
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 2e-3,
        patience: int = 8,
        val_fraction: float = 0.15,
        seed: int = 0,
        verbose: bool = False,
    ) -> "EnsemblePropertyPredictor":
        """آموزش ensemble. ``targets`` به شکل (n, n_tasks)."""
        targets = np.asarray(targets, dtype=np.float32)
        if targets.ndim == 1:
            targets = targets[:, None]
        if targets.shape[1] != len(self.target_names):
            raise ValueError("تعداد ستون‌های targets با target_names نمی‌خواند")
        if len(smiles) != len(targets):
            raise ValueError("طول smiles و targets برابر نیست")

        kept = self.encoder.prepare(smiles)
        if len(kept) < 10:
            raise ValueError("تعداد SMILES معتبر برای آموزش کافی نیست (حداقل ۱۰)")
        targets = targets[kept]
        self.encoder.fit_globals(self.encoder.globals)
        self.scaler = TargetScaler.fit(targets)
        scaled = torch.from_numpy(self.scaler.transform(targets))

        self.models, self.histories = [], []
        for member in range(self.n_ensemble):
            model = self._build(seed + member)
            history = train_regressor(
                model,
                forward_fn=lambda m, idx: m(*self.encoder.batch(idx)),
                targets=scaled,
                n_samples=len(kept),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_fraction=val_fraction,
                patience=patience,
                seed=seed + 100 * member,
                verbose=verbose,
            )
            model.eval()
            self.models.append(model)
            self.histories.append(history)
        return self

    def _fine_tune_unlocked(
        self,
        smiles: Sequence[str],
        targets: np.ndarray,
        epochs: int = 20,
        learning_rate: float = 3e-4,
        patience: int = 5,
        seed: int = 0,
    ) -> None:
        """
        Fine-tuning با Early Stopping روی داده جدید آزمایشگاهی (FR-06).

        مقیاس‌گر اهداف و نرمال‌سازی global ثابت می‌ماند تا وزن‌های قبلی معتبر بمانند.
        """
        if not self.models:
            raise RuntimeError("ابتدا fit() را فراخوانی کنید")
        targets = np.asarray(targets, dtype=np.float32)
        if targets.ndim == 1:
            targets = targets[:, None]
        kept = self.encoder.prepare(smiles)
        if len(kept) < 4:
            raise ValueError("برای fine-tuning حداقل ۴ نمونه معتبر لازم است")
        # اهداف جزئی (NaN) مجازند؛ ماسک loss آن‌ها را نادیده می‌گیرد
        scaled = torch.from_numpy(self.scaler.transform(targets[kept]).astype(np.float32))
        for member, model in enumerate(self.models):
            train_regressor(
                model,
                forward_fn=lambda m, idx: m(*self.encoder.batch(idx)),
                targets=scaled,
                n_samples=len(kept),
                epochs=epochs,
                batch_size=16,
                learning_rate=learning_rate,
                val_fraction=0.25,
                patience=patience,
                seed=seed + member,
            )
            model.eval()

    # ------------------------------------------------------------------
    def _member_predictions_unlocked(self, smiles: Sequence[str]) -> Tuple[np.ndarray, List[int]]:
        if not self.models:
            raise RuntimeError("مدل آموزش ندیده؛ ابتدا fit() یا load() را فراخوانی کنید")
        kept = self.encoder.prepare(smiles)
        if not kept:
            return np.zeros((len(self.models), 0, len(self.target_names)), dtype=np.float32), []
        out = np.zeros((len(self.models), len(kept), len(self.target_names)), dtype=np.float32)
        order = list(range(len(kept)))
        with torch.no_grad():
            for m, model in enumerate(self.models):
                model.eval()
                for start in range(0, len(order), 128):
                    idx = order[start : start + 128]
                    pred = model(*self.encoder.batch(idx)).numpy()
                    out[m, start : start + len(idx)] = self.scaler.inverse_transform(pred)
        return out, kept

    def fit(self, *args, **kwargs):
        with self._lock:
            return self._fit_unlocked(*args, **kwargs)

    def fine_tune(self, *args, **kwargs):
        with self._lock:
            return self._fine_tune_unlocked(*args, **kwargs)

    def _member_predictions(self, smiles):
        with self._lock:
            return self._member_predictions_unlocked(smiles)

    def atom_attention(self, smiles):
        with self._lock:
            return self._atom_attention_unlocked(smiles)

    def predict_with_uncertainty(
        self, smiles: Sequence[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
        """میانگین و انحراف‌معیار ensemble؛ همچنین اندیس نمونه‌های معتبر ورودی."""
        members, kept = self._member_predictions(smiles)
        mean = members.mean(axis=0)
        std = members.std(axis=0, ddof=1) if len(self.models) > 1 else np.zeros_like(mean)
        return (
            pd.DataFrame(mean, columns=self.target_names, index=kept),
            pd.DataFrame(std, columns=self.target_names, index=kept),
            kept,
        )

    def predict(self, smiles: Sequence[str]) -> pd.DataFrame:
        """میانگین ensemble (ردیف‌های SMILES نامعتبر حذف می‌شوند؛ index = اندیس ورودی)."""
        return self.predict_with_uncertainty(smiles)[0]

    def member_predictions(self, smiles: Sequence[str]) -> Tuple[np.ndarray, List[int]]:
        """پیش‌بینی خام هر عضو به شکل (n_members, n, n_tasks) برای Query-by-Committee."""
        return self._member_predictions(smiles)

    def _atom_attention_unlocked(self, smiles: str) -> Optional[np.ndarray]:
        """وزن attention هر اتم (میانگین اعضا) — خروجی تفسیرپذیری FR-09."""
        if not self.models:
            raise RuntimeError("مدل آموزش ندیده")
        kept = self.encoder.prepare([smiles])
        if not kept:
            return None
        weights = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                weights.append(model.attention_weights(*self.encoder.batch([0]))[0].numpy())
        n = int(self.encoder.graphs[0].n_atoms)
        return np.mean(weights, axis=0)[:n]

    # ------------------------------------------------------------------
    def evaluate(self, smiles: Sequence[str], targets: np.ndarray) -> pd.DataFrame:
        """RMSE و R² هر هدف روی داده ارزیابی (مبنای NFR-01..NFR-03)."""
        from ..benchmarking.metrics import r_squared, rmse

        targets = np.asarray(targets, dtype=np.float32)
        predictions, kept = self._member_predictions(smiles)
        mean = predictions.mean(axis=0)
        rows = []
        for j, name in enumerate(self.target_names):
            rows.append(
                {
                    "target": name,
                    "rmse": rmse(targets[kept, j], mean[:, j]),
                    "r2": r_squared(targets[kept, j], mean[:, j]),
                }
            )
        return pd.DataFrame(rows).set_index("target")

    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "architecture": self.architecture,
            "target_names": self.target_names,
            "n_ensemble": self.n_ensemble,
            "model_kwargs": self.model_kwargs,
            "scaler": self.scaler.state_dict(),
            "encoder": self.encoder.state_dict(),
        }
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save([m.state_dict() for m in self.models], path / "weights.pt")

    @classmethod
    def load(cls, directory: str) -> "EnsemblePropertyPredictor":
        path = Path(directory)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        # زیرکلاس‌ها (PhysicochemicalPredictor/BiologicalPredictor) امضای __init__ متفاوتی
        # دارند؛ برای بازیابی همیشه از مقداردهی اولیه پایه استفاده می‌کنیم.
        predictor = object.__new__(cls)
        EnsemblePropertyPredictor.__init__(
            predictor,
            meta["target_names"],
            meta["architecture"],
            n_ensemble=meta["n_ensemble"],
            model_kwargs=meta["model_kwargs"],
            max_atoms=meta["encoder"]["max_atoms"],
        )
        predictor.scaler = TargetScaler.from_state_dict(meta["scaler"])
        predictor.encoder = GraphEncoder.from_state_dict(meta["encoder"])
        states = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        predictor.models = []
        for seed, state in enumerate(states):
            model = predictor._build(seed)
            model.load_state_dict(state)
            model.eval()
            predictor.models.append(model)
        return predictor
