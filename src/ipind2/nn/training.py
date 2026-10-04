"""
ابزارهای آموزش مشترک: نرمال‌سازی هدف، Early Stopping و حلقه آموزش رگرسیون چندوظیفه‌ای.

``train_regressor`` هم برای آموزش اولیه واحدهای ۲/۳ و هم برای fine-tuning در حلقه
یادگیری فعال (واحد ۶، FR-06: «Fine-tuning با Early Stopping») استفاده می‌شود.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class TargetScaler:
    """نرمال‌سازی z-score هر وظیفه (هدف) به‌صورت مستقل.

    اهداف واحد ۲ مقیاس‌های کاملاً متفاوتی دارند (اندازه ~۱۰۰ nm، PDI ~۰.۱)؛ بدون
    نرمال‌سازی، loss وظیفه‌های بزرگ‌مقیاس بر آموزش غالب می‌شود.
    """

    mean: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    std: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=np.float32))

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetScaler":
        targets = np.asarray(targets, dtype=np.float32)
        mean = targets.mean(axis=0)
        std = targets.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, targets: np.ndarray) -> np.ndarray:
        return (np.asarray(targets, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        return np.asarray(scaled, dtype=np.float32) * self.std + self.mean

    def state_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state: Dict[str, Sequence[float]]) -> "TargetScaler":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
        )


class EarlyStopping:
    """توقف زودهنگام بر اساس بهبود نیافتن loss اعتبارسنجی."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        if patience < 1:
            raise ValueError("patience باید حداقل ۱ باشد")
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = float("inf")
        self.best_epoch: int = -1
        self._bad_epochs = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, loss: float, epoch: int, model: Optional[nn.Module] = None) -> bool:
        """یک epoch را ثبت می‌کند و ``True`` برمی‌گرداند اگر باید متوقف شد."""
        if loss < self.best - self.min_delta:
            self.best = loss
            self.best_epoch = epoch
            self._bad_epochs = 0
            if model is not None:
                self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self._bad_epochs += 1
        return self._bad_epochs >= self.patience

    def restore(self, model: nn.Module) -> None:
        """بازگرداندن بهترین وزن‌های ثبت‌شده (اگر موجود باشد)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def masked_huber_loss(predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Huber loss که مقادیر NaN در ``targets`` را نادیده می‌گیرد.

    داده آزمایشگاهی معمولاً فقط بخشی از اهداف را اندازه می‌گیرد (مثلاً ۴ از ۱۲ ویژگی)؛
    این ماسک اجازه می‌دهد همان سطرِ جزئی‌برچسب برای fine-tuning استفاده شود.
    """
    mask = ~torch.isnan(targets)
    if not bool(mask.any()):
        return predictions.sum() * 0.0
    diff = torch.where(mask, predictions - torch.nan_to_num(targets), torch.zeros_like(predictions))
    absolute = diff.abs()
    huber = torch.where(absolute <= delta, 0.5 * diff**2, delta * (absolute - 0.5 * delta))
    return (huber * mask).sum() / mask.sum()


@dataclass
class TrainingHistory:
    """تاریخچه آموزش برای گزارش و تست همگرایی."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    stopped_epoch: int = 0
    best_epoch: int = -1


def train_regressor(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, Sequence[int]], torch.Tensor],
    targets: torch.Tensor,
    n_samples: int,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    val_fraction: float = 0.2,
    patience: int = 10,
    seed: int = 0,
    verbose: bool = False,
) -> TrainingHistory:
    """
    حلقه آموزش رگرسیون چندوظیفه‌ای با Huber loss و Early Stopping.

    ``forward_fn(model, batch_indices) -> پیش‌بینی (B, T)`` تا این تابع از نوع ورودی
    (گراف برای واحد ۲، دنباله توکن برای واحد ۳) مستقل بماند.

    Args:
        targets: اهداف نرمال‌شده به شکل (n_samples, n_tasks).
        val_fraction: سهم داده اعتبارسنجی؛ ۰ یعنی بدون early stopping.
    """
    if n_samples < 2:
        raise ValueError("آموزش به حداقل ۲ نمونه نیاز دارد")

    generator = np.random.default_rng(seed)
    indices = generator.permutation(n_samples)
    n_val = int(round(val_fraction * n_samples)) if val_fraction > 0 else 0
    n_val = min(max(n_val, 1 if val_fraction > 0 else 0), n_samples - 1)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = masked_huber_loss
    stopper = EarlyStopping(patience=patience)
    history = TrainingHistory()

    for epoch in range(epochs):
        model.train()
        epoch_losses: List[float] = []
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[start : start + batch_size]
            if len(batch) < 2:
                continue
            optimizer.zero_grad()
            predictions = forward_fn(model, batch)
            loss = loss_fn(predictions, targets[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(train_loss)

        if n_val > 0:
            model.eval()
            with torch.no_grad():
                val_predictions = forward_fn(model, val_idx)
                val_loss = float(loss_fn(val_predictions, targets[val_idx]).item())
            history.val_loss.append(val_loss)
            if verbose:
                print(f"epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")
            if stopper.step(val_loss, epoch, model):
                history.stopped_epoch = epoch
                history.best_epoch = stopper.best_epoch
                stopper.restore(model)
                return history
        elif verbose:
            print(f"epoch {epoch}: train={train_loss:.4f}")

    history.stopped_epoch = epochs - 1
    history.best_epoch = stopper.best_epoch if n_val > 0 else epochs - 1
    if n_val > 0:
        stopper.restore(model)
    return history


def split_indices(
    n_samples: int, test_fraction: float = 0.2, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """تقسیم تصادفی اندیس‌ها به (train, test) — برای اعتبارسنجی NFR-01..NFR-03."""
    indices = np.random.default_rng(seed).permutation(n_samples)
    n_test = max(1, int(round(test_fraction * n_samples)))
    return indices[n_test:], indices[:n_test]
