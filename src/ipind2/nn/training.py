"""
Shared training tools: target normalization, Early Stopping and the multi-task regression training loop.

``train_regressor`` is used both for the initial training of Units 2/3 and for fine-tuning in the
active learning loop (Unit 6, FR-06: "Fine-tuning with Early Stopping").
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class TargetScaler:
    """z-score normalization of each task (target) independently.

    Unit 2 targets have completely different scales (size ~100 nm, PDI ~0.1); without
    normalization, the loss of large-scale tasks dominates training.
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
    """Early stopping based on no improvement of the validation loss."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = float("inf")
        self.best_epoch: int = -1
        self._bad_epochs = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, loss: float, epoch: int, model: Optional[nn.Module] = None) -> bool:
        """Records one epoch and returns ``True`` if training should stop."""
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
        """Restore the best recorded weights (if available)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def masked_huber_loss(predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Huber loss that ignores NaN values in ``targets``.

    Lab data usually measures only some of the targets (e.g., 4 of 12 properties);
    this mask allows that same partially-labeled row to be used for fine-tuning.
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
    """Training history for reporting and convergence tests."""

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
    Multi-task regression training loop with Huber loss and Early Stopping.

    ``forward_fn(model, batch_indices) -> prediction (B, T)`` so that this function stays independent of the input type
    (graph for Unit 2, token sequence for Unit 3).

    Args:
        targets: normalized targets with shape (n_samples, n_tasks).
        val_fraction: share of validation data; 0 means no early stopping.
    """
    if n_samples < 2:
        raise ValueError("Training requires at least 2 samples")

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
    """Random split of indices into (train, test) — for NFR-01..NFR-03 validation."""
    indices = np.random.default_rng(seed).permutation(n_samples)
    n_test = max(1, int(round(test_fraction * n_samples)))
    return indices[n_test:], indices[:n_test]
