"""
Shared neural-network components for Units 2 and 3 (implemented in pure torch).

``torch-geometric`` is deliberately not used: nanocarrier graphs are small (≤128 atoms) and
dense message passing on them is both fast and makes installing the platform on CPU/GPU possible
without a dedicated wheel (see docs/ARCHITECTURE.md).
"""

from .layers import AttentionReadout, DenseMessagePassing, masked_softmax
from .training import EarlyStopping, TargetScaler, train_regressor

__all__ = [
    "masked_softmax",
    "DenseMessagePassing",
    "AttentionReadout",
    "TargetScaler",
    "EarlyStopping",
    "train_regressor",
]
