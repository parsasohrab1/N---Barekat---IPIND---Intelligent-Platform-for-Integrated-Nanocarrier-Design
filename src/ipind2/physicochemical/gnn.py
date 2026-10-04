"""
GNN چندوظیفه‌ای (MPNN + Attention) برای پیش‌بینی هم‌زمان ۷ ویژگی فیزیکوشیمیایی (FR-02).

معماری: سه لایه message passing → readout با attention اتمی → الحاق با بردار ویژگی global
(توصیف‌گر + گروه عاملی) → trunk مشترک → سر خطی هر وظیفه. تجمیع global همان الگوی
D-MPNN + RDKit-features است که روی داده‌های کوچک پایدارتر از GNN خالص است.
"""

from typing import Sequence

import torch
from torch import nn

from ..featurization import EXTENDED_DIM, NODE_FEATURE_DIM
from ..nn.layers import AttentionReadout, DenseMessagePassing
from ..nn.predictor import EnsemblePropertyPredictor

PHYSICO_TARGET_COLUMNS = (
    "phys_size_nm",
    "phys_zeta_potential_mV",
    "phys_pdi",
    "phys_colloidal_stability_hours",
    "phys_drug_loading_efficiency_percent",
    "phys_drug_loading_content_percent",
    "phys_release_rate_constant",
)


class MultiTaskGNN(nn.Module):
    def __init__(
        self,
        n_tasks: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.05,
        global_dim: int = EXTENDED_DIM,
    ):
        super().__init__()
        dims = [NODE_FEATURE_DIM] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(
            DenseMessagePassing(dims[i], dims[i + 1]) for i in range(n_layers)
        )
        self.readout = AttentionReadout(hidden_dim)
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(n_tasks))

    def _encode(self, nodes, adjacency, mask, global_features):
        hidden = nodes
        for layer in self.layers:
            hidden = layer(hidden, adjacency, mask)
        pooled, weights = self.readout(hidden, mask)
        joint = torch.cat([pooled, self.global_mlp(global_features)], dim=-1)
        return self.trunk(joint), weights

    def forward(self, nodes, adjacency, mask, global_features):
        shared, _ = self._encode(nodes, adjacency, mask, global_features)
        return torch.cat([head(shared) for head in self.heads], dim=-1)

    def attention_weights(self, nodes, adjacency, mask, global_features):
        return self._encode(nodes, adjacency, mask, global_features)[1]


EnsemblePropertyPredictor.MODEL_FACTORIES["gnn"] = MultiTaskGNN


class PhysicochemicalPredictor(EnsemblePropertyPredictor):
    """پیش‌بین فیزیکوشیمیایی (واحد ۲)؛ ۷ هدف FR-02 به‌طور پیش‌فرض."""

    def __init__(
        self,
        target_names: Sequence[str] = PHYSICO_TARGET_COLUMNS,
        n_ensemble: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 3,
        max_atoms: int = 96,
    ):
        super().__init__(
            target_names,
            architecture="gnn",
            n_ensemble=n_ensemble,
            model_kwargs={"hidden_dim": hidden_dim, "n_layers": n_layers},
            max_atoms=max_atoms,
        )
