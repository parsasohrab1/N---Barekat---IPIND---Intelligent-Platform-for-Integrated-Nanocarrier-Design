"""
Multi-task Transformer + GNN for biological properties (FR-03).

Graph branch: two MPNN layers capture local bond information; then a Transformer encoder
(self-attention with padding mask) models long-range dependencies between atoms (e.g., polar head ↔ hydrophobic
tail); an attention-based readout + global features produce the output.
"""

from typing import Sequence

import torch
from torch import nn

from ..featurization import EXTENDED_DIM, NODE_FEATURE_DIM
from ..nn.layers import AttentionReadout, DenseMessagePassing
from ..nn.predictor import EnsemblePropertyPredictor

BIO_TARGET_COLUMNS = (
    "bio_cytotoxicity_ic50_ug_ml",
    "bio_cytotoxicity_ic50_hepg2_ug_ml",
    "bio_cytotoxicity_ic50_hela_ug_ml",
    "bio_cellular_uptake_efficiency_percent",
    "bio_serum_protein_binding_percent",
    "bio_circulation_half_life_hours",
    "bio_tumor_to_background_ratio",
)


class GraphTransformer(nn.Module):
    def __init__(
        self,
        n_tasks: int,
        hidden_dim: int = 64,
        n_gnn_layers: int = 2,
        n_transformer_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.05,
        global_dim: int = EXTENDED_DIM,
    ):
        super().__init__()
        dims = [NODE_FEATURE_DIM] + [hidden_dim] * n_gnn_layers
        self.gnn = nn.ModuleList(
            DenseMessagePassing(dims[i], dims[i + 1]) for i in range(n_gnn_layers)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers, enable_nested_tensor=False
        )
        self.readout = AttentionReadout(hidden_dim)
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(n_tasks))

    def _encode(self, nodes, adjacency, mask, global_features):
        hidden = nodes
        for layer in self.gnn:
            hidden = layer(hidden, adjacency, mask)
        hidden = self.transformer(hidden, src_key_padding_mask=(mask <= 0))
        hidden = torch.nan_to_num(hidden) * mask.unsqueeze(-1)
        pooled, weights = self.readout(hidden, mask)
        joint = torch.cat([pooled, self.global_mlp(global_features)], dim=-1)
        return self.trunk(joint), weights

    def forward(self, nodes, adjacency, mask, global_features):
        shared, _ = self._encode(nodes, adjacency, mask, global_features)
        return torch.cat([head(shared) for head in self.heads], dim=-1)

    def attention_weights(self, nodes, adjacency, mask, global_features):
        return self._encode(nodes, adjacency, mask, global_features)[1]


EnsemblePropertyPredictor.MODEL_FACTORIES["graph_transformer"] = GraphTransformer


class BiologicalPredictor(EnsemblePropertyPredictor):
    """Biological predictor (Unit 3): toxicity on 3 cell lines + 4 other FR-03 properties."""

    def __init__(
        self,
        target_names: Sequence[str] = BIO_TARGET_COLUMNS,
        n_ensemble: int = 3,
        hidden_dim: int = 64,
        max_atoms: int = 96,
    ):
        super().__init__(
            target_names,
            architecture="graph_transformer",
            n_ensemble=n_ensemble,
            model_kwargs={"hidden_dim": hidden_dim},
            max_atoms=max_atoms,
        )
