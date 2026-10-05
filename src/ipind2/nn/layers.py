"""Base layers: dense message passing and attention-based readout (for FR-02/FR-09)."""

import torch
from torch import nn


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    softmax ignoring padding positions.

    Args:
        scores: tensor of scores.
        mask: tensor of the same shape as ``scores`` where 1=valid and 0=padding.
    """
    very_negative = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(mask <= 0, very_negative)
    weights = torch.softmax(masked, dim=dim)
    # If a row is entirely padding, softmax gives a uniform value; we zero it.
    return weights * (mask > 0).to(weights.dtype)


class DenseMessagePassing(nn.Module):
    """
    An MPNN layer on a dense adjacency matrix.

    h' = ReLU(LayerNorm(W_self · h + W_neigh · (Â h)))  with Â the row-normalized adjacency.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_features: torch.Tensor,  # (B, N, F)
        adjacency: torch.Tensor,  # (B, N, N)
        mask: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)
        messages = torch.matmul(adjacency / degree, node_features)
        hidden = self.self_proj(node_features) + self.neigh_proj(messages)
        hidden = torch.relu(self.norm(hidden))
        return hidden * mask.unsqueeze(-1)


class AttentionReadout(nn.Module):
    """
    Aggregation of atoms into a molecular vector with attention weights.

    The weights, in addition to pooling, are the interpretability output of Unit 7 (FR-09): the contribution of each atom to the
    prediction. For this reason ``forward`` also returns the weights.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor, mask: torch.Tensor):
        scores = self.score(node_features).squeeze(-1)  # (B, N)
        weights = masked_softmax(scores, mask, dim=-1)
        pooled = torch.einsum("bn,bnf->bf", weights, node_features)
        return pooled, weights
