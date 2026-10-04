"""لایه‌های پایه: message passing متراکم و readout مبتنی بر attention (برای FR-02/FR-09)."""

import torch
from torch import nn


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    softmax با نادیده‌گرفتن موقعیت‌های padding.

    Args:
        scores: تنسور امتیازها.
        mask: تنسور هم‌شکل با ``scores`` که ۱=معتبر و ۰=padding.
    """
    very_negative = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(mask <= 0, very_negative)
    weights = torch.softmax(masked, dim=dim)
    # اگر یک سطر کاملاً padding باشد softmax مقدار یکنواخت می‌دهد؛ صفر می‌کنیم.
    return weights * (mask > 0).to(weights.dtype)


class DenseMessagePassing(nn.Module):
    """
    یک لایه MPNN روی ماتریس مجاورت متراکم.

    h' = ReLU(LayerNorm(W_self · h + W_neigh · (Â h)))  با Â مجاورت نرمال‌شده سطری.
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
    تجمیع اتم‌ها به یک بردار مولکولی با وزن‌های attention.

    وزن‌ها علاوه بر pooling، خروجی تفسیرپذیری واحد ۷ (FR-09) هستند: سهم هر اتم در
    پیش‌بینی. برای همین ``forward`` وزن‌ها را هم برمی‌گرداند.
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
