"""
مدل‌های مولد شرطی روی فضای ویژگی مولکولی: Conditional VAE و Conditional GAN (FR-01).

هر دو مدل بردار ویژگی استانداردشده مولکول (توصیف‌گر + گروه عاملی) را به‌شرط «ویژگی
هدف + نوع اسکلت» مدل می‌کنند. خروجی مدل یک *بردار ویژگی هدف* است؛ تبدیل آن به SMILES
معتبر با بازیابی نزدیک‌ترین همسایه در فضای قالب‌های معتبر انجام می‌شود
(``conditional.py``). این طراحی عمداً نرخ اعتبار ۱۰۰٪ می‌دهد و تنها بخش «چه چیزی
تولید شود» را یاد می‌گیرد.
"""

import torch
from torch import nn


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    layers = []
    dim = in_dim
    for _ in range(depth):
        layers += [nn.Linear(dim, hidden), nn.ReLU()]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class ConditionalVAE(nn.Module):
    """VAE شرطی: q(z|x,c) و p(x|z,c) با نمونه‌برداری reparameterized."""

    def __init__(self, feature_dim: int, cond_dim: int, latent_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.encoder = _mlp(feature_dim + cond_dim, hidden, 2 * latent_dim)
        self.decoder = _mlp(latent_dim + cond_dim, hidden, feature_dim)

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        mu, log_var = self.encoder(torch.cat([x, c], dim=-1)).chunk(2, dim=-1)
        return mu, log_var.clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, log_var = self.encode(x, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return self.decode(z, c), mu, log_var

    @staticmethod
    def loss(recon, x, mu, log_var, beta: float = 0.5) -> torch.Tensor:
        reconstruction = ((recon - x) ** 2).sum(dim=-1).mean()
        kl = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=-1)).mean()
        return reconstruction + beta * kl

    @torch.no_grad()
    def sample(self, c: torch.Tensor) -> torch.Tensor:
        z = torch.randn(c.shape[0], self.latent_dim)
        return self.decode(z, c)


class ConditionalGAN(nn.Module):
    """GAN شرطی (non-saturating) با مولد G(z,c) و متمایزکننده D(x,c)."""

    def __init__(self, feature_dim: int, cond_dim: int, latent_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.generator = _mlp(latent_dim + cond_dim, hidden, feature_dim)
        self.discriminator = _mlp(feature_dim + cond_dim, hidden, 1)

    def generate(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.generator(torch.cat([z, c], dim=-1))

    def discriminate(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.discriminator(torch.cat([x, c], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, c: torch.Tensor) -> torch.Tensor:
        z = torch.randn(c.shape[0], self.latent_dim)
        return self.generate(z, c)
