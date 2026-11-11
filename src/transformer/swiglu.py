from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """Step 6: Gated feed-forward network (B, T, C) -> (B, T, H_ff) -> (B, T, C) using SiLU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Elementwise multiply SiLU-gated projection with up-projection: (B, T, C) -> (B, T, H_ff).
        activated = F.silu(self.gate(x)) * self.up(x)
        # Project back to embedding dimension: (B, T, H_ff) -> (B, T, C).
        return self.dropout(self.down(activated))
