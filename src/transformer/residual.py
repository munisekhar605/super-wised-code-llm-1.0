from __future__ import annotations

import torch
from torch import nn


class ResidualConnection(nn.Module):
    """Step 4: Preserve and add previous hidden states (B, T, C) + Dropout(Sublayer(x))."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, residual: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        # Add residual stream to sublayer output: (B, T, C) -> (B, T, C).
        return residual + self.dropout(sublayer_output)
