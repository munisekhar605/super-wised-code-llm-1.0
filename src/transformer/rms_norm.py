from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Step 5: Normalize hidden states (B, T, C) using Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute reciprocal RMS in FP32 for numerical stability: (B, T, C) -> (B, T, 1).
        rms_scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Scale and apply learnable gain: (B, T, C) -> (B, T, C).
        return (x.float() * rms_scale).to(x.dtype) * self.weight
