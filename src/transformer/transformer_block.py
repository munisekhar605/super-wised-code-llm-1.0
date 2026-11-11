from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.transformer.attention import CausalGroupedQueryAttention, ModelConfig
from src.transformer.residual import ResidualConnection
from src.transformer.rms_norm import RMSNorm
from src.transformer.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Step 7: Decoder block combining RMSNorm, Causal GQA Attention, SwiGLU, and Residual connections."""

    def __init__(self, cfg: dict[str, Any] | ModelConfig) -> None:
        super().__init__()
        config = ModelConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
        dim = config.n_embd
        hidden = int(config.intermediate_size or max(64, int(8 * dim / 3 / 64) * 64))

        self.norm1 = RMSNorm(dim, eps=config.rms_norm_eps)
        self.attn = CausalGroupedQueryAttention(config)
        self.resid_attn = ResidualConnection(dropout=config.dropout)
        self.norm2 = RMSNorm(dim, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(dim, hidden, dropout=config.dropout)
        self.resid_mlp = ResidualConnection(dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm causal GQA attention with residual connection: (B, T, C) -> (B, T, C).
        x = self.resid_attn(x, self.attn(self.norm1(x)))
        # Pre-norm SwiGLU feed-forward network with residual connection: (B, T, C) -> (B, T, C).
        return self.resid_mlp(x, self.mlp(self.norm2(x)))


Block = TransformerBlock
