from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Split last dimension in half and rotate pairs: (..., D) -> (..., D).
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Step 2: Precompute and apply Rotary Position Embeddings (RoPE) inside attention."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"RoPE requires a positive even head dimension, got {head_dim}")
        self.head_dim = head_dim
        self.base = float(base)
        # Compute inverse frequencies: (D // 2,).
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        # Compute position-frequency outer product and cache cos/sin tables: (1, 1, T, D).
        pos = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Extend RoPE cache dynamically if sequence length exceeds cached length.
        if seq_len > self.cos_cached.size(2) or self.cos_cached.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
            self._build_cache(max(seq_len, self.cos_cached.size(2)))
        cos = self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device)
        return cos, sin


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Rotate query (B, H_q, T, D) and key (B, H_kv, T, D) tensors with positional angles.
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
