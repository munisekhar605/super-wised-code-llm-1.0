from __future__ import annotations

import torch


def build_causal_mask(seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    # Build lower-triangular boolean causal mask (1, 1, T, T) where True allows attention.
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))[None, None, :, :]


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    # Mask future token positions (j > i) with -inf in attention score matrix (B, H, T, T).
    seq_len = scores.size(-1)
    mask = build_causal_mask(seq_len, device=scores.device)
    return scores.masked_fill(~mask, -float("inf"))
