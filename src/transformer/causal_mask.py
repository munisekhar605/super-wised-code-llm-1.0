from __future__ import annotations

import torch


def build_causal_mask(
    seq_len: int, key_len: int | None = None, device: torch.device | str = "cpu"
) -> torch.Tensor:
    # Build causal boolean mask (1, 1, T_q, T_k) where True allows attention.
    k_len = seq_len if key_len is None else key_len
    diagonal = max(0, k_len - seq_len)
    return torch.tril(torch.ones((seq_len, k_len), dtype=torch.bool, device=device), diagonal=diagonal)[None, None, :, :]


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    # Mask future token positions (j > i) with -inf in attention score matrix (B, H, T_q, T_k).
    t_q, t_k = scores.size(-2), scores.size(-1)
    mask = build_causal_mask(t_q, key_len=t_k, device=scores.device)
    return scores.masked_fill(~mask, -float("inf"))
