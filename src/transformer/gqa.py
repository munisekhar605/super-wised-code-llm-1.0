from __future__ import annotations

import torch


def repeat_kv_heads(x: torch.Tensor, num_queries_per_kv: int) -> torch.Tensor:
    # Repeat key/value heads across query groups: (B, H_kv, T, D) -> (B, H_q, T, D).
    if num_queries_per_kv == 1:
        return x
    return x.repeat_interleave(num_queries_per_kv, dim=1)
