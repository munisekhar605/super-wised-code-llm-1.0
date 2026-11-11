from __future__ import annotations

import torch
from torch.nn import functional as F


def sample_next_token(
    next_logits: torch.Tensor, temperature: float = 0.8, top_k: int = 50
) -> torch.Tensor:
    # Select next token ID (B, 1) from vocabulary logits (B, V) using greedy or top-k sampling.
    if temperature <= 0.0 or top_k == 1:
        return torch.argmax(next_logits, dim=-1, keepdim=True)
    scaled = next_logits / max(temperature, 1e-5)
    if top_k > 0:
        k = min(top_k, scaled.size(-1))
        values, _ = torch.topk(scaled, k)
        scaled = scaled.masked_fill(scaled < values[:, [-1]], -float("inf"))
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)
