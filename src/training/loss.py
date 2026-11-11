from __future__ import annotations

import torch
from torch.nn import functional as F


def compute_causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Compute token-level cross-entropy loss flattening (B, T, V) and (B, T).
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
