from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    cfg: dict[str, Any] | None = None,
    *,
    lr: float | None = None,
    weight_decay: float | None = None,
    beta1: float | None = None,
    beta2: float | None = None,
) -> torch.optim.AdamW:
    # Configure AdamW optimizer with weight decay on 2D weights and 0.0 on 1D normalization/biases.
    c = cfg or {}
    learning_rate = lr if lr is not None else float(c.get("learning_rate", 3e-4))
    decay = weight_decay if weight_decay is not None else float(c.get("weight_decay", 0.1))
    b1 = beta1 if beta1 is not None else float(c.get("beta1", 0.9))
    b2 = beta2 if beta2 is not None else float(c.get("beta2", 0.95))

    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(b1, b2),
    )
