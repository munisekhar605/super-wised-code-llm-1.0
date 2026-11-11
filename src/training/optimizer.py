from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.AdamW:
    # Configure AdamW optimizer using config learning rate, betas, and weight decay.
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 3e-4)),
        betas=(float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.95))),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
    )
