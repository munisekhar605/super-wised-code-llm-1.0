from __future__ import annotations

import math
from typing import Any


def compute_lr(step: int, cfg: dict[str, Any]) -> float:
    # Compute linear warmup followed by cosine decay learning rate at a given step.
    base_lr = float(cfg.get("learning_rate", 3e-4))
    warmup_steps = int(cfg.get("warmup_steps", 0))
    max_steps = int(cfg.get("max_steps", 100))
    min_lr_ratio = float(cfg.get("min_lr_ratio", 0.1))
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
