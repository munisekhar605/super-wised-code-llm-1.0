from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.data.dataset import get_batch


def evaluate_validation_loss(
    model: nn.Module,
    val_tokens: np.ndarray | np.memmap,
    cfg: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    batches: int,
) -> float:
    # Evaluate mean cross-entropy loss across validation batches without gradient tracking.
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(max(1, batches)):
            x, y = get_batch(val_tokens, int(cfg["batch_size"]), int(cfg["block_size"]), device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))
