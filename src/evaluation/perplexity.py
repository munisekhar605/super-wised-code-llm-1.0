from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from src.data.dataset import get_batch


def compute_perplexity(loss: float) -> float:
    # Convert cross-entropy loss (in nats) into clipped perplexity exp(loss).
    return float(math.exp(min(float(loss), 20.0)))


def evaluate_perplexity(
    model: nn.Module,
    tokens: np.ndarray | np.memmap,
    cfg: dict[str, Any],
    device: torch.device,
    batches: int = 20,
) -> dict[str, float]:
    # Compute mean validation loss, standard deviation, and perplexity over sampled batches.
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(max(1, batches)):
            x, y = get_batch(tokens, int(cfg["batch_size"]), int(cfg["block_size"]), device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))
    return {
        "validation_loss_mean": mean_loss,
        "validation_loss_std": std_loss,
        "validation_perplexity": compute_perplexity(mean_loss),
    }
