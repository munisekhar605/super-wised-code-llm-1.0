from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.utils.reproducibility import capture_rng_state, restore_rng_state


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val: float,
    cfg: dict[str, Any],
    scaler: torch.amp.GradScaler,
) -> None:
    # Atomically write model weights, optimizer state, scaler state, and RNG state to disk.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "best_val": float(best_val),
        "config": dict(cfg),
        "scaler": scaler.state_dict(),
        **capture_rng_state(),
    }
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    # Load checkpoint weights and optionally restore optimizer, GradScaler, and RNG states.
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    restore_rng_state(state, device)
    return state
