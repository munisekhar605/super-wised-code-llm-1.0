from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    # Capture CPU ByteTensor RNG states for Python, NumPy, PyTorch, and CUDA.
    cuda_rng = (
        [s.cpu().to(dtype=torch.uint8) for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    return {
        "torch_rng": torch.get_rng_state().cpu().to(dtype=torch.uint8),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": cuda_rng,
    }


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    # Restore RNG states safely onto CPU ByteTensors regardless of checkpoint map_location.
    if state.get("torch_rng") is not None:
        torch.set_rng_state(state["torch_rng"].cpu().to(dtype=torch.uint8))
    if state.get("numpy_rng") is not None:
        np.random.set_state(state["numpy_rng"])
    if state.get("python_rng") is not None:
        random.setstate(state["python_rng"])
    if device.type == "cuda" and state.get("cuda_rng") is not None:
        cuda_states = [s.cpu().to(dtype=torch.uint8) for s in state["cuda_rng"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
