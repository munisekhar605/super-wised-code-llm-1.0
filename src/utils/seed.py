from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int = 1337) -> None:
    # Seed Python, NumPy, and PyTorch CPU/CUDA random number generators.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
