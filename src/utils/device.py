from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    # Select CUDA device when available, otherwise fall back to CPU.
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_amp_context(device: torch.device) -> tuple[bool, torch.dtype]:
    # Determine whether automatic mixed precision is supported and select bfloat16 or float16.
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    return use_amp, amp_dtype
