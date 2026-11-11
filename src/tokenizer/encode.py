from __future__ import annotations

from typing import Any

import torch


def encode_text(tokenizer: Any, text: str, append_eos: bool = True) -> list[int]:
    # Encode source text into token IDs and optionally append the EOS token ID.
    ids: list[int] = list(tokenizer.encode(text, add_special_tokens=False))
    if ids and append_eos and tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    return ids


def encode_prompt_tensor(tokenizer: Any, prompt: str, device: torch.device | str = "cpu") -> torch.Tensor:
    # Encode a prompt string into a 2D batch tensor (1, T) on the target device.
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("Prompt produced zero token IDs.")
    return torch.tensor([ids], dtype=torch.long, device=device)
