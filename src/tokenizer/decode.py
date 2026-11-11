from __future__ import annotations

from typing import Any

import torch


def decode_tokens(
    tokenizer: Any, token_ids: torch.Tensor | list[int], skip_special_tokens: bool = True
) -> str:
    # Decode a 1D tensor or list of token IDs back into code without BPE space corruption.
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim > 1:
            token_ids = token_ids[0]
        token_ids = token_ids.detach().cpu().tolist()
    return str(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    )
