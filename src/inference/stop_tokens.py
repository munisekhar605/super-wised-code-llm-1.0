from __future__ import annotations

import torch


def should_stop_on_eos(next_id: torch.Tensor, eos_token_id: int | None) -> bool:
    # Return True when all batch items have generated the EOS token ID.
    if eos_token_id is None:
        return False
    return bool(torch.all(next_id == eos_token_id).item())
