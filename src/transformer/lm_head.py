from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LMHead(nn.Module):
    """Steps 8 & 9: Project normalized hidden states (B, T, C) to next-token vocabulary logits (B, T, V)."""

    def __init__(self, n_embd: int, vocab_size: int) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.weight = nn.Parameter(torch.empty(vocab_size, n_embd))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(
        self, hidden_states: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Step 8 & 9: Project hidden states to next-token vocabulary logits: (B, T, C) -> (B, T, V).
        logits = F.linear(hidden_states, self.weight)
        loss = None
        if targets is not None:
            # Compute autoregressive next-token cross-entropy loss across flattened (B * T, V).
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
