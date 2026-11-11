from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TokenEmbedding(nn.Module):
    """Step 1: Convert token IDs (B, T) into dense token embeddings (B, T, C)."""

    def __init__(self, vocab_size: int, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.weight = nn.Parameter(torch.empty(vocab_size, n_embd))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Lookup token vectors and apply embedding dropout: (B, T) -> (B, T, C).
        return self.dropout(F.embedding(input_ids, self.weight))
