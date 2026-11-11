from __future__ import annotations

from typing import Any

from src.transformer.attention import ModelConfig
from src.transformer.swiglu import SwiGLU


class FeedForward(SwiGLU):
    """Feed-forward network wrapper configuring SwiGLU dimensions from ModelConfig."""

    def __init__(self, cfg: dict[str, Any] | ModelConfig) -> None:
        config = ModelConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
        dim = config.n_embd
        hidden = int(config.intermediate_size or max(64, int(8 * dim / 3 / 64) * 64))
        super().__init__(dim=dim, hidden_dim=hidden, dropout=config.dropout)
