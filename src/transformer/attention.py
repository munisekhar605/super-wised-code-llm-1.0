from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.tokenizer.tokenizer import REQUIRED_TOKENIZER_ID
from src.transformer.gqa import repeat_kv_heads
from src.transformer.rope import RotaryEmbedding, apply_rope


@dataclass
class ModelConfig:
    vocab_size: int = 49152
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int | None = None
    n_embd: int = 256
    intermediate_size: int | None = None
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    model_name: str = "super-wised-code-llm-1.0"
    tokenizer_name: str = REQUIRED_TOKENIZER_ID

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.intermediate_size is None:
            self.intermediate_size = max(64, int(8 * self.n_embd / 3 / 64) * 64)
        self.validate()

    def validate(self) -> None:
        if self.vocab_size <= 0 or self.block_size <= 0 or self.n_layer <= 0:
            raise ValueError("vocab_size, block_size, and n_layer must be positive integers")
        if self.n_embd <= 0 or self.n_head <= 0 or (self.n_kv_head or 0) <= 0:
            raise ValueError("n_embd, n_head, and n_kv_head must be positive integers")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_head % int(self.n_kv_head) != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("Head dimension (n_embd // n_head) must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | ModelConfig) -> ModelConfig:
        if isinstance(cfg, cls):
            return cfg
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in valid_keys})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalGroupedQueryAttention(nn.Module):
    """Step 3: Causal Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE)."""

    def __init__(self, cfg: dict[str, Any] | ModelConfig) -> None:
        super().__init__()
        config = ModelConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
        dim = config.n_embd
        heads = config.n_head
        kv_heads = int(config.n_kv_head or heads)

        self.heads = heads
        self.kv_heads = kv_heads
        self.num_queries_per_kv = heads // kv_heads
        self.head_dim = config.head_dim

        # Linear projections for queries (H_q * D), keys (H_kv * D), values (H_kv * D), and output (C).
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.block_size,
            base=config.rope_theta,
        )
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        # Project and reshape into multi-head tensors: (B, T, C) -> (B, H, T, D).
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)

        # Step 2: Add positional information to queries and keys using RoPE.
        cos, sin = self.rope(q, t)
        q, k = apply_rope(q, k, cos, sin)

        # Expand KV heads across query head groups: (B, H_kv, T, D) -> (B, H_q, T, D).
        k = repeat_kv_heads(k, self.num_queries_per_kv)
        v = repeat_kv_heads(v, self.num_queries_per_kv)

        # Step 3: Compute causal scaled dot-product attention: (B, H_q, T, D) -> (B, H_q, T, D).
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Merge heads and project back to hidden dimension: (B, H_q, T, D) -> (B, T, C).
        merged = attn_out.transpose(1, 2).contiguous().view(b, t, c)
        return self.o_proj(merged)


Attention = CausalGroupedQueryAttention
