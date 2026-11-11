from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.transformer.attention import ModelConfig
from src.transformer.embeddings import TokenEmbedding
from src.transformer.lm_head import LMHead
from src.transformer.rms_norm import RMSNorm
from src.transformer.transformer_block import TransformerBlock


class SuperWisedCodeLM(nn.Module):
    """Decoder-only Transformer connecting the complete 9-step causal language modeling flow."""

    def __init__(self, cfg: dict[str, Any] | ModelConfig) -> None:
        super().__init__()
        self.model_config = ModelConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
        self.config = cfg if isinstance(cfg, dict) else self.model_config.to_dict()
        self.vocab_size = self.model_config.vocab_size
        self.block_size = self.model_config.block_size
        self.n_layer = self.model_config.n_layer
        self.n_embd = self.model_config.n_embd

        # Step 1: Convert token IDs (B, T) to dense vectors (B, T, C).
        self.tok_embeddings = TokenEmbedding(
            vocab_size=self.vocab_size,
            n_embd=self.n_embd,
            dropout=self.model_config.dropout,
        )
        # Steps 2–7: Stack of decoder Transformer blocks (RoPE + Causal GQA + Residual + RMSNorm + SwiGLU).
        self.layers = nn.ModuleList([TransformerBlock(self.model_config) for _ in range(self.n_layer)])
        # Step 5: Final RMSNorm before vocabulary projection: (B, T, C) -> (B, T, C).
        self.norm = RMSNorm(self.n_embd, eps=self.model_config.rms_norm_eps)
        # Steps 8–9: Language-model head projecting hidden states (B, T, C) to next-token logits (B, T, V).
        self.lm_head = LMHead(self.n_embd, self.vocab_size)
        if self.model_config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # Initialize all parameters from scratch with scaled residual projections.
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * self.n_layer)
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down.weight")):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, TokenEmbedding, LMHead)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected 2D input_ids (B, T), got shape {tuple(input_ids.shape)}")
        seq_len = input_ids.size(1)
        if seq_len == 0:
            raise ValueError("input_ids cannot be empty along sequence dimension")
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size={self.block_size}")

        # Step 1: Convert token IDs into embeddings: (B, T) -> (B, T, C).
        x = self.tok_embeddings(input_ids)
        # Steps 2–7: Pass sequentially through all decoder Transformer blocks: (B, T, C) -> (B, T, C).
        for block in self.layers:
            x = block(x)
        # Step 5: Normalize final hidden representations: (B, T, C) -> (B, T, C).
        x = self.norm(x)
        # Steps 8–9: Project to vocabulary logits (B, T, V) and optional cross-entropy loss.
        return self.lm_head(x, targets)

    def predict_next_token(
        self, input_ids: torch.Tensor, temperature: float = 0.0, top_k: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute next-token logits (B, V) and select next token ID (B, 1) from the final position.
        idx = input_ids[:, -self.block_size :]
        logits, _ = self(idx)
        next_logits = logits[:, -1, :]
        if temperature <= 0.0 or top_k == 1:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            scaled = next_logits / max(temperature, 1e-5)
            if top_k > 0:
                k = min(top_k, scaled.size(-1))
                values, _ = torch.topk(scaled, k)
                scaled = scaled.masked_fill(scaled < values[:, [-1]], -float("inf"))
            probs = F.softmax(scaled, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        return next_id, next_logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        # Autoregressively append predicted next tokens: (B, T) -> (B, T + max_new_tokens).
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            next_id, _ = self.predict_next_token(input_ids, temperature=temperature, top_k=top_k)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if eos_token_id is not None and torch.all(next_id == eos_token_id):
                break
        if was_training:
            self.train()
        return input_ids


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
