from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * scale * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :].to(x.dtype), emb.sin()[None, None, :, :].to(x.dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim, heads = cfg["n_embd"], cfg["n_head"]
        kv_heads = cfg.get("n_kv_head", heads)
        if dim % heads or heads % kv_heads:
            raise ValueError("n_embd must divide by n_head, and n_head by n_kv_head")
        self.heads, self.kv_heads = heads, kv_heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = cfg.get("dropout", 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(q, t)
        q, k = apply_rope(q, k, cos, sin)
        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim = cfg["n_embd"]
        hidden = cfg.get("intermediate_size", max(64, int(8 * dim / 3 / 64) * 64))
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden, cfg.get("dropout", 0.0))
        self.resid_dropout = nn.Dropout(cfg.get("dropout", 0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.norm1(x)))
        return x + self.mlp(self.norm2(x))


class SuperWisedCodeLM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.config = dict(cfg)
        self.vocab_size = int(cfg["vocab_size"])
        self.block_size = int(cfg["block_size"])
        self.tok_embeddings = nn.Embedding(self.vocab_size, cfg["n_embd"])
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.norm = RMSNorm(cfg["n_embd"])
        self.lm_head = nn.Linear(cfg["n_embd"], self.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layer"]))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if input_ids.size(1) > self.block_size:
            raise ValueError(f"Sequence length {input_ids.size(1)} exceeds block_size={self.block_size}")
        x = self.dropout(self.tok_embeddings(input_ids))
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50):
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
