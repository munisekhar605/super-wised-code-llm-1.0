from __future__ import annotations

import unittest

import torch

from src.transformer.attention import ModelConfig
from src.transformer.feed_forward import FeedForward
from src.transformer.residual import ResidualConnection
from src.transformer.rms_norm import RMSNorm
from src.transformer.swiglu import SwiGLU
from src.transformer.transformer_block import TransformerBlock


class TransformerBlockTests(unittest.TestCase):
    def test_rmsnorm_swiglu_residual_and_block(self) -> None:
        norm = RMSNorm(dim=64, eps=1e-6)
        x = torch.randn(2, 10, 64) * 3.0
        normed = norm(x)
        rms = normed.pow(2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-4))

        mlp = SwiGLU(dim=64, hidden_dim=128, dropout=0.0)
        self.assertEqual(tuple(mlp(normed).shape), (2, 10, 64))

        cfg = ModelConfig(
            vocab_size=128,
            block_size=32,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=64,
            intermediate_size=128,
        )
        ffn = FeedForward(cfg)
        self.assertEqual(tuple(ffn(normed).shape), (2, 10, 64))

        resid = ResidualConnection(dropout=0.0)
        self.assertTrue(torch.allclose(resid(x, normed), x + normed))

        block = TransformerBlock(cfg)
        inp = torch.randn(2, 10, 64, requires_grad=True)
        out = block(inp)
        self.assertEqual(tuple(out.shape), (2, 10, 64))
        out.sum().backward()
        self.assertIsNotNone(inp.grad)
        self.assertTrue(torch.isfinite(inp.grad).all().item())


if __name__ == "__main__":
    unittest.main()
