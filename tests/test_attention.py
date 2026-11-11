from __future__ import annotations

import unittest

import torch

from src.transformer.attention import CausalGroupedQueryAttention, ModelConfig
from src.transformer.causal_mask import apply_causal_mask, build_causal_mask
from src.transformer.gqa import repeat_kv_heads


class AttentionTests(unittest.TestCase):
    def test_gqa_projections_and_head_repetition(self) -> None:
        cfg = ModelConfig(vocab_size=128, block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=64)
        attn = CausalGroupedQueryAttention(cfg)
        self.assertEqual(attn.q_proj.out_features, 64)
        self.assertEqual(attn.k_proj.out_features, 32)
        self.assertEqual(attn.v_proj.out_features, 32)

        kv = torch.randn(2, 2, 10, 16)
        repeated = repeat_kv_heads(kv, num_queries_per_kv=2)
        self.assertEqual(tuple(repeated.shape), (2, 4, 10, 16))

    def test_causal_mask_and_attention_causality(self) -> None:
        mask = build_causal_mask(4)
        self.assertEqual(tuple(mask.shape), (1, 1, 4, 4))
        scores = torch.zeros(1, 2, 4, 4)
        masked = apply_causal_mask(scores)
        self.assertTrue(torch.isinf(masked[0, 0, 0, 1]).item())
        self.assertEqual(float(masked[0, 0, 2, 1].item()), 0.0)

        cfg = ModelConfig(vocab_size=128, block_size=32, n_layer=1, n_head=4, n_kv_head=2, n_embd=64)
        attn = CausalGroupedQueryAttention(cfg)
        attn.eval()
        x1 = torch.randn(1, 8, 64)
        x2 = x1.clone()
        x2[:, 4:, :] = torch.randn(1, 4, 64)
        with torch.no_grad():
            y1 = attn(x1)
            y2 = attn(x2)
        self.assertTrue(torch.allclose(y1[:, :4, :], y2[:, :4, :], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
