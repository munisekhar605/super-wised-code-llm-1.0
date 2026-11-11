from __future__ import annotations

import unittest

import torch

from src.transformer.rope import RotaryEmbedding, apply_rope, rotate_half


class RoPETests(unittest.TestCase):
    def test_rope_even_dim_validation_and_norm_preservation(self) -> None:
        with self.assertRaises(ValueError):
            RotaryEmbedding(head_dim=15)

        rope = RotaryEmbedding(head_dim=16, max_seq_len=8)
        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 2, 16, 16)
        cos, sin = rope(q, seq_len=16)
        self.assertEqual(tuple(cos.shape), (1, 1, 16, 16))
        self.assertEqual(tuple(sin.shape), (1, 1, 16, 16))

        q_rot, k_rot = apply_rope(q, k, cos, sin)
        self.assertEqual(tuple(q_rot.shape), (2, 4, 16, 16))
        self.assertEqual(tuple(k_rot.shape), (2, 2, 16, 16))
        self.assertTrue(torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4))
        self.assertTrue(torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-4))
        self.assertEqual(tuple(rotate_half(q).shape), (2, 4, 16, 16))


if __name__ == "__main__":
    unittest.main()
