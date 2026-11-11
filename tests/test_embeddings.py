from __future__ import annotations

import unittest

import torch

from src.transformer.embeddings import TokenEmbedding


class EmbeddingTests(unittest.TestCase):
    def test_token_embedding_shape_and_gradients(self) -> None:
        emb = TokenEmbedding(vocab_size=128, n_embd=64, dropout=0.0)
        tokens = torch.randint(0, 128, (2, 12))
        out = emb(tokens)
        self.assertEqual(tuple(out.shape), (2, 12, 64))
        self.assertEqual(tuple(emb.weight.shape), (128, 64))

        out.sum().backward()
        self.assertIsNotNone(emb.weight.grad)
        self.assertGreater(emb.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
