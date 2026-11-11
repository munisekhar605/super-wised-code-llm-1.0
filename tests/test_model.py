from __future__ import annotations

import unittest

import torch

from src.transformer.attention import ModelConfig
from src.transformer.lm_head import LMHead
from src.transformer.model import SuperWisedCodeLM, count_parameters


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.cfg = ModelConfig(
            vocab_size=128,
            block_size=32,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=64,
            intermediate_size=128,
            dropout=0.0,
            tie_word_embeddings=True,
        )

    def test_lm_head_and_complete_forward_pass_gradient_flow(self) -> None:
        head = LMHead(n_embd=64, vocab_size=128)
        h = torch.randn(2, 8, 64)
        targets = torch.randint(0, 128, (2, 8))
        logits_h, loss_h = head(h, targets)
        self.assertEqual(tuple(logits_h.shape), (2, 8, 128))
        self.assertIsNotNone(loss_h)

        model = SuperWisedCodeLM(self.cfg)
        self.assertIs(model.lm_head.weight, model.tok_embeddings.weight)
        self.assertGreater(count_parameters(model), 0)

        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 16, 128))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), f"Non-finite gradient for {name}")
            self.assertGreater(param.grad.abs().sum().item(), 0.0, f"Zero gradient for {name}")

    def test_causal_masking_and_context_bounds(self) -> None:
        model = SuperWisedCodeLM(self.cfg)
        model.eval()
        x1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        x2 = torch.tensor([[1, 2, 3, 4, 99, 100, 101, 102]], dtype=torch.long)
        with torch.no_grad():
            logits1, _ = model(x1)
            logits2, _ = model(x2)
        self.assertTrue(torch.allclose(logits1[:, :4, :], logits2[:, :4, :], atol=1e-5))

        with self.assertRaises(ValueError):
            model(torch.ones((1, 33), dtype=torch.long))
        with self.assertRaises(ValueError):
            model(torch.ones((1, 0), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
