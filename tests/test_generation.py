from __future__ import annotations

import unittest

import torch

from src.inference.generate import generate_tokens
from src.inference.sampling import sample_next_token
from src.inference.stop_tokens import should_stop_on_eos
from src.transformer.attention import ModelConfig
from src.transformer.model import SuperWisedCodeLM


class GenerationTests(unittest.TestCase):
    def test_next_token_prediction_and_autoregressive_generation(self) -> None:
        torch.manual_seed(42)
        cfg = ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=4, n_kv_head=2, n_embd=32)
        model = SuperWisedCodeLM(cfg)
        model.eval()

        prompt = torch.tensor([[2, 5, 9, 12]], dtype=torch.long)
        next_id, next_logits = model.predict_next_token(prompt, temperature=0.0, top_k=1)
        self.assertEqual(tuple(next_id.shape), (1, 1))
        self.assertEqual(tuple(next_logits.shape), (1, 64))
        self.assertEqual(int(next_id.item()), int(sample_next_token(next_logits, temperature=0.0, top_k=1).item()))

        greedy1 = generate_tokens(model, prompt, max_new_tokens=5, temperature=0.0, top_k=1)
        greedy2 = model.generate(prompt, max_new_tokens=5, temperature=0.0, top_k=1)
        self.assertEqual(tuple(greedy1.shape), (1, 9))
        self.assertTrue(torch.equal(greedy1, greedy2))

        self.assertTrue(should_stop_on_eos(torch.tensor([[0]]), eos_token_id=0))
        self.assertFalse(should_stop_on_eos(torch.tensor([[3]]), eos_token_id=0))


if __name__ == "__main__":
    unittest.main()
