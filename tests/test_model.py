import unittest
import torch
from src.model import SuperWisedCodeLM, count_parameters

class ModelTests(unittest.TestCase):
    def test_forward_and_loss(self):
        cfg = {"vocab_size": 128, "n_layer": 2, "n_head": 4, "n_kv_head": 2, "n_embd": 64,
               "intermediate_size": 128, "block_size": 32, "dropout": 0.0}
        model = SuperWisedCodeLM(cfg)
        x = torch.randint(0, 128, (2, 16))
        logits, loss = model(x, x)
        self.assertEqual(tuple(logits.shape), (2, 16, 128))
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(count_parameters(model), 0)

    def test_context_limit(self):
        cfg = {"vocab_size": 32, "n_layer": 1, "n_head": 2, "n_embd": 16, "block_size": 8}
        model = SuperWisedCodeLM(cfg)
        with self.assertRaises(ValueError):
            model(torch.ones((1, 9), dtype=torch.long))

if __name__ == "__main__": unittest.main()
