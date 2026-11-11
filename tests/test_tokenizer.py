from __future__ import annotations

import unittest

from src.tokenizer.decode import decode_tokens
from src.tokenizer.encode import encode_prompt_tensor, encode_text
from src.tokenizer.tokenizer import (
    REQUIRED_TOKENIZER_ID,
    load_starcoder2_tokenizer,
    validate_tokenizer_source,
)


class TokenizerTests(unittest.TestCase):
    def test_validate_tokenizer_enforces_starcoder2_only(self) -> None:
        self.assertEqual(validate_tokenizer_source(REQUIRED_TOKENIZER_ID), REQUIRED_TOKENIZER_ID)
        with self.assertRaises(ValueError):
            validate_tokenizer_source("gpt2")
        with self.assertRaises(ValueError):
            validate_tokenizer_source("meta-llama/Llama-2-7b-hf")

    def test_encode_and_decode_roundtrip(self) -> None:
        tok = load_starcoder2_tokenizer(REQUIRED_TOKENIZER_ID)
        sample = "export function add(a: number, b: number): number { return a + b; }"
        ids = encode_text(tok, sample, append_eos=True)
        self.assertGreater(len(ids), 5)
        self.assertEqual(ids[-1], tok.eos_token_id)

        decoded = decode_tokens(tok, ids[:-1], skip_special_tokens=True)
        self.assertEqual(decoded, sample)

        tensor_ids = encode_prompt_tensor(tok, sample, device="cpu")
        self.assertEqual(tensor_ids.ndim, 2)
        self.assertEqual(tensor_ids.size(0), 1)


if __name__ == "__main__":
    unittest.main()
