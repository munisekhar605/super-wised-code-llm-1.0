from __future__ import annotations

from typing import Any

from src.tokenizer.encode import encode_text


def tokenize_record(tokenizer: Any, text: str) -> list[int]:
    # Tokenize a single filtered code record and append the EOS token ID.
    return encode_text(tokenizer, text, append_eos=True)
