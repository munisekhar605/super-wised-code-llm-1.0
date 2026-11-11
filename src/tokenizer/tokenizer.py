from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

REQUIRED_TOKENIZER_ID = "bigcode/starcoder2-15b"


def validate_tokenizer_source(tokenizer_name_or_path: str | Path) -> str:
    # Enforce using only the bigcode/starcoder2-15b tokenizer (or its saved directory).
    target = str(tokenizer_name_or_path).strip()
    if target == REQUIRED_TOKENIZER_ID:
        return target
    candidate_path = Path(target)
    if candidate_path.is_dir():
        cfg_file = candidate_path / "tokenizer_config.json"
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            name_or_path = str(cfg.get("name_or_path", ""))
            if REQUIRED_TOKENIZER_ID in name_or_path or (candidate_path / "vocab.json").exists():
                return target
    raise ValueError(
        f"Only the '{REQUIRED_TOKENIZER_ID}' tokenizer is permitted for this project (got '{target}')."
    )


def load_starcoder2_tokenizer(tokenizer_name_or_path: str | Path = REQUIRED_TOKENIZER_ID):
    # Load the fast StarCoder2 tokenizer with synchronized BOS/EOS token IDs without model weights.
    validated = validate_tokenizer_source(tokenizer_name_or_path)
    config = AutoConfig.for_model(
        "starcoder2",
        vocab_size=49152,
        bos_token_id=0,
        eos_token_id=0,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        validated,
        config=config,
        bos_token_id=0,
        eos_token_id=0,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.eos_token_id != 0 or tokenizer.bos_token_id != 0:
        tokenizer.bos_token_id = 0
        tokenizer.eos_token_id = 0
    return tokenizer
