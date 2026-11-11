from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tokenizer.tokenizer import REQUIRED_TOKENIZER_ID


def validate_dataset(output_dir: Path, min_split_tokens: int = 2) -> dict[str, Any]:
    # Verify binary token files, manifest consistency, vocabulary bounds, and split lengths.
    manifest_path = output_dir / "manifest.json"
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    for p in (manifest_path, train_path, val_path):
        if not p.exists():
            raise FileNotFoundError(f"Dataset validation failed: missing {p}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tokenizer") != REQUIRED_TOKENIZER_ID and not str(manifest.get("tokenizer", "")).endswith(
        "starcoder2-15b"
    ):
        raise ValueError(
            f"Dataset manifest tokenizer must be '{REQUIRED_TOKENIZER_ID}', found '{manifest.get('tokenizer')}'"
        )
    vocab_size = int(manifest["vocab_size"])
    eos_token_id = int(manifest["eos_token_id"])
    if not (0 <= eos_token_id < vocab_size):
        raise ValueError(f"Invalid eos_token_id={eos_token_id} for vocab_size={vocab_size}")

    for split_name, bin_path, stat_key in (
        ("train", train_path, "train_tokens"),
        ("validation", val_path, "validation_tokens"),
    ):
        byte_size = bin_path.stat().st_size
        if byte_size == 0 or byte_size % 4 != 0:
            raise ValueError(f"{bin_path} byte size ({byte_size}) must be a positive multiple of 4 (uint32)")
        tokens = np.fromfile(bin_path, dtype=np.uint32)
        expected_len = int(manifest["stats"][stat_key])
        if len(tokens) != expected_len:
            raise ValueError(
                f"{split_name} token count mismatch: binary has {len(tokens)}, manifest records {expected_len}"
            )
        if len(tokens) < min_split_tokens:
            raise ValueError(
                f"{split_name} split has {len(tokens)} tokens, which is below minimum required {min_split_tokens}"
            )
        max_token = int(np.max(tokens))
        if max_token >= vocab_size:
            raise ValueError(
                f"{split_name} split contains out-of-vocabulary token ID {max_token} (vocab_size={vocab_size})"
            )

    return {
        "valid": True,
        "output_dir": str(output_dir),
        "tokenizer": manifest["tokenizer"],
        "vocab_size": vocab_size,
        "train_tokens": int(manifest["stats"]["train_tokens"]),
        "validation_tokens": int(manifest["stats"]["validation_tokens"]),
    }
