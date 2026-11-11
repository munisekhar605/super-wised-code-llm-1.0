from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.deduplicate import Deduplicator
from src.data.download import generate_batch1_sample_jsonl, iter_hf, iter_local
from src.data.filter import (
    LICENSE_ALLOWLIST,
    classify_batch1_ecosystem,
    contains_secret,
    is_batch1_compatible,
    looks_like_code,
    norm_license,
    reject_path,
)
from src.data.pack import stable_validation_split, write_uint32_tokens
from src.data.tokenize import tokenize_record
from src.data.validate import validate_dataset
from src.tokenizer.tokenizer import REQUIRED_TOKENIZER_ID, load_starcoder2_tokenizer


def load_tokens(path: Path) -> np.memmap:
    # Memory-map a packed uint32 token file for zero-copy training reads.
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run prepare_data first.")
    return np.memmap(path, dtype=np.uint32, mode="r")


def get_batch(
    tokens: np.ndarray | np.memmap, batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Sample random contiguous (input, target) windows of shape (B, T) shifted by 1 token.
    if len(tokens) <= block_size + 1:
        raise ValueError(f"Need more than block_size+1={block_size + 1} tokens, got {len(tokens)}")
    starts = np.random.randint(0, len(tokens) - block_size - 1, size=batch_size)
    x = np.stack([tokens[i : i + block_size].astype(np.int64) for i in starts])
    y = np.stack([tokens[i + 1 : i + block_size + 1].astype(np.int64) for i in starts])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    # Run full Batch 1 pipeline: download/read -> filter -> deduplicate -> tokenize -> pack -> validate.
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_starcoder2_tokenizer(args.tokenizer)
    allowed = LICENSE_ALLOWLIST | {x.lower().strip() for x in args.allow_license}

    if getattr(args, "generate_sample_jsonl", False):
        sample_path = args.input_jsonl or (args.output_dir.parent / "raw" / "raw.jsonl")
        generate_batch1_sample_jsonl(sample_path, num_records=getattr(args, "sample_records", 240))
        args.input_jsonl = sample_path

    iterator = iter_local(args.input_jsonl) if args.input_jsonl else iter_hf(args)
    train_path, val_path = args.output_dir / "train.bin", args.output_dir / "val.bin"
    for path in (train_path, val_path):
        if path.exists():
            path.unlink()

    stats = {
        "records_seen": 0,
        "kept": 0,
        "rejected_short_or_generated": 0,
        "rejected_non_batch1": 0,
        "rejected_secret_pattern": 0,
        "rejected_license": 0,
        "duplicates": 0,
        "train_records": 0,
        "validation_records": 0,
        "train_tokens": 0,
        "validation_tokens": 0,
    }
    dedup = Deduplicator()
    source_ids: dict[str, int] = {}
    ecosystem_counts: dict[str, int] = {
        "javascript": 0,
        "typescript": 0,
        "nodejs": 0,
        "expressjs": 0,
        "nestjs": 0,
    }

    with train_path.open("wb") as train_f, val_path.open("wb") as val_f:
        for row in iterator:
            if stats["records_seen"] >= args.max_records:
                break
            stats["records_seen"] += 1
            if reject_path(row.get("path")):
                stats["rejected_short_or_generated"] += 1
                continue
            text = row.get(args.text_column)
            if not isinstance(text, str):
                for fallback_col in ("content", "code"):
                    candidate = row.get(fallback_col)
                    if isinstance(candidate, str):
                        text = candidate
                        break
            if not isinstance(text, str):
                stats["rejected_short_or_generated"] += 1
                continue
            text = text.strip()
            if not looks_like_code(text, min_chars=args.min_chars):
                stats["rejected_short_or_generated"] += 1
                continue
            if not is_batch1_compatible(row, text):
                stats["rejected_non_batch1"] += 1
                continue
            if contains_secret(text):
                stats["rejected_secret_pattern"] += 1
                continue
            lic = norm_license(row.get(args.license_column))
            if lic:
                licenses = {x.strip() for x in lic.split(";") if x.strip()}
                if not licenses.intersection(allowed):
                    stats["rejected_license"] += 1
                    continue
            elif not args.allow_unlicensed_after_review:
                stats["rejected_license"] += 1
                continue

            if dedup.is_duplicate(text):
                stats["duplicates"] += 1
                continue

            encoded = tokenize_record(tokenizer, text)
            if not encoded:
                continue

            is_val = stable_validation_split(row, text, args.validation_fraction)
            out_f = val_f if is_val else train_f
            n_written = write_uint32_tokens(out_f, encoded)

            bucket = "validation" if is_val else "train"
            stats[f"{bucket}_records"] += 1
            stats[f"{bucket}_tokens"] += n_written
            stats["kept"] += 1

            for tag in classify_batch1_ecosystem(row, text):
                ecosystem_counts[tag] = ecosystem_counts.get(tag, 0) + 1

            source = str(row.get("source") or row.get("repo_name") or row.get("repository") or "unspecified")
            source_ids[source] = source_ids.get(source, 0) + 1

    manifest = {
        "schema_version": 1,
        "batch": "Batch 1 (JavaScript, TypeScript, Node.js, Express.js, NestJS)",
        "tokenizer": REQUIRED_TOKENIZER_ID,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "dtype": "uint32",
        "source_dataset": args.hf_dataset or str(args.input_jsonl),
        "split": args.split if args.hf_dataset else None,
        "text_column": args.text_column,
        "license_column": args.license_column,
        "allowlist": sorted(allowed),
        "allow_unlicensed_after_review": args.allow_unlicensed_after_review,
        "is_synthetic_smoke_corpus": bool(getattr(args, "generate_sample_jsonl", False)),
        "validation_fraction": args.validation_fraction,
        "stats": stats,
        "ecosystem_counts": ecosystem_counts,
        "source_counts": source_ids,
        "warning": "Regex secret screening and source license metadata are imperfect; manually review corpus provenance and samples.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    min_tokens = getattr(args, "min_split_tokens", 2)
    if stats["train_tokens"] < min_tokens or stats["validation_tokens"] < min_tokens:
        raise SystemExit(
            f"Not enough train/validation tokens (train={stats['train_tokens']}, "
            f"val={stats['validation_tokens']}, required={min_tokens})."
        )
    manifest["validation_report"] = validate_dataset(args.output_dir, min_split_tokens=min_tokens)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
