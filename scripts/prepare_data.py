from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import prepare_dataset
from src.data.validate import validate_dataset
from src.tokenizer.tokenizer import REQUIRED_TOKENIZER_ID


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter, deduplicate, screen secrets, tokenize, and validate Batch 1 JS/TS/Node/Express/NestJS code."
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--input-jsonl", type=Path, help="Local JSONL file with code records.")
    src.add_argument("--hf-dataset", help="Hugging Face dataset repo ID to stream.")
    p.add_argument(
        "--generate-sample-jsonl",
        action="store_true",
        help="Generate synthetic Batch 1 corpus for local/CI smoke testing ONLY. Do not use synthetic data as real training data.",
    )
    p.add_argument("--sample-records", type=int, default=240)
    p.add_argument("--validate-only", action="store_true", help="Validate an existing processed dataset directory.")
    p.add_argument("--hf-config", help="Optional Hugging Face dataset subset name.")
    p.add_argument(
        "--data-dir",
        "--hf-data-dir",
        dest="data_dir",
        help="Optional dataset subfolder / language directory (e.g. 'javascript' or 'typescript' for bigcode/starcoderdata).",
    )
    p.add_argument("--hf-token", help="Optional Hugging Face authentication token for gated datasets.")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--license-column", default="license")
    p.add_argument("--tokenizer", default=REQUIRED_TOKENIZER_ID)
    p.add_argument("--output-dir", type=Path, default=Path("data/tokenized"))
    p.add_argument("--max-records", type=int, default=50000)
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--min-split-tokens", type=int, default=2)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--allow-unlicensed-after-review", action="store_true")
    p.add_argument("--allow-license", action="append", default=[])
    return p


def main() -> None:
    p = build_arg_parser()
    args = p.parse_args()
    if args.validate_only:
        report = validate_dataset(args.output_dir, min_split_tokens=args.min_split_tokens)
        print(json.dumps(report, indent=2))
        return
    if not args.input_jsonl and not args.hf_dataset and not args.generate_sample_jsonl:
        p.error("Provide --input-jsonl, --hf-dataset, --generate-sample-jsonl, or --validate-only.")
    manifest = prepare_dataset(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
