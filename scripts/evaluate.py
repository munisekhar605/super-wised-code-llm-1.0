from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.report import run_evaluation
from src.tokenizer.tokenizer import REQUIRED_TOKENIZER_ID


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate validation perplexity and Batch 1 code generation.")
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p.add_argument("--data-dir", type=Path, default=Path("data/tokenized"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--tokenizer", default=REQUIRED_TOKENIZER_ID)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument(
        "--prompt",
        default="// Write a JavaScript function that adds two numbers.\nexport function add(a, b) {\n",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main() -> None:
    p = build_arg_parser()
    args = p.parse_args()
    report = run_evaluation(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
