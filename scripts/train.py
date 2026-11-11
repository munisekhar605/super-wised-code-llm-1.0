from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tokenizer.tokenizer import REQUIRED_TOKENIZER_ID
from src.training.trainer import run_training


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SuperWisedCodeLM with automatic checkpoint resume and logging.")
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p.add_argument("--training-config", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=Path("data/tokenized"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--resume", action="store_true", help="Explicitly require resuming from output_dir/latest.pt.")
    p.add_argument("--no-resume", action="store_true", help="Start fresh even if output_dir/latest.pt exists.")
    p.add_argument("--max-steps", type=int, default=None, help="Optional override for max_steps.")
    p.add_argument("--tokenizer", default=REQUIRED_TOKENIZER_ID)
    return p


def main() -> None:
    p = build_arg_parser()
    args = p.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
