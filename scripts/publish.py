from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tokenizer.tokenizer import (
    REQUIRED_TOKENIZER_ID,
    load_starcoder2_tokenizer,
)
from src.transformer.model import SuperWisedCodeLM, count_parameters


def package_release(args: argparse.Namespace) -> Path:
    # Package checkpoint weights (SafeTensors), config, modular src/, tokenizer, and model card.
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = state.get("config") or json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer = load_starcoder2_tokenizer(args.tokenizer)
    cfg["vocab_size"] = len(tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = SuperWisedCodeLM(cfg)
    model.load_state_dict(state["model"])
    param_count = count_parameters(model)

    save_file(
        {k: v.clone().contiguous().cpu() for k, v in model.state_dict().items()},
        args.output_dir / "model.safetensors",
    )
    (args.output_dir / "model_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (args.output_dir / "training_state.json").write_text(
        json.dumps(
            {
                "step": state.get("step"),
                "best_validation_loss": state.get("best_val"),
                "parameter_count": param_count,
                "tokenizer": REQUIRED_TOKENIZER_ID,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    dst_src = args.output_dir / "src"
    if dst_src.exists():
        shutil.rmtree(dst_src)
    shutil.copytree(
        ROOT / "src",
        dst_src,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    tokenizer.save_pretrained(args.output_dir)

    card = (
        "---\n"
        "language:\n"
        "- javascript\n"
        "- typescript\n"
        "tags:\n"
        "- code\n"
        "- text-generation\n"
        "- from-scratch\n"
        "- nodejs\n"
        "- expressjs\n"
        "- nestjs\n"
        "---\n"
        f"# {args.repo_id.split('/')[-1]}\n\n"
        "A custom decoder-only coding-model checkpoint (RoPE, RMSNorm, SwiGLU, GQA, causal attention) "
        f"trained from randomly initialized weights. It uses the `{REQUIRED_TOKENIZER_ID}` tokenizer only; "
        "no pretrained StarCoder2 model weights are loaded.\n"
    )
    (args.output_dir / "README.md").write_text(card, encoding="utf-8")
    print(f"Release artifacts written to {args.output_dir.resolve()}")
    if not args.push:
        print("Local release package created. Add --push with HF_TOKEN set to upload to Hugging Face Hub.")
        return args.output_dir

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise OSError("Set HF_TOKEN in your environment before using --push.")
    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=token)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.output_dir),
        commit_message="Publish Super-Wised Coding LLM checkpoint",
    )
    print(f"Uploaded: https://huggingface.co/{args.repo_id}")
    return args.output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Package and optionally publish an evaluated checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p.add_argument("--tokenizer", default=REQUIRED_TOKENIZER_ID)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("release"))
    p.add_argument("--private", action="store_true")
    p.add_argument("--push", action="store_true")
    return p


def main() -> None:
    p = build_arg_parser()
    args = p.parse_args()
    package_release(args)


if __name__ == "__main__":
    main()
