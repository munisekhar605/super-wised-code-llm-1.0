from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.data.dataset import load_tokens
from src.evaluation.benchmarks import run_batch1_benchmarks
from src.evaluation.perplexity import evaluate_perplexity
from src.inference.generate import generate_tokens
from src.tokenizer.decode import decode_tokens
from src.tokenizer.encode import encode_prompt_tensor
from src.tokenizer.tokenizer import load_starcoder2_tokenizer
from src.training.trainer import load_merged_config
from src.transformer.model import SuperWisedCodeLM, count_parameters
from src.utils.device import get_device


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    # Evaluate validation perplexity and Batch 1 code generation and optionally write JSON report.
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    cfg = load_merged_config(args.config)
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    tokenizer = load_starcoder2_tokenizer(manifest.get("tokenizer", args.tokenizer))

    device = get_device()
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = dict(state.get("config", cfg))
    cfg["vocab_size"] = len(tokenizer)

    model = SuperWisedCodeLM(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    val_tokens = load_tokens(args.data_dir / "val.bin")
    ppl_metrics = evaluate_perplexity(model, val_tokens, cfg, device, batches=args.batches)

    prompt_ids = encode_prompt_tensor(tokenizer, args.prompt, device=device)
    generated = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded_primary = decode_tokens(tokenizer, generated[0], skip_special_tokens=True)
    suite_samples = run_batch1_benchmarks(
        model,
        tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    report = {
        **ppl_metrics,
        "checkpoint_step": state.get("step"),
        "parameter_count": count_parameters(model),
        "generated_sample": decoded_primary,
        "batch1_generation_suite": suite_samples,
    }
    if getattr(args, "output_json", None):
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
