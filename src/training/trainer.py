from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from src.data.dataset import get_batch, load_tokens
from src.data.validate import validate_dataset
from src.tokenizer.tokenizer import (
    REQUIRED_TOKENIZER_ID,
    load_starcoder2_tokenizer,
    validate_tokenizer_source,
)
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.optimizer import build_optimizer
from src.training.scheduler import compute_lr
from src.training.validation import evaluate_validation_loss
from src.transformer.model import SuperWisedCodeLM, count_parameters
from src.utils.device import get_amp_context, get_device
from src.utils.logging import append_jsonl
from src.utils.seed import seed_everything


def load_merged_config(config_path: Path, training_config_path: Path | None = None) -> dict[str, Any]:
    # Load model configuration and merge optional separate training configuration.
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if training_config_path and training_config_path.exists():
        cfg.update(json.loads(training_config_path.read_text(encoding="utf-8")))
    elif "learning_rate" not in cfg:
        default_train = config_path.parent / "training.json"
        if default_train.exists():
            cfg.update(json.loads(default_train.read_text(encoding="utf-8")))
    return cfg


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    # Execute training loop with validation, checkpointing, automatic resume, and JSONL logging.
    cfg = load_merged_config(args.config, getattr(args, "training_config", None))
    if getattr(args, "max_steps", None) is not None:
        cfg["max_steps"] = int(args.max_steps)

    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    tokenizer_name = manifest.get("tokenizer", args.tokenizer)
    validate_tokenizer_source(args.tokenizer)
    validate_tokenizer_source(tokenizer_name)

    tokenizer = load_starcoder2_tokenizer(tokenizer_name)
    cfg["tokenizer_name"] = REQUIRED_TOKENIZER_ID
    cfg["vocab_size"] = len(tokenizer)
    if manifest.get("vocab_size") and cfg["vocab_size"] != int(manifest["vocab_size"]):
        raise ValueError("Tokenizer vocabulary size does not match data manifest.")

    if manifest_path.exists():
        validate_dataset(args.data_dir, min_split_tokens=int(cfg["block_size"]) + 2)

    train_tokens = load_tokens(args.data_dir / "train.bin")
    val_tokens = load_tokens(args.data_dir / "val.bin")
    device = get_device()
    seed_everything(int(cfg.get("seed", 1337)))

    model = SuperWisedCodeLM(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    use_amp, amp_dtype = get_amp_context(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    log_path = getattr(args, "log_file", None) or (args.output_dir / "metrics.jsonl")

    start_step = 0
    best_val = float("inf")
    should_resume = not getattr(args, "no_resume", False) and (getattr(args, "resume", False) or latest.exists())
    if getattr(args, "resume", False) and not latest.exists():
        raise FileNotFoundError(f"Cannot resume: {latest} does not exist")

    if should_resume and latest.exists():
        state = load_checkpoint(latest, model, device, optimizer=optimizer, scaler=scaler)
        start_step = int(state["step"]) + 1
        best_val = float(state.get("best_val", best_val))
        print(f"Resumed from {latest} at step {start_step} (best_val={best_val:.4f})")

    param_count = count_parameters(model)
    print(
        f"device={device} parameters={param_count:,} vocab={cfg['vocab_size']} "
        f"train_tokens={len(train_tokens):,} val_tokens={len(val_tokens):,}"
    )

    model.train()
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    accum = int(cfg.get("gradient_accumulation_steps", 1))
    tokens_per_step = int(cfg["batch_size"]) * int(cfg["block_size"]) * accum

    for step in range(start_step, int(cfg["max_steps"])):
        step_t0 = time.time()
        lr = compute_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        running_loss = 0.0
        for _ in range(accum):
            x, y = get_batch(train_tokens, int(cfg["batch_size"]), int(cfg["block_size"]), device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                assert loss is not None
                loss = loss / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}.")
            scaler.scale(loss).backward()
            running_loss += float(loss.item())

        scaler.unscale_(optimizer)
        grad_norm = float(clip_grad_norm_(model.parameters(), float(cfg["grad_clip"])).item())
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        tok_per_sec = tokens_per_step / max(time.time() - step_t0, 1e-6)
        is_last = step == int(cfg["max_steps"]) - 1

        if step % int(cfg["eval_interval"]) == 0 or is_last:
            val_loss = evaluate_validation_loss(model, val_tokens, cfg, device, amp_dtype, int(cfg["eval_batches"]))
            val_ppl = float(math.exp(min(val_loss, 20.0)))
            print(
                f"step={step} train_loss={running_loss:.4f} val_loss={val_loss:.4f} "
                f"val_ppl={val_ppl:.2f} lr={lr:.3e} tok_s={tok_per_sec:.0f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best_path, model, optimizer, step, best_val, cfg, scaler)
            append_jsonl(
                log_path,
                {
                    "step": step,
                    "train_loss": round(running_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "val_perplexity": round(val_ppl, 4),
                    "best_val_loss": round(best_val, 6),
                    "lr": lr,
                    "grad_norm": round(grad_norm, 6),
                    "tokens_per_sec": round(tok_per_sec, 2),
                    "elapsed_sec": round(time.time() - started, 2),
                },
            )

        if step % int(cfg["save_interval"]) == 0 or is_last:
            save_checkpoint(latest, model, optimizer, step, best_val, cfg, scaler)

    if not best_path.exists() and latest.exists():
        val_loss = evaluate_validation_loss(model, val_tokens, cfg, device, amp_dtype, int(cfg["eval_batches"]))
        best_val = min(best_val, val_loss)
        save_checkpoint(best_path, model, optimizer, max(0, start_step - 1), best_val, cfg, scaler)

    print(f"Training complete. best_val_loss={best_val:.4f}; outputs={args.output_dir}")
    return {
        "best_val_loss": best_val,
        "last_step": max(start_step, int(cfg["max_steps"])) - 1,
        "parameter_count": param_count,
        "output_dir": str(args.output_dir),
    }
