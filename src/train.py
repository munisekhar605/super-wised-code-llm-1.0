from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer
from src.model import SuperWisedCodeLM, count_parameters


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_tokens(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src.prepare_data first.")
    return np.memmap(path, dtype=np.uint32, mode="r")


def get_batch(tokens, batch_size: int, block_size: int, device: torch.device):
    if len(tokens) <= block_size + 1:
        raise ValueError(f"Need more than block_size+1={block_size + 1} tokens, got {len(tokens)}")
    starts = np.random.randint(0, len(tokens) - block_size - 1, size=batch_size)
    x = np.stack([tokens[i:i + block_size].astype(np.int64) for i in starts])
    y = np.stack([tokens[i + 1:i + block_size + 1].astype(np.int64) for i in starts])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def save_checkpoint(path: Path, model, optimizer, step, best_val, cfg, scaler):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "best_val": best_val, "config": cfg, "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp)
    tmp.replace(path)


def evaluate(model, val_tokens, cfg, device, amp_dtype, batches):
    model.eval(); losses = []
    with torch.no_grad():
        for _ in range(max(1, batches)):
            x, y = get_batch(val_tokens, cfg["batch_size"], cfg["block_size"], device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tokenizer", default="bigcode/starcoder2-15b")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    tokenizer_name = manifest.get("tokenizer", args.tokenizer)
    if args.tokenizer != "bigcode/starcoder2-15b" and manifest.get("tokenizer") and args.tokenizer != manifest["tokenizer"]:
        raise ValueError("Tokenizer differs from data manifest; use the same tokenizer IDs used during preprocessing.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    cfg["vocab_size"] = len(tokenizer)
    if cfg["vocab_size"] != manifest.get("vocab_size", cfg["vocab_size"]):
        raise ValueError("Tokenizer vocabulary size does not match data manifest.")
    train_tokens, val_tokens = load_tokens(args.data_dir / "train.bin"), load_tokens(args.data_dir / "val.bin")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.get("seed", 1337))
    model = SuperWisedCodeLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"]), weight_decay=cfg["weight_decay"])
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    start_step, best_val = 0, float("inf")
    latest = args.output_dir / "latest.pt"
    if args.resume:
        if not latest.exists(): raise FileNotFoundError(f"Cannot resume: {latest} does not exist")
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {})); start_step = int(state["step"]) + 1
        best_val = float(state.get("best_val", best_val))
        if "torch_rng" in state: torch.set_rng_state(state["torch_rng"])
        if "numpy_rng" in state: np.random.set_state(state["numpy_rng"])
        if "python_rng" in state: random.setstate(state["python_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resumed at step {start_step}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} parameters={count_parameters(model):,} vocab={cfg['vocab_size']} train_tokens={len(train_tokens):,} val_tokens={len(val_tokens):,}")
    model.train(); started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, cfg["max_steps"]):
        if step < cfg["warmup_steps"]:
            lr = cfg["learning_rate"] * (step + 1) / max(1, cfg["warmup_steps"])
        else:
            progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
            lr = cfg["learning_rate"] * (cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(progress, 1))))
        for group in optimizer.param_groups: group["lr"] = lr
        accum = cfg.get("gradient_accumulation_steps", 1)
        running_loss = 0.0
        for _ in range(accum):
            x, y = get_batch(train_tokens, cfg["batch_size"], cfg["block_size"], device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}; stop and inspect data/precision.")
            scaler.scale(loss).backward(); running_loss += float(loss.item())
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if step % cfg["eval_interval"] == 0 or step == cfg["max_steps"] - 1:
            val_loss = evaluate(model, val_tokens, cfg, device, amp_dtype, cfg["eval_batches"])
            print(f"step={step} train_loss={running_loss:.4f} val_loss={val_loss:.4f} lr={lr:.3e} elapsed_min={(time.time()-started)/60:.1f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.output_dir / "best.pt", model, optimizer, step, best_val, cfg, scaler)
        if step % cfg["save_interval"] == 0 or step == cfg["max_steps"] - 1:
            save_checkpoint(latest, model, optimizer, step, best_val, cfg, scaler)
    print(f"Training complete. best_val_loss={best_val:.4f}; outputs={args.output_dir}")

if __name__ == "__main__": main()
