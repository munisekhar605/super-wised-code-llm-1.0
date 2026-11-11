from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from src.model import SuperWisedCodeLM
from src.train import get_batch, load_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", default="bigcode/starcoder2-15b")
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--prompt", default="// Write a JavaScript function that adds two numbers.\n")
    p.add_argument("--max-new-tokens", type=int, default=80)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(manifest.get("tokenizer", args.tokenizer), use_fast=True)
    cfg["vocab_size"] = len(tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = state.get("config", cfg)
    model = SuperWisedCodeLM(cfg).to(device)
    model.load_state_dict(state["model"]); model.eval()
    tokens = load_tokens(args.data_dir / "val.bin")
    losses = []
    with torch.no_grad():
        for _ in range(max(1, args.batches)):
            x, y = get_batch(tokens, cfg["batch_size"], cfg["block_size"], device)
            _, loss = model(x, y); losses.append(loss.item())
    ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    generated = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=0.7, top_k=40)
    print(json.dumps({"validation_loss_mean": float(np.mean(losses)), "validation_loss_std": float(np.std(losses)),
                      "checkpoint_step": state.get("step"), "parameter_count": sum(p.numel() for p in model.parameters()),
                      "generated_sample": tokenizer.decode(generated[0], skip_special_tokens=False)}, indent=2))

if __name__ == "__main__": main()
