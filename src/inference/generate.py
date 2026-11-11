from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.inference.sampling import sample_next_token
from src.inference.stop_tokens import should_stop_on_eos
from src.tokenizer.decode import decode_tokens
from src.tokenizer.encode import encode_prompt_tensor
from src.tokenizer.tokenizer import load_starcoder2_tokenizer
from src.transformer.model import SuperWisedCodeLM
from src.utils.device import get_device


@torch.no_grad()
def generate_tokens(
    model: SuperWisedCodeLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    # Autoregressively generate tokens (B, T) -> (B, T + max_new_tokens).
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        idx = input_ids[:, -model.block_size :]
        logits, _ = model(idx)
        next_id = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
        input_ids = torch.cat((input_ids, next_id), dim=1)
        if should_stop_on_eos(next_id, eos_token_id):
            break
    if was_training:
        model.train()
    return input_ids


def run_inference(args: argparse.Namespace) -> str:
    # Load checkpoint and decode generated code completion for a prompt.
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    device = get_device()
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = state["config"]
    tokenizer = load_starcoder2_tokenizer(args.tokenizer)
    if len(tokenizer) != int(cfg["vocab_size"]):
        raise ValueError("Tokenizer vocabulary size does not match checkpoint.")
    model = SuperWisedCodeLM(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    ids = encode_prompt_tensor(tokenizer, args.prompt, device=device)
    output = generate_tokens(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )
    return decode_tokens(tokenizer, output[0], skip_special_tokens=True)
