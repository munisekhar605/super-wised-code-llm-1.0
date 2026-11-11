from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from src.model import SuperWisedCodeLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--tokenizer', default='bigcode/starcoder2-15b')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--top-k', type=int, default=40)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = state['config']
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if len(tokenizer) != cfg['vocab_size']:
        raise ValueError('Tokenizer vocabulary size does not match the checkpoint.')
    model = SuperWisedCodeLM(cfg).to(device)
    model.load_state_dict(state['model']); model.eval()
    ids = tokenizer(args.prompt, return_tensors='pt').input_ids.to(device)
    output = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(output[0], skip_special_tokens=False))

if __name__ == '__main__': main()
