from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer
from src.model import SuperWisedCodeLM


def main():
    p = argparse.ArgumentParser(description="Package and optionally publish an evaluated checkpoint to the Hugging Face Hub.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokenizer", default="bigcode/starcoder2-15b")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("release"))
    p.add_argument("--private", action="store_true")
    p.add_argument("--push", action="store_true", help="Actually upload. Without this flag, only creates local release artifacts.")
    args = p.parse_args()
    if not args.checkpoint.exists(): raise FileNotFoundError(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = state.get("config") or json.loads(args.config.read_text())
    tokenizer_name = args.tokenizer
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = SuperWisedCodeLM(cfg)
    model.load_state_dict(state["model"])
    save_file({k: v.clone().contiguous().cpu() for k, v in model.state_dict().items()}, args.output_dir / "model.safetensors")
    (args.output_dir / "model_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (args.output_dir / "training_state.json").write_text(json.dumps({"step": state.get("step"), "best_validation_loss": state.get("best_val"), "parameter_count": sum(p.numel() for p in model.parameters())}, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("model.py"), args.output_dir / "model.py")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.save_pretrained(args.output_dir)
    card = f'''---\nlanguage:\n- javascript\n- typescript\ntags:\n- code\n- text-generation\n- from-scratch\n---\n# {args.repo_id.split('/')[-1]}\n\nA custom decoder-only coding-model checkpoint trained from randomly initialized weights. It uses the `{tokenizer_name}` tokenizer; no pretrained StarCoder2 model weights are loaded.\n\n## Architecture\n\nSee `model_config.json` and `model.py`. This repository currently packages a custom PyTorch model, not a Transformers `AutoModelForCausalLM` implementation. Load it using the provided model class and matching configuration.\n\n## Training status\n\n- Checkpoint step: {state.get('step')}\n- Best validation loss recorded by training: {state.get('best_val')}\n- Parameter count: {sum(p.numel() for p in model.parameters()):,}\n\nThese metrics are not a coding benchmark. Evaluate generated code with held-out tests before describing this checkpoint as production-ready. Review tokenizer and dataset licenses and add a complete dataset provenance/model card before making the repository public.\n'''
    (args.output_dir / "README.md").write_text(card, encoding="utf-8")
    print(f"Release artifacts written to {args.output_dir.resolve()}")
    if not args.push:
        print("Not uploaded. Review artifacts, dataset rights, and model card. Add --push to upload.")
        return
    token = os.environ.get("HF_TOKEN")
    if not token: raise EnvironmentError("Set HF_TOKEN in your environment before using --push.")
    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=token)
    api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(args.output_dir), commit_message="Publish Super-Wised Coding LLM checkpoint")
    print(f"Uploaded: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__": main()
