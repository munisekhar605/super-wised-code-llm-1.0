from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.publish import package_release
from src.data.dataset import prepare_dataset
from src.data.deduplicate import compute_dedup_hashes
from src.data.filter import (
    classify_batch1_ecosystem,
    contains_secret,
    is_batch1_compatible,
    looks_like_code,
    norm_license,
    reject_path,
)
from src.data.pack import stable_validation_split
from src.data.validate import validate_dataset
from src.evaluation.code_execution import check_code_structure
from src.evaluation.report import run_evaluation
from src.tokenizer.tokenizer import REQUIRED_TOKENIZER_ID
from src.training.trainer import run_training


class TrainingAndDataPipelineTests(unittest.TestCase):
    def test_batch1_filtering_secrets_and_dedup(self) -> None:
        self.assertEqual(norm_license(["Apache-2.0", "MIT"]), "apache-2.0;mit")
        self.assertFalse(reject_path("src/controllers/auth.controller.ts"))
        self.assertTrue(reject_path("node_modules/express/index.js"))
        self.assertTrue(reject_path("scripts/train.py"))

        nest_code = "import { Controller, Get } from '@nestjs/common';\n@Controller('users')\nexport class UsersController {}"
        tags = classify_batch1_ecosystem({"path": "src/users.controller.ts"}, nest_code)
        self.assertIn("nestjs", tags)
        self.assertFalse(is_batch1_compatible({"language": "python"}, "def add(a, b):\n    return a + b\n"))

        self.assertTrue(contains_secret("const key = 'AKIAIOSFODNN7EXAMPLE';"))
        self.assertFalse(contains_secret("export function sum(a: number, b: number): number { return a + b; }"))

        code_a = "export function greet(name: string): string {\n  return `Hello, ${name}`;\n}\n"
        code_b = "export function greet(name: string): string {   return `Hello, ${name}`; }\r\n"
        self.assertEqual(compute_dedup_hashes(code_a)[1], compute_dedup_hashes(code_b)[1])
        self.assertTrue(looks_like_code(code_a))
        self.assertEqual(
            stable_validation_split({"repo_name": "acme/api"}, "a", 0.2),
            stable_validation_split({"repo_name": "acme/api"}, "b", 0.2),
        )
        self.assertEqual(check_code_structure(code_a)["unmatched_closing_delimiters"], 0)

    def test_end_to_end_prepare_train_resume_evaluate_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_jsonl = root / "raw.jsonl"
            data_dir = root / "tokenized"
            ckpt_dir = root / "checkpoints"
            release_dir = root / "release"
            cfg_path = root / "model.json"

            prep_args = argparse.Namespace(
                input_jsonl=raw_jsonl,
                hf_dataset=None,
                generate_sample_jsonl=True,
                sample_records=80,
                validate_only=False,
                hf_config=None,
                split="train",
                text_column="text",
                license_column="license",
                tokenizer=REQUIRED_TOKENIZER_ID,
                output_dir=data_dir,
                max_records=500,
                min_chars=40,
                min_split_tokens=100,
                validation_fraction=0.2,
                allow_unlicensed_after_review=False,
                allow_license=[],
            )
            manifest = prepare_dataset(prep_args)
            self.assertTrue(validate_dataset(data_dir, min_split_tokens=100)["valid"])

            cfg = {
                "model_name": "super-wised-code-llm-test",
                "tokenizer_name": REQUIRED_TOKENIZER_ID,
                "vocab_size": manifest["vocab_size"],
                "n_layer": 2,
                "n_head": 4,
                "n_kv_head": 2,
                "n_embd": 64,
                "intermediate_size": 128,
                "block_size": 64,
                "dropout": 0.0,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.95,
                "grad_clip": 1.0,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_steps": 4,
                "eval_interval": 2,
                "eval_batches": 2,
                "save_interval": 2,
                "warmup_steps": 1,
                "min_lr_ratio": 0.1,
                "seed": 42,
            }
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

            res1 = run_training(
                argparse.Namespace(
                    config=cfg_path,
                    training_config=None,
                    data_dir=data_dir,
                    output_dir=ckpt_dir,
                    log_file=ckpt_dir / "metrics.jsonl",
                    resume=False,
                    no_resume=True,
                    max_steps=2,
                    tokenizer=REQUIRED_TOKENIZER_ID,
                )
            )
            self.assertEqual(res1["last_step"], 1)

            res2 = run_training(
                argparse.Namespace(
                    config=cfg_path,
                    training_config=None,
                    data_dir=data_dir,
                    output_dir=ckpt_dir,
                    log_file=ckpt_dir / "metrics.jsonl",
                    resume=True,
                    no_resume=False,
                    max_steps=4,
                    tokenizer=REQUIRED_TOKENIZER_ID,
                )
            )
            self.assertEqual(res2["last_step"], 3)

            eval_report = run_evaluation(
                argparse.Namespace(
                    config=cfg_path,
                    data_dir=data_dir,
                    checkpoint=ckpt_dir / "best.pt",
                    tokenizer=REQUIRED_TOKENIZER_ID,
                    batches=2,
                    prompt="export function add(a, b) {\n",
                    max_new_tokens=10,
                    temperature=0.0,
                    top_k=1,
                    output_json=ckpt_dir / "eval.json",
                )
            )
            self.assertEqual(len(eval_report["batch1_generation_suite"]), 5)

            package_release(
                argparse.Namespace(
                    checkpoint=ckpt_dir / "best.pt",
                    config=cfg_path,
                    tokenizer=REQUIRED_TOKENIZER_ID,
                    repo_id="test-org/super-wised-code-llm-1.0",
                    output_dir=release_dir,
                    private=True,
                    push=False,
                )
            )
            self.assertTrue((release_dir / "model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
