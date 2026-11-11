from __future__ import annotations

from typing import Any

import torch

from src.evaluation.code_execution import check_code_structure, validate_code_syntax
from src.inference.generate import generate_tokens
from src.tokenizer.decode import decode_tokens
from src.tokenizer.encode import encode_prompt_tensor
from src.transformer.model import SuperWisedCodeLM

BATCH1_EVAL_PROMPTS = [
    {
        "ecosystem": "javascript",
        "prompt": "export function calculateInvoiceTotal(items, taxRate = 0.18) {\n",
    },
    {
        "ecosystem": "typescript",
        "prompt": "export interface CacheEntry<T> {\n  readonly key: string;\n",
    },
    {
        "ecosystem": "nodejs",
        "prompt": "import { createHash } from 'node:crypto';\nexport function sha256Hex(input) {\n",
    },
    {
        "ecosystem": "expressjs",
        "prompt": "import { Router } from 'express';\nexport function createHealthRouter() {\n  const router = Router();\n",
    },
    {
        "ecosystem": "nestjs",
        "prompt": "import { Controller, Get } from '@nestjs/common';\n@Controller('health')\nexport class HealthController {\n",
    },
]


def run_batch1_benchmarks(
    model: SuperWisedCodeLM,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int = 48,
    temperature: float = 0.7,
    top_k: int = 40,
) -> list[dict[str, Any]]:
    # Run code-generation benchmark prompts across all 5 Batch 1 ecosystems.
    results: list[dict[str, Any]] = []
    for item in BATCH1_EVAL_PROMPTS:
        p_ids = encode_prompt_tensor(tokenizer, item["prompt"], device=device)
        out_ids = generate_tokens(
            model,
            p_ids,
            max_new_tokens=min(max_new_tokens, 48),
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion_ids = out_ids[0, p_ids.size(1) :]
        full_text = decode_tokens(tokenizer, out_ids[0], skip_special_tokens=True)
        completion_text = decode_tokens(tokenizer, completion_ids, skip_special_tokens=True)
        results.append(
            {
                "ecosystem": item["ecosystem"],
                "prompt": item["prompt"],
                "completion": completion_text,
                "generated_tokens": int(completion_ids.numel()),
                "structure": check_code_structure(full_text),
                "syntax_validation": validate_code_syntax(full_text, item["ecosystem"]),
            }
        )
    return results
