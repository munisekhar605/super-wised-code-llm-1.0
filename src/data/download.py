from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def iter_local(path: Path) -> Iterable[dict[str, Any]]:
    # Stream records line-by-line from a local JSONL file.
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"skip malformed JSONL line {line_no}", file=sys.stderr)
                continue
            if isinstance(item, dict):
                yield item


def iter_hf(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    # Stream records from a Hugging Face dataset repository.
    import os
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"path": args.hf_dataset, "split": args.split, "streaming": True}
    if getattr(args, "hf_config", None):
        kwargs["name"] = args.hf_config
    data_dir = getattr(args, "data_dir", None) or getattr(args, "hf_data_dir", None)
    if data_dir:
        kwargs["data_dir"] = data_dir
    token = getattr(args, "hf_token", None) or os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    ds = load_dataset(**kwargs)
    yield from ds


def generate_batch1_sample_jsonl(output_path: Path, num_records: int = 240) -> int:
    # Generate a synthetic Batch 1 (JS/TS/Node.js/Express.js/NestJS) smoke-test JSONL corpus.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    templates = [
        (
            "javascript",
            "src/utils/math_{i}.js",
            "repo-js-core-{bucket}",
            (
                "export function calculateInvoiceTotal{i}(items, taxRate = 0.18) {{\n"
                "  if (!Array.isArray(items)) throw new TypeError('items must be an array');\n"
                "  const subtotal = items.reduce((acc, item) => acc + Number(item.price || 0) * Number(item.qty || 1), 0);\n"
                "  const tax = Math.round(subtotal * taxRate * 100) / 100;\n"
                "  return {{ invoiceId: {i}, subtotal, tax, total: subtotal + tax }};\n"
                "}}\n"
            ),
        ),
        (
            "typescript",
            "src/services/cache_{i}.ts",
            "repo-ts-platform-{bucket}",
            (
                "export interface CacheEntry<T> {{\n"
                "  readonly key: string;\n"
                "  readonly value: T;\n"
                "  readonly expiresAt: number;\n"
                "}}\n\n"
                "export class MemoryCacheService{i}<T = unknown> {{\n"
                "  private readonly store = new Map<string, CacheEntry<T>>();\n"
                "  constructor(private readonly defaultTtlMs: number = {ttl}) {{}}\n"
                "  set(key: string, value: T, ttlMs = this.defaultTtlMs): void {{\n"
                "    this.store.set(key, {{ key, value, expiresAt: Date.now() + ttlMs }});\n"
                "  }}\n"
                "  get(key: string): T | undefined {{\n"
                "    const entry = this.store.get(key);\n"
                "    if (!entry) return undefined;\n"
                "    if (Date.now() > entry.expiresAt) {{\n"
                "      this.store.delete(key);\n"
                "      return undefined;\n"
                "    }}\n"
                "    return entry.value;\n"
                "  }}\n"
                "}}\n"
            ),
        ),
        (
            "nodejs",
            "src/runtime/worker_{i}.mjs",
            "repo-node-runtime-{bucket}",
            (
                "import {{ createHash }} from 'node:crypto';\n"
                "import {{ EventEmitter }} from 'node:events';\n\n"
                "export class JobProcessor{i} extends EventEmitter {{\n"
                "  async processPayload(payload) {{\n"
                "    const serialized = JSON.stringify(payload ?? {{}});\n"
                "    const digest = createHash('sha256').update(Buffer.from(serialized, 'utf8')).digest('hex');\n"
                "    this.emit('processed', {{ jobId: {i}, digest, bytes: Buffer.byteLength(serialized) }});\n"
                "    return {{ jobId: {i}, digest }};\n"
                "  }}\n"
                "}}\n"
            ),
        ),
        (
            "expressjs",
            "src/routes/orders_{i}.ts",
            "repo-express-api-{bucket}",
            (
                "import {{ Router, Request, Response, NextFunction }} from 'express';\n\n"
                "export function createOrderRouter{i}(): Router {{\n"
                "  const router = Router();\n"
                "  router.post('/orders/{i}', async (req: Request, res: Response, next: NextFunction) => {{\n"
                "    try {{\n"
                "      const {{ customerId, items }} = req.body ?? {{}};\n"
                "      if (!customerId || !Array.isArray(items) || items.length === 0) {{\n"
                "        res.status(400).json({{ error: 'Invalid order payload for route {i}' }});\n"
                "        return;\n"
                "      }}\n"
                "      res.status(201).json({{ orderId: 'ORD-{i}', customerId, count: items.length }});\n"
                "    }} catch (err) {{\n"
                "      next(err);\n"
                "    }}\n"
                "  }});\n"
                "  return router;\n"
                "}}\n"
            ),
        ),
        (
            "nestjs",
            "src/modules/health_{i}.controller.ts",
            "repo-nestjs-service-{bucket}",
            (
                "import {{ Controller, Get, Injectable, Module, Param, ParseIntPipe }} from '@nestjs/common';\n\n"
                "@Injectable()\n"
                "export class TelemetryService{i} {{\n"
                "  getStatus(shardId: number) {{\n"
                "    return {{ serviceId: {i}, shardId, healthy: true, uptimeSec: {ttl} }};\n"
                "  }}\n"
                "}}\n\n"
                "@Controller('v1/telemetry-{i}')\n"
                "export class TelemetryController{i} {{\n"
                "  constructor(private readonly telemetry: TelemetryService{i}) {{}}\n"
                "  @Get(':shardId')\n"
                "  checkShard(@Param('shardId', ParseIntPipe) shardId: number) {{\n"
                "    return this.telemetry.getStatus(shardId);\n"
                "  }}\n"
                "}}\n\n"
                "@Module({{ controllers: [TelemetryController{i}], providers: [TelemetryService{i}] }})\n"
                "export class TelemetryModule{i} {{}}\n"
            ),
        ),
    ]
    licenses = ["MIT", "Apache-2.0", "BSD-3-Clause", "ISC"]
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for i in range(num_records):
            stack, path_tmpl, repo_tmpl, code_tmpl = templates[i % len(templates)]
            bucket = i % 20
            record = {
                "text": code_tmpl.format(i=i, ttl=1000 + i * 25, bucket=bucket),
                "license": licenses[i % len(licenses)],
                "language": "typescript" if path_tmpl.endswith(".ts") else "javascript",
                "framework": stack,
                "repo_name": repo_tmpl.format(bucket=bucket),
                "source": f"synthetic-batch1-{stack}",
                "path": path_tmpl.format(i=i),
            }
            f.write(json.dumps(record) + "\n")
            written += 1
    return written
