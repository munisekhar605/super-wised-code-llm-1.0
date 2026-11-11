from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from transformers import AutoTokenizer

LICENSE_ALLOWLIST = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "0bsd",
    "mpl-2.0", "unlicense", "cc0-1.0", "zlib", "artistic-2.0"
}
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)(api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
]
GENERATED_MARKERS = ("@generated", "do not edit", "sourceMappingURL=data:")
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".lock", ".map", ".min.js", ".min.css", ".wasm"}


def norm_license(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(norm_license(v) for v in value)
    return str(value).strip().lower().replace("license:", "").strip()


def reject_path(path_value: Any) -> bool:
    if not isinstance(path_value, str):
        return False
    path = path_value.replace("\\", "/").lower()
    parts = set(path.split("/"))
    if parts.intersection({"node_modules", "vendor", "dist", "build", "coverage", ".git", "__pycache__", "third_party"}):
        return True
    return path.endswith((".min.js", ".min.css", ".map", ".lock", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".wasm"))


def looks_like_code(text: str) -> bool:
    if len(text.strip()) < 40 or len(text) > 500_000:
        return False
    low = text.lower()
    if any(marker in low for marker in GENERATED_MARKERS):
        return False
    if text.count("\x00") > 0:
        return False
    return True


def contains_secret(text: str) -> bool:
    return any(p.search(text) for p in SECRET_PATTERNS)


def stable_validation_split(record: dict, text: str, fraction: float = 0.02) -> bool:
    repo = record.get("repo_name") or record.get("repository")
    path = record.get("path")
    if repo:
        key = "repo:" + str(repo)
    elif path:
        key = "path:" + str(record.get("source") or "") + "\0" + str(path)
    else:
        key = "text:" + text[:300]
    digest = hashlib.sha256(key.encode("utf-8", "ignore")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def iter_local(path: Path) -> Iterable[dict]:
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


def iter_hf(args) -> Iterable[dict]:
    from datasets import load_dataset
    kwargs = {"path": args.hf_dataset, "split": args.split, "streaming": True}
    if args.hf_config:
        kwargs["name"] = args.hf_config
    ds = load_dataset(**kwargs)
    for row in ds:
        yield row


def main() -> None:
    p = argparse.ArgumentParser(description="Filter and tokenize coding records into packed uint32 files.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-jsonl", type=Path)
    src.add_argument("--hf-dataset")
    p.add_argument("--hf-config")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--license-column", default="license")
    p.add_argument("--tokenizer", default="bigcode/starcoder2-15b")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-records", type=int, default=50000)
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--validation-fraction", type=float, default=0.02)
    p.add_argument("--allow-unlicensed-after-review", action="store_true", help="Use only after manually verifying source rights and recording provenance.")
    p.add_argument("--allow-license", action="append", default=[], help="Additional explicitly reviewed SPDX license ID; repeat as needed.")
    args = p.parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        p.error("validation-fraction must be between 0 and 0.5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no EOS token; choose a compatible tokenizer.")
    allowed = LICENSE_ALLOWLIST | {x.lower().strip() for x in args.allow_license}
    iterator = iter_local(args.input_jsonl) if args.input_jsonl else iter_hf(args)
    train_path, val_path = args.output_dir / "train.bin", args.output_dir / "val.bin"
    for path in (train_path, val_path):
        if path.exists():
            path.unlink()
    stats = {"records_seen": 0, "kept": 0, "rejected_short_or_generated": 0, "rejected_secret_pattern": 0,
             "rejected_license": 0, "duplicates": 0, "train_records": 0, "validation_records": 0,
             "train_tokens": 0, "validation_tokens": 0}
    seen_hashes: set[str] = set()
    source_ids: dict[str, int] = {}
    with train_path.open("wb") as train_f, val_path.open("wb") as val_f:
        for row in iterator:
            if stats["records_seen"] >= args.max_records:
                break
            stats["records_seen"] += 1
            if reject_path(row.get("path")):
                stats["rejected_short_or_generated"] += 1
                continue
            text = row.get(args.text_column)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < args.min_chars or not looks_like_code(text):
                stats["rejected_short_or_generated"] += 1
                continue
            if contains_secret(text):
                stats["rejected_secret_pattern"] += 1
                continue
            lic = norm_license(row.get(args.license_column))
            if lic:
                licenses = {x.strip() for x in lic.split(";") if x.strip()}
                if not licenses.intersection(allowed):
                    stats["rejected_license"] += 1
                    continue
            elif not args.allow_unlicensed_after_review:
                stats["rejected_license"] += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
            if digest in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(digest)
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if not encoded:
                continue
            encoded.append(tokenizer.eos_token_id)
            is_val = stable_validation_split(row, text, args.validation_fraction)
            out_f = val_f if is_val else train_f
            np.asarray(encoded, dtype=np.uint32).tofile(out_f)
            bucket = "validation" if is_val else "train"
            stats[f"{bucket}_records"] += 1
            stats[f"{bucket}_tokens"] += len(encoded)
            stats["kept"] += 1
            source = str(row.get("source") or row.get("repo_name") or row.get("repository") or "unspecified")
            source_ids[source] = source_ids.get(source, 0) + 1
            if stats["records_seen"] % 1000 == 0:
                print(f"processed={stats['records_seen']} kept={stats['kept']} train_tokens={stats['train_tokens']} val_tokens={stats['validation_tokens']}")
    manifest = {
        "schema_version": 1, "tokenizer": args.tokenizer, "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id, "dtype": "uint32", "source_dataset": args.hf_dataset or str(args.input_jsonl),
        "split": args.split if args.hf_dataset else None, "text_column": args.text_column,
        "license_column": args.license_column, "allowlist": sorted(allowed), "allow_unlicensed_after_review": args.allow_unlicensed_after_review,
        "validation_fraction": args.validation_fraction, "stats": stats, "source_counts": source_ids,
        "warning": "Regex secret screening and source license metadata are imperfect; manually review corpus provenance and samples."
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if stats["train_tokens"] < 2 or stats["validation_tokens"] < 2:
        raise SystemExit("Not enough train/validation tokens. Increase the sample or inspect filters/split.")

if __name__ == "__main__":
    main()
