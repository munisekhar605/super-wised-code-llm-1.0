from __future__ import annotations

import hashlib
from typing import Any, BinaryIO

import numpy as np


def stable_validation_split(record: dict[str, Any], text: str, fraction: float = 0.02) -> bool:
    # Deterministically assign entire repositories or sources to train vs validation splits.
    repo = record.get("repo_name") or record.get("repository")
    path = record.get("path")
    source = record.get("source")
    if repo:
        key = "repo:" + str(repo).strip().lower()
    elif source and source != "unspecified":
        key = "source:" + str(source).strip().lower()
    elif path:
        key = "path:" + str(path).strip().lower()
    else:
        key = "text:" + text[:300]
    digest = hashlib.sha256(key.encode("utf-8", "ignore")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def write_uint32_tokens(handle: BinaryIO, token_ids: list[int]) -> int:
    # Pack token IDs as contiguous uint32 binary values and write to file stream.
    arr = np.asarray(token_ids, dtype=np.uint32)
    arr.tofile(handle)
    return int(arr.size)
