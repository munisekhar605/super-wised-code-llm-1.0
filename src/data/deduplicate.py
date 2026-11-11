from __future__ import annotations

import hashlib
import re


def compute_dedup_hashes(text: str) -> tuple[str, str]:
    # Return exact SHA-256 digest and whitespace-normalized SHA-256 digest.
    exact = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
    normalized_text = re.sub(r"\s+", " ", text.strip())
    normalized = hashlib.sha256(normalized_text.encode("utf-8", "ignore")).hexdigest()
    return exact, normalized


class Deduplicator:
    """Stateful exact and whitespace-normalized content deduplicator."""

    def __init__(self) -> None:
        self.seen_hashes: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        exact_hash, norm_hash = compute_dedup_hashes(text)
        if exact_hash in self.seen_hashes or norm_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(exact_hash)
        self.seen_hashes.add(norm_hash)
        return False
