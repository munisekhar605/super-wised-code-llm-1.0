from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(log_path: Path, entry: dict[str, Any]) -> None:
    # Append a single structured JSON record to a JSONL metrics log file.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
