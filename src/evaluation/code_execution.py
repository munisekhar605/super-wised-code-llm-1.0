from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any


def check_code_structure(code: str) -> dict[str, Any]:
    # Verify non-empty code output and balanced parentheses, braces, and brackets.
    pairs = {"(": ")", "{": "}", "[": "]"}
    closing = {")": "(", "}": "{", "]": "["}
    stack: list[str] = []
    unmatched_closing = 0
    for ch in code:
        if ch in pairs:
            stack.append(ch)
        elif ch in closing:
            if stack and stack[-1] == closing[ch]:
                stack.pop()
            else:
                unmatched_closing += 1
    non_empty_lines = [line for line in code.splitlines() if line.strip()]
    return {
        "non_empty": len(code.strip()) > 0,
        "line_count": len(non_empty_lines),
        "unclosed_delimiters": len(stack),
        "unmatched_closing_delimiters": unmatched_closing,
        "is_balanced": len(stack) == 0 and unmatched_closing == 0,
    }


def validate_javascript_syntax(code: str, timeout_sec: float = 5.0) -> dict[str, Any]:
    # Validate JavaScript syntax using Node.js --check if available, otherwise structural fallback.
    structure = check_code_structure(code)
    if not structure["non_empty"]:
        return {"valid": False, "error": "Empty code output", "engine": "structural", **structure}

    node_bin = shutil.which("node")
    if not node_bin:
        return {
            "valid": structure["is_balanced"],
            "error": None if structure["is_balanced"] else "Unbalanced delimiters in code",
            "engine": "structural_fallback",
            **structure,
        }

    fd, path = tempfile.mkstemp(suffix=".mjs")
    try:
        os.write(fd, code.encode("utf-8", "replace"))
        os.close(fd)
        res = subprocess.run(
            [node_bin, "--check", path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        is_valid = res.returncode == 0
        err = None if is_valid else (res.stderr.strip() or res.stdout.strip())
        return {"valid": is_valid, "error": err, "engine": "node_check", **structure}
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "valid": structure["is_balanced"],
            "error": f"Node check exception: {exc}",
            "engine": "structural_fallback",
            **structure,
        }
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def validate_typescript_syntax(code: str, timeout_sec: float = 5.0) -> dict[str, Any]:
    # Validate TypeScript syntax using Node.js native type stripping if available.
    structure = check_code_structure(code)
    if not structure["non_empty"]:
        return {"valid": False, "error": "Empty code output", "engine": "structural", **structure}

    node_bin = shutil.which("node")
    if not node_bin:
        return {
            "valid": structure["is_balanced"],
            "error": None if structure["is_balanced"] else "Unbalanced delimiters in code",
            "engine": "structural_fallback",
            **structure,
        }

    script = (
        "import { stripTypeScriptTypes } from 'node:module';"
        "import fs from 'node:fs';"
        "const src = fs.readFileSync(0, 'utf8');"
        "try {"
        "  stripTypeScriptTypes(src);"
        "  process.exit(0);"
        "} catch (err) {"
        "  process.stderr.write(String(err));"
        "  process.exit(1);"
        "}"
    )
    try:
        res = subprocess.run(
            [node_bin, "--input-type=module", "-e", script],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        is_valid = res.returncode == 0
        err = None if is_valid else (res.stderr.strip() or res.stdout.strip())
        return {"valid": is_valid, "error": err, "engine": "node_ts_strip", **structure}
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "valid": structure["is_balanced"],
            "error": f"Node TS check exception: {exc}",
            "engine": "structural_fallback",
            **structure,
        }


def validate_code_syntax(code: str, ecosystem_or_lang: str = "javascript") -> dict[str, Any]:
    # Dispatch syntax validation according to target Batch 1 ecosystem or language.
    target = ecosystem_or_lang.lower().strip()
    if target in {"typescript", "ts", "nestjs", "nest"}:
        return validate_typescript_syntax(code)
    return validate_javascript_syntax(code)
