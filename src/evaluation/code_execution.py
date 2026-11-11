from __future__ import annotations

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
        "NoUnmatchedClosing": unmatched_closing == 0,
    }
