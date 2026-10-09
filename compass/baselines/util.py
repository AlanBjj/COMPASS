"""Small shared helpers for baseline methods (answer extraction, integer-score parsing).

The judge sees the FULL answer text, so extraction here only needs to be lenient: prefer the text
after an explicit "Answer:"/"FINAL:" marker, else return the whole completion. Keeping reasoning
in the returned answer is fine — the LLM judge extracts the final answer itself (judge.py).
"""

from __future__ import annotations

import re
from typing import Optional


def extract_answer(text: str) -> str:
    """Return the text after the last 'Answer:' / 'FINAL:' marker, else the whole string."""
    if not text:
        return ""
    m = list(re.finditer(r"(?:answer|final)\s*[:\-]\s*", text, flags=re.IGNORECASE))
    if m:
        return text[m[-1].end():].strip()
    return text.strip()


def parse_int_score(text: str, lo: int = 0, hi: int = 10, default: Optional[int] = None) -> Optional[int]:
    """First integer in `text`, clamped to [lo, hi]. Returns `default` if none found."""
    m = re.search(r"-?\d+", text or "")
    if not m:
        return default
    return max(lo, min(hi, int(m.group(0))))
