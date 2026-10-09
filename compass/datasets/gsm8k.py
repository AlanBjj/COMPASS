"""GSM8K loader. JSONL with fields question / answer; the gold numeric answer follows the
'#### ' marker. Grade-school math benchmark."""

from __future__ import annotations

import re
from typing import List, Optional

from .base import Example, read_jsonl

_FINAL = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _final_number(answer: str) -> str:
    m = _FINAL.search(answer or "")
    return m.group(1).replace(",", "") if m else ""


def load_gsm8k(path: str, sample_size: Optional[int] = None) -> List[Example]:
    # Loads ALL items; the registry applies the dev/test split and sample_size truncation.
    out: List[Example] = []
    for i, it in enumerate(read_jsonl(path)):
        out.append(
            Example(
                id=str(i),
                question=it["question"],
                dataset="gsm8k",
                gold={"answer": it.get("answer", ""), "final": _final_number(it.get("answer", ""))},
            )
        )
    return out
