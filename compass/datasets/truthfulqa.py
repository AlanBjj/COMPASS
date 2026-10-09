"""TruthfulQA loader. Raw JSON list with fields Question / Best Answer / Correct Answers /
Incorrect Answers (semicolon-separated). Common-sense hallucination benchmark."""

from __future__ import annotations

from typing import List, Optional

from .base import Example, read_json


def load_truthfulqa(path: str, sample_size: Optional[int] = None) -> List[Example]:
    # Loads ALL items; the registry applies the dev/test split and sample_size truncation.
    out: List[Example] = []
    for i, it in enumerate(read_json(path)):
        out.append(
            Example(
                id=str(it.get("id", i)),
                question=it["Question"],
                dataset="truthfulqa",
                gold={
                    "best": it.get("Best Answer", ""),
                    "correct": it.get("Correct Answers", ""),
                    "incorrect": it.get("Incorrect Answers", ""),
                },
            )
        )
    return out
