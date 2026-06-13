"""FreshQA loader. JSON list with question and acceptable answer(s); also carries
false_premise (FreshQA has false-premise questions). Time-sensitive retrieval benchmark.
Tolerates both `answers` (list) and `answer` (single) so the legacy stub and the full
download both load."""

from __future__ import annotations

from typing import List, Optional

from .base import Example, read_json


def load_freshqa(path: str, sample_size: Optional[int] = None) -> List[Example]:
    # Loads ALL items; the registry applies the official DEV/TEST split (carried in
    # gold["split"]) and the sample_size truncation.
    out: List[Example] = []
    for i, it in enumerate(read_json(path)):
        answers = it.get("answers")
        if not answers:
            single = it.get("answer")
            answers = [single] if single else []
        out.append(
            Example(
                id=str(it.get("id", i)),
                question=it["question"],
                dataset="freshqa",
                gold={
                    "answers": list(answers),
                    "false_premise": it.get("false_premise"),
                    # Official FreshQA split ("DEV"/"TEST"); the registry filters on it.
                    "split": it.get("split"),
                },
            )
        )
    return out
