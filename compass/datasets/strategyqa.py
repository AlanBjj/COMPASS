"""StrategyQA loader — NEW (legacy had no StrategyQA loader or data at all). JSON list with
question and a boolean answer; normalized to "yes"/"no". Implicit multi-hop reasoning
benchmark. Field names vary across releases (qid/id, answer as bool or yes/no), handled here."""

from __future__ import annotations

from typing import List, Optional

from .base import Example, read_json


def _yes_no(answer) -> str:
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    s = str(answer).strip().lower()
    if s in ("yes", "true", "1"):
        return "yes"
    if s in ("no", "false", "0"):
        return "no"
    return s


def load_strategyqa(path: str, sample_size: Optional[int] = None) -> List[Example]:
    # Loads ALL items; the registry applies the dev/test split and sample_size truncation.
    out: List[Example] = []
    for i, it in enumerate(read_json(path)):
        out.append(
            Example(
                id=str(it.get("qid", it.get("id", i))),
                question=it["question"],
                dataset="strategyqa",
                gold={"answer": _yes_no(it.get("answer"))},
            )
        )
    return out
