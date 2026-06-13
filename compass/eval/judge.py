"""Automated judge (gpt-5.4-mini). Produces both metrics the paper reports —
Hallucination Rate (binary) and Accuracy (0-100) — in ONE call, for ALL four datasets.

Fixes the legacy evaluator, which only computed HR for TruthfulQA (so the paper's HR numbers
for GSM8K/FreshQA/StrategyQA could not be produced) and whose score parsing was brittle
(grabbed the first digit, defaulted silently). Here the reference is adapted per dataset and a
parse failure is marked invalid (hallucination/accuracy = -1) and excluded from metrics rather
than silently defaulted. Judge runs on a SEPARATE client (judge endpoint) and bills the
'judge' role. The human-eval/IAA validation is a separate experiment on top of this judge.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..datasets.base import Example
from ..llm import render
from ..llm.client import JUDGE, LLMClient


@dataclass
class Judgment:
    hallucination: int   # 0 / 1, or -1 if the judge output could not be parsed
    accuracy: int        # 0-100, or -1 if unparsable
    raw: str

    @property
    def valid(self) -> bool:
        return self.hallucination >= 0

    def as_dict(self) -> dict:
        return {"hallucination": self.hallucination, "accuracy": self.accuracy}


def format_reference(example: Example) -> str:
    """Build the dataset-specific reference block for the judge."""
    g, d = example.gold, example.dataset
    if d == "truthfulqa":
        return (f"Correct answers: {g.get('correct','')}\n"
                f"Best answer: {g.get('best','')}\n"
                f"Incorrect answers: {g.get('incorrect','')}")
    if d == "gsm8k":
        return f"Correct final answer: {g.get('final','')}\nReference solution: {g.get('answer','')}"
    if d == "strategyqa":
        return f"Correct answer (yes/no): {g.get('answer','')}"
    if d == "freshqa":
        ref = f"Acceptable answers: {', '.join(map(str, g.get('answers', [])))}"
        if str(g.get("false_premise")).lower() in ("true", "1"):
            ref += "\n(Note: this is a false-premise question.)"
        return ref
    return str(g)


def parse_judgment(text: str) -> Tuple[int, int]:
    """Parse the judge's JSON. Raises on malformed output (caller marks it invalid).

    The judge may emit more than one flat JSON object (e.g. echoed few-shot examples
    followed by its real verdict). Extract every flat object and return the LAST one
    that parses and carries the required keys — the actual verdict comes last.
    """
    candidates = re.findall(r"\{[^{}]*\}", text or "", flags=re.DOTALL)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            h = 1 if int(obj["hallucination"]) else 0
            a = max(0, min(100, int(obj["accuracy"])))
            return h, a
        except (ValueError, KeyError, TypeError):
            continue
    raise ValueError("no parseable JSON verdict in judge output")


async def judge_one(example: Example, answer: str, client: LLMClient) -> Judgment:
    text = await client.chat_text(
        render("judge", question=example.question, reference=format_reference(example), answer=answer),
        JUDGE,
        temperature=0.0,
        max_tokens=512,  # gpt-5.x judges emit reasoning_content before the JSON — leave room
    )
    try:
        h, a = parse_judgment(text)
        return Judgment(h, a, text)
    except Exception:
        return Judgment(-1, -1, text)  # invalid -> excluded from metrics (no silent default)


async def judge_all(
    examples: Sequence[Example], answers: Sequence[str], client: LLMClient
) -> List[Judgment]:
    return list(await asyncio.gather(*[judge_one(e, a, client) for e, a in zip(examples, answers)]))
