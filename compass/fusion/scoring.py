"""Step 4 — Sub-answer Quality Scoring (paper §III-D, Eq. 7-8).

An evaluator (CONTROLLER) rates each sub-answer on Accuracy and Relevance in [1,5];
Q = 0.6*s_acc + 0.4*s_rel; fusion weights w_i = Q_i / sum_j Q_j. Reuses the legacy scoring
prompt and weight math. The ONE thing legacy was missing — "if all sub-answers are low
quality (Q < 2) fall back to a direct answer for the main query" — is provided here as
`is_low_quality`, which the pipeline checks (Algorithm 1, fallback line). Weights come from
config (acc_weight/rel_weight), not hardcoded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..decompose.decompose import SubQuery
from ..llm import render
from ..llm.client import CONTROLLER, LLMClient


@dataclass
class ScoredSubAnswer:
    question: str
    type: str
    answer: str
    s_acc: int
    s_rel: int
    quality: float   # Q = acc_weight*s_acc + rel_weight*s_rel
    weight: float    # normalized fusion weight w_i = Q_i / sum Q_j

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "type": self.type,
            "answer": self.answer,
            "s_acc": self.s_acc,
            "s_rel": self.s_rel,
            "quality": round(self.quality, 3),
            "weight": round(self.weight, 4),
        }


def parse_scores(text: str) -> Tuple[int, int]:
    """Parse 'accuracy: X / relevance: X' (X in 1-5). Defaults to 3 on a miss."""
    t = (text or "").lower()
    a = re.search(r"accuracy\D*([1-5])", t)
    r = re.search(r"relevance\D*([1-5])", t)
    s_acc = int(a.group(1)) if a else 3
    s_rel = int(r.group(1)) if r else 3
    return max(1, min(5, s_acc)), max(1, min(5, s_rel))


def compute_weights(scored: List[ScoredSubAnswer]) -> List[ScoredSubAnswer]:
    """w_i = Q_i / sum_j Q_j; uniform if total is zero."""
    if not scored:
        return scored
    total = sum(s.quality for s in scored)
    for s in scored:
        s.weight = (s.quality / total) if total > 0 else (1.0 / len(scored))
    return scored


def is_low_quality(scored: Sequence[ScoredSubAnswer], threshold: float) -> bool:
    """True iff EVERY sub-answer is below threshold (paper: all Q < 2 -> direct fallback)."""
    return bool(scored) and all(s.quality < threshold for s in scored)


async def score_subanswers(
    main_q: str,
    pairs: Sequence[Tuple[SubQuery, str]],
    client: LLMClient,
    *,
    acc_weight: float,
    rel_weight: float,
    temperature: float = 0.0,
    max_tokens: int = 64,
) -> List[ScoredSubAnswer]:
    """Score each (sub-query, sub-answer) pair, then normalize fusion weights."""
    scored: List[ScoredSubAnswer] = []
    for sq, answer in pairs:
        text = await client.chat_text(
            render("scoring", main_q=main_q, sub_q=sq.question, sub_a=answer),
            CONTROLLER,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        s_acc, s_rel = parse_scores(text)
        quality = acc_weight * s_acc + rel_weight * s_rel
        scored.append(
            ScoredSubAnswer(sq.question, sq.type, answer, s_acc, s_rel, quality, 0.0)
        )
    return compute_weights(scored)
