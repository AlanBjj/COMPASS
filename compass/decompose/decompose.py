"""Step 2 — Sub-queries Decomposition (paper §III-B, Eq. 2).

A complex query is decomposed into 2-4 typed sub-queries S = {(q_i, t_i)}, with types in the
canonical set {Math, Logic, Fact, General} that Step 3 routes on. Reuses the legacy
decomposition prompt + line parser, but normalizes the type labels to the canonical four and
enforces the 2-4 bound.

On Eq. 2 / Info(q_i|q). The paper's objective
    S* = argmax_S [ sum_i Info(q_i | q) - lambda * |S| ]
formalizes WHAT decomposition optimizes, not an explicit per-candidate computation. We give
Info(q_i|q) the operational meaning "the marginal information gain of sub-query q_i toward
answering the main query q." It is optimized IMPLICITLY by the decomposition prompt, which asks
for the most critical, non-redundant sub-questions; the lambda*|S| compactness regularizer is
realized by the 2-4 bound plus the explicit non-redundancy instruction. We do NOT compute Info
numerically — stating this honestly matches the actual
implementation rather than overclaiming an algorithm that was never run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ..llm import render
from ..llm.client import CONTROLLER, LLMClient

# Canonical sub-query types (must match the solver router in Step 3).
MATH, LOGIC, FACT, GENERAL = "Math", "Logic", "Fact", "General"

_LINE = re.compile(r"^\s*\d+\.\s*\[([^\]]+)\]\s*(.+)$")

# Arithmetic-cue heuristic (shared with the Step 3 router via re-export). A sub-question that
# clearly asks for a numeric computation — a quantity/count/duration cue together with a digit —
# is treated as arithmetic. The 7B decomposer often mistypes such aggregation sub-steps ("how
# many more...", "how old is X currently") as Logic or Fact; left uncorrected this breaks the
# homogeneity check, so a pure-arithmetic problem is sliced down the harmful decompose path
# instead of collapsing to a single homogeneous Math call.
_NUMERIC_CUE = re.compile(
    r"how many|how much|how long|how old|how far|how fast|what fraction|what percent|"
    r"\btotal\b|\bsum\b|\bdifference\b|\bproduct\b|\btimes\b|\bper\b|\baverage\b|"
    r"\bremaining\b|\bleft over\b|\bin all\b|\baltogether\b|"
    # Computation verbs: solve/compute/calculate steps in a math chain (often the combine step)
    r"\bcalculat\w*|\bcomput\w*|\bsolve\b|\bequation\b|\bmultiply\b|\bdivide\b|\bsubtract\b|\badd up\b",
    re.IGNORECASE,
)


def _looks_arithmetic(text: str) -> bool:
    return bool(_NUMERIC_CUE.search(text)) and any(c.isdigit() for c in text)


@dataclass
class SubQuery:
    question: str
    type: str  # one of MATH / LOGIC / FACT / GENERAL

    def as_dict(self) -> dict:
        return {"question": self.question, "type": self.type}


def normalize_type(raw: str) -> str:
    """Map a free-form type label onto the canonical four. Defaults to General."""
    t = (raw or "").lower()
    if any(k in t for k in ("math", "calc", "arithmet", "numeric", "comput")):
        return MATH
    if any(k in t for k in ("logic", "reason", "inference", "deduc")):
        return LOGIC
    if any(k in t for k in ("fact", "retriev", "information", "knowledge-intensive")):
        return FACT
    if any(k in t for k in ("common", "general", "sense", "world")):
        return GENERAL
    return GENERAL


def parse_decomposition(text: str, *, max_subqueries: int = 4) -> List[SubQuery]:
    """Parse '1. [Type] question' lines into SubQuery objects, capped at max_subqueries.

    Arithmetic re-typing: the decomposer (a 7B controller) frequently labels clearly-arithmetic
    aggregation sub-steps ("how many more...", "how old is X now") as Logic or Fact. We reclassify
    such Logic/Fact sub-questions to Math when they match the arithmetic cue. This makes an
    all-arithmetic problem read as homogeneous Math (so the pipeline collapses it to one whole-
    question call rather than slicing it). We only override Logic/Fact — a genuinely heterogeneous
    problem (e.g. a fact lookup that is NOT itself a numeric computation) keeps its type and still
    decomposes correctly.
    """
    subs: List[SubQuery] = []
    for line in (text or "").strip().splitlines():
        m = _LINE.match(line)
        if m:
            q = m.group(2).strip()
            t = normalize_type(m.group(1))
            if t in (LOGIC, FACT) and _looks_arithmetic(q):
                t = MATH
            subs.append(SubQuery(question=q, type=t))
    return subs[:max_subqueries]


async def decompose(
    question: str,
    client: LLMClient,
    *,
    min_subqueries: int = 2,
    max_subqueries: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> List[SubQuery]:
    """Generate typed sub-queries for a complex query (a CONTROLLER call).

    The 2-4 bound is enforced as an upper cap on parsing; we do NOT fabricate sub-queries to
    reach min_subqueries (that would be inventing content). If parsing yields nothing, we fall
    back to a single General sub-query equal to the main query so the pipeline can proceed.
    """
    prompt = render("decompose", question=question)
    text = await client.chat_text(
        prompt, CONTROLLER, temperature=temperature, max_tokens=max_tokens
    )
    subs = parse_decomposition(text, max_subqueries=max_subqueries)
    if not subs:
        subs = [SubQuery(question=question, type=GENERAL)]
    return subs
