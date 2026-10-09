"""Heuristic complexity score H in [0,1] for the Hybrid Decomposition Gate.

H is a normalized score computed by dataset-AGNOSTIC rule matching. The nine rules rho_1..rho_9 of
the paper capture general STRUCTURAL signals of reasoning complexity, not phrasings tied to any
particular benchmark:
- rho_1 multi-step / planning connectives; rho_2 logical connectives;
- rho_3 a number together with a factual/temporal cue (type mix);
- rho_4 multiple entity separators; rho_5 three or more numbers;
- rho_6 an explicit objective together with an explicit constraint;
- rho_7 comparison / multi-hop structure — resolving each side is itself a step;
- rho_8 three or more distinct proper nouns;
- rho_9 a recency cue — the answer turns on time-varying, externally verifiable facts ("latest /
  current / as of / recent / this year"), the heuristic-side analog of the structured
  `external_dependency` dimension.
Each rule adds its weight; H is clipped to [0,1].
"""

from __future__ import annotations

import re

# rho_1: multi-step / planning / optimization connectors.
_MULTI_STEP = [
    r"\b(step|steps|then|first|second|third|finally|next)\b",
    r"\b(plan|schedule|route|itinerary|pipeline|workflow)\b",
    r"\b(maximize|minimize|subject to|constraint|under constraints|optimal)\b",
    r"\b(compare|trade[- ]?off|balance|prioritize)\b",
]
# rho_2: logical connectors.
_LOGIC = [r"\b(prove|deduce|assume|therefore|hence|implies)\b", r"\bif\b.*\bthen\b"]
# rho_3: factual/temporal cues (pair with a number for the type mix).
_FACTUAL = [r"\b(latest|current|currently|updated|as of|recent|today|now)\b",
            r"\b(according to|published|reported)\b"]
# rho_9: recency / up-to-date external-dependency cues. Unlike rho_3 (which only contributes when a
# number co-occurs, signalling factual+numeric type-mix), these fire on their own: a question
# about the CURRENT / LATEST / most-recent state of the world depends on time-varying external
# facts and should be routed toward retrieval rather than answered from stale parametric memory.
_RECENCY = [r"\b(latest|current|currently|most recent|recently|as of|today|nowadays|this year)\b",
            r"\bright now\b"]
# rho_4: multiple-entity separators.
_ENTITY_SEPARATORS = [",", ";", " and ", " versus ", " vs ", " or "]
# rho_7: comparison / multi-hop — resolving each side is itself a step.
_COMPARISON = [
    r"\bmore\b.*\bthan\b", r"\bsame\b.*\bas\b", r"\bboth\b.*\band\b",
    r"\bcompared?\b\s+(to|with)\b",
    r"\b(older|younger|taller|shorter|bigger|smaller|earlier|later|longer|greater|fewer|"
    r"higher|lower|faster|slower)\b.*\bthan\b",
]

_W_MULTI_STEP = 0.25
_W_LOGIC = 0.20
_W_TYPE_MIX = 0.20
_W_MANY_ENTITIES = 0.20      # rho_4: >= 2 entity separators
_W_MANY_NUMBERS = 0.15       # rho_5: >= 3 numbers
_W_GOAL_CONSTRAINT = 0.25    # rho_6: explicit objective + explicit constraint co-occur
_W_COMPARISON = 0.30         # rho_7: comparison / multi-hop structure
_W_MANY_NAMED = 0.20         # rho_8: >= 3 distinct proper nouns
_W_RECENCY = 0.20            # rho_9: recency / up-to-date external-dependency cue


def _matches_any(patterns, text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def heuristic_score(question: str) -> float:
    """Return H in [0,1] from dataset-agnostic rule matching."""
    q = (question or "").lower()
    score = 0.0
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", q)

    if _matches_any(_MULTI_STEP, q):
        score += _W_MULTI_STEP
    if _matches_any(_LOGIC, q):
        score += _W_LOGIC
    if numbers and _matches_any(_FACTUAL, q):
        score += _W_TYPE_MIX
    if sum(q.count(sep) for sep in _ENTITY_SEPARATORS) >= 2:
        score += _W_MANY_ENTITIES
    if len(numbers) >= 3:
        score += _W_MANY_NUMBERS

    has_goal = bool(re.search(r"\b(maximize|minimize|optimal|best|cheapest|fastest)\b", q))
    has_constraint = bool(re.search(r"\b(constraint|subject to|without|at most|at least|within)\b", q))
    if has_goal and has_constraint:
        score += _W_GOAL_CONSTRAINT

    # rho_7: comparison / multi-hop structure.
    if _matches_any(_COMPARISON, q):
        score += _W_COMPARISON
    # rho_8: multiple distinct named entities (rough proper-noun count on original casing).
    proper_nouns = {w for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", question or "")}
    if len(proper_nouns) >= 3:
        score += _W_MANY_NAMED
    # rho_9: recency / up-to-date external-dependency cue (fires on its own; see _RECENCY).
    if _matches_any(_RECENCY, q):
        score += _W_RECENCY

    return max(0.0, min(1.0, score))
