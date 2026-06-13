"""Heuristic complexity score H in [0,1] for the Hybrid Decomposition Gate (paper Step 1).

H is "a normalized score computed via rule-based pattern matching (e.g. detecting multiple
entities, numerical constraints, or logical connectors)" (paper §III-A).

This module defines the concrete, dataset-AGNOSTIC rule set that realizes the paper's
description; each rule + weight is documented
for the supplement. (The legacy TruthfulQA-specific shortcut was removed — R3.)

The rules capture general STRUCTURAL signals of reasoning complexity that hold across domains,
not phrasings tied to any particular benchmark:
- multi-step / planning / logical / multi-entity structure (R1-R6);
- comparison / multi-hop structure — resolving each side is itself a step (R8);
- multiple distinct named entities (R9);
- recency / up-to-date external dependency — the answer turns on time-varying, externally
  verifiable facts that cannot be answered from static parametric memory (R10). This is the
  heuristic-side analog of the structured `external_dependency` dimension: a question phrased
  around "latest / current / as of / recent / now / this year" depends on the outside world
  and should be routed toward retrieval, not parametric guessing. It is a general structural
  signal (no dataset-specific phrasing).
These are properties of the QUESTION's structure; their generality is evaluated empirically
(held-out generalization test), not assumed.
"""

from __future__ import annotations

import re

# R1: multi-step / planning / optimization connectors.
_MULTI_STEP = [
    r"\b(step|steps|then|first|second|third|finally|next)\b",
    r"\b(plan|schedule|route|itinerary|pipeline|workflow)\b",
    r"\b(maximize|minimize|subject to|constraint|under constraints|optimal)\b",
    r"\b(compare|trade[- ]?off|balance|prioritize)\b",
]
# R2: logical connectors.
_LOGIC = [r"\b(prove|deduce|assume|therefore|hence|implies)\b", r"\bif\b.*\bthen\b"]
# R3: factual/temporal cues (pairs with numbers for type-mix).
_FACTUAL = [r"\b(latest|current|currently|updated|as of|recent|today|now)\b",
            r"\b(according to|published|reported)\b"]
# R10: recency / up-to-date external-dependency cues. Unlike R3 (which only contributes when a
# number co-occurs, signalling factual+numeric type-mix), these fire on their own: a question
# about the CURRENT / LATEST / most-recent state of the world depends on time-varying external
# facts and should be routed toward retrieval rather than answered from stale parametric memory.
_RECENCY = [r"\b(latest|current|currently|most recent|recently|as of|today|nowadays|this year)\b",
            r"\bright now\b"]
# R4: multiple-entity separators.
_ENTITY_SEPARATORS = [",", ";", " and ", " versus ", " vs ", " or "]
# R8: comparison / multi-hop — resolving each side is itself a step.
_COMPARISON = [
    r"\bmore\b.*\bthan\b", r"\bsame\b.*\bas\b", r"\bboth\b.*\band\b",
    r"\bcompared?\b\s+(to|with)\b",
    r"\b(older|younger|taller|shorter|bigger|smaller|earlier|later|longer|greater|fewer|"
    r"higher|lower|faster|slower)\b.*\bthan\b",
]

_W_MULTI_STEP = 0.25
_W_LOGIC = 0.20
_W_TYPE_MIX = 0.20
_W_MANY_ENTITIES = 0.20      # >= 2 entity separators
_W_MANY_NUMBERS = 0.15       # >= 3 distinct numbers
_W_GOAL_CONSTRAINT = 0.25    # explicit objective + explicit constraint co-occur
_W_COMPARISON = 0.30         # R8: comparison / multi-hop structure
_W_MANY_NAMED = 0.20         # R9: >= 3 distinct proper nouns
_W_RECENCY = 0.20            # R10: recency / up-to-date external-dependency cue


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

    # R8: comparison / multi-hop structure.
    if _matches_any(_COMPARISON, q):
        score += _W_COMPARISON
    # R9: multiple distinct named entities (rough proper-noun count on original casing).
    proper_nouns = {w for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", question or "")}
    if len(proper_nouns) >= 3:
        score += _W_MANY_NAMED
    # R10: recency / up-to-date external-dependency cue (fires on its own; see _RECENCY).
    if _matches_any(_RECENCY, q):
        score += _W_RECENCY

    return max(0.0, min(1.0, score))
