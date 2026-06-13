"""Structured complexity score L in [0,1] for the Hybrid Decomposition Gate (paper Step 1).

L is "derived from an LLM's structured evaluation of the four dimensions. The model assigns
discrete values {0,1,2} to each dimension, which are aggregated and normalized to [0,1]"
(paper §III-A). Four dimensions x max 2 = 8, so L = sum / 8.

This is a CONTROLLER call (the open-source controller). On any parse
failure we fall back to L=0.0 (conservative -> routes to the direct path), matching the
legacy behavior, rather than guessing.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Tuple

from ..llm import render
from ..llm.client import CONTROLLER, LLMClient

_DIMENSIONS = ("multi_step", "cognitive_diversity", "constraint_coupling", "external_dependency")
_MAX_SUM = 2 * len(_DIMENSIONS)  # = 8


def _extract_json(text: str) -> Dict[str, object]:
    """Pull the first JSON object out of the model output (tolerates surrounding prose)."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return json.loads(match.group(0))


def parse_L(text: str) -> Tuple[float, Dict[str, int]]:
    """Parse model output into (L in [0,1], per-dimension scores). Raises on bad output."""
    obj = _extract_json(text)
    scores: Dict[str, int] = {}
    for dim in _DIMENSIONS:
        val = int(obj.get(dim, 0))
        scores[dim] = max(0, min(2, val))
    L = sum(scores.values()) / _MAX_SUM
    return L, scores


async def structured_score(
    question: str,
    client: LLMClient,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> Tuple[float, Dict[str, object]]:
    """Return (L, raw) where raw holds the per-dimension scores or an error marker."""
    prompt = render("gate_structured", question=question)
    text = await client.chat_text(
        prompt, CONTROLLER, temperature=temperature, max_tokens=max_tokens
    )
    try:
        L, scores = parse_L(text)
        return L, {"scores": scores}
    except Exception as err:  # noqa: BLE001 - conservative fallback on malformed output
        return 0.0, {"error": str(err), "raw_output": text}
