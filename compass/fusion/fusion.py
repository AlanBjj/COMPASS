"""Step 5 — Confidence-Aware Fusion (paper §III-E, Eq. 9).

A* = Fuse(q, {(q_i, a_i, w_i)}): synthesize (not concatenate) the weighted sub-answers into a
coherent answer, then run a final Consistency Check — A* is compared against the
high-confidence sub-answers and, if a contradiction is detected, a refinement step resolves it.

Reuses the legacy weighted-evidence fusion (which already synthesizes rather than concatenates),
but REWRITES the consistency step: legacy `global_consistency_check` just re-prompted "improve
the answer"; here the check is explicitly against the high-confidence sub-answers with a
conditional revise, matching the paper. Fusion + consistency are BACKBONE calls.
"""

from __future__ import annotations

from typing import List, Optional

from ..llm import render
from ..llm.client import BACKBONE, LLMClient
from .scoring import ScoredSubAnswer

_N_HIGH_CONFIDENCE = 2  # how many top-weighted sub-answers seed the consistency check


def _truncate(text: str, max_len: int = 1200) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_len else text[:max_len] + " ..."


def _format_evidence(scored: List[ScoredSubAnswer]) -> str:
    rows = sorted(scored, key=lambda s: s.weight, reverse=True)
    return "\n".join(
        f"[{s.weight:.3f}] {s.question} -> {_truncate(s.answer)}" for s in rows
    )


def _format_high_confidence(scored: List[ScoredSubAnswer]) -> str:
    rows = sorted(scored, key=lambda s: s.weight, reverse=True)[:_N_HIGH_CONFIDENCE]
    return "\n".join(f"- {s.question} -> {_truncate(s.answer)}" for s in rows)


def _is_meta_commentary(text: str) -> bool:
    """True when `text` talks ABOUT the answer (verification prose) rather than BEING the answer.

    Qwen2-7B sometimes returns the consistency-check reasoning instead of the answer itself
    (e.g. "Yes, the final answer contradicts the high-confidence sub-answers..."); such output
    must not be stored as the final answer.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    # Leading-phrase patterns that signal verification prose.
    meta_prefixes = (
        "yes, the final answer",
        "no, the final answer",
        "the final answer does",
        "the final answer is consistent",
    )
    if t.startswith(meta_prefixes):
        return True
    if "high-confidence sub-answer" in t:
        return True
    # "final answer" co-occurring with consistency-check vocabulary.
    if "final answer" in t and any(
        kw in t for kw in ("contradict", "consistent", "sub-answer")
    ):
        return True
    return False


async def fuse(
    main_q: str,
    scored: List[ScoredSubAnswer],
    client: LLMClient,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Weighted synthesis of sub-answers, followed by a consistency check + conditional refine."""
    a_star = await client.chat_text(
        render("fusion", main_q=main_q, evidence=_format_evidence(scored)),
        BACKBONE,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    final = await client.chat_text(
        render(
            "consistency",
            main_q=main_q,
            answer=a_star,
            high_conf=_format_high_confidence(scored),
        ),
        BACKBONE,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Guard: if the consistency call returned verification prose instead of the answer,
    # fall back to the fused answer a_star rather than storing meta-commentary.
    if _is_meta_commentary(final):
        return a_star
    return final
