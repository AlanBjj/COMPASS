"""Step 3 (General) — Direct Generation (paper §III-C, Eq. 6).

Paper: a_i = DirectGen(q_i) — for common-sense / general-world-knowledge sub-queries, use the
model's parametric knowledge directly, avoiding unnecessary retrieval. Reuses the legacy
common-knowledge prompt. A backbone call. Also serves as the direct-path / fallback generator
used by the gate's direct route and the low-quality fallback.
"""

from __future__ import annotations

from typing import Optional

from ..llm import render
from ..llm.client import BACKBONE, LLMClient


async def solve_general(
    question: str,
    main_q: str,
    client: LLMClient,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    prompt = render("solver_general", question=question, main_q=main_q)
    return await client.chat_text(
        prompt, BACKBONE, temperature=temperature, max_tokens=max_tokens
    )
