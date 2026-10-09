"""General operator — direct generation.

a_i = DirectGen(q_i): for common-sense / general-world-knowledge sub-queries, use the model's
parametric knowledge directly, avoiding unnecessary retrieval. A backbone call. Also serves as the
generator of the direct plan and the low-quality fallback.
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
    prompt_name: str = "solver_general",
) -> str:
    # prompt_name: whole yes/no questions use prompts/solver_yesno.txt (implicit facts first).
    prompt = render(prompt_name, question=question, main_q=main_q)
    return await client.chat_text(
        prompt, BACKBONE, temperature=temperature, max_tokens=max_tokens
    )
