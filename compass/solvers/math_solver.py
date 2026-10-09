"""Math operator — free-form chain-of-thought solver whose final number TRACE checks.

The prompt walks through variable identification, relationship modeling, and execution, and ends
with a final-answer line that TRACE canonicalizes. A backbone call.
"""

from __future__ import annotations

from typing import Optional

from ..llm import render
from ..llm.client import BACKBONE, LLMClient


async def solve_math(
    question: str,
    main_q: str,
    client: LLMClient,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    prompt = render("solver_math", question=question, main_q=main_q)
    return await client.chat_text(
        prompt, BACKBONE, temperature=temperature, max_tokens=max_tokens
    )
