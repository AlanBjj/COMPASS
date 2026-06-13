"""Step 3 (Math) — Structured Computational Solver (paper §III-C, Eq. 3).

Paper: a_i = Execute(Formulate(Variables(q_i))) — a three-step procedure (Variable
Identification -> Relationship Modeling -> Deterministic Execution). Rewritten from the legacy
generic "solve step by step" prompt, which had none of the structured stages, to the explicit
three-step prompt the paper specifies. A backbone call.
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
