"""Logic operator — bidirectional consistency check.

a_i = Check(Forward(q_i), Backward(q_i)). Derive the answer by forward chaining (premises ->
conclusion), verify by backward chaining (conclusion -> conditions), then a check step reconciles
the two. Three backbone calls.
"""

from __future__ import annotations

from typing import Optional

from ..llm import render
from ..llm.client import BACKBONE, LLMClient


async def solve_logic(
    question: str,
    main_q: str,
    client: LLMClient,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    forward = await client.chat_text(
        render("solver_logic_forward", question=question, main_q=main_q),
        BACKBONE,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    backward = await client.chat_text(
        render("solver_logic_backward", question=question, main_q=main_q),
        BACKBONE,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await client.chat_text(
        render(
            "solver_logic_check",
            question=question,
            main_q=main_q,
            forward=forward,
            backward=backward,
        ),
        BACKBONE,
        temperature=temperature,
        max_tokens=max_tokens,
    )
