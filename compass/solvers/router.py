"""Step 3 — Type-Aware router: dispatch each typed sub-query to its expert solver.

Implements the Algorithm 1 routing block: Math -> StructuredCompute, Logic ->
BidirectionalCheck, Fact -> WeightedRAG, else -> DirectGen. Cleaner than the legacy
substring-matching dispatch: types are already normalized to the canonical four in Step 2.
"""

from __future__ import annotations

from typing import Optional

from ..decompose.decompose import FACT, LOGIC, MATH, SubQuery, _looks_arithmetic
from ..llm.client import LLMClient
from .general_solver import solve_general
from .logic_solver import solve_logic
from .math_solver import solve_math
from .rag_solver import Retriever, solve_rag

# `_looks_arithmetic` is shared with Step 2 (decompose), which already retypes clearly-arithmetic
# Logic/Fact sub-steps to Math. This guard remains as a defence in depth: any Fact sub-query that
# is really a numeric computation goes to the math solver, not RAG web-search (where the number
# would be lost and the answer would degrade to "the evidence does not contain...").


async def route(
    subquery: SubQuery,
    client: LLMClient,
    *,
    main_q: str,
    retriever: Optional[Retriever] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Answer one typed sub-query with its expert solver. Fact with no configured retriever
    degrades to direct generation (so the pipeline still runs without a search backend).

    `main_q` is the original main question, passed to each solver as context so sub-queries
    can resolve numbers/references that the decomposition dropped (sub-queries are not always
    self-contained on a 7B controller)."""
    q = subquery.question
    if subquery.type == MATH:
        return await solve_math(q, main_q, client, temperature=temperature, max_tokens=max_tokens)
    if subquery.type == LOGIC:
        return await solve_logic(q, main_q, client, temperature=temperature, max_tokens=max_tokens)
    if subquery.type == FACT:
        # Numeric-leak guard: a Fact sub-query that is really an arithmetic question goes to the
        # math solver, not RAG, so the number is not lost in web-search.
        if _looks_arithmetic(q):
            return await solve_math(q, main_q, client, temperature=temperature, max_tokens=max_tokens)
        if retriever is None:
            return await solve_general(q, main_q, client, temperature=temperature, max_tokens=max_tokens)
        return await solve_rag(q, main_q, client, retriever, temperature=temperature, max_tokens=max_tokens)
    return await solve_general(q, main_q, client, temperature=temperature, max_tokens=max_tokens)
