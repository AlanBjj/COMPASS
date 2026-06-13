"""Hybrid Decomposition Gate (paper Step 1, Eq. 1 / Algorithm 1 lines 2-4).

    Decision = I[ alpha * H + (1 - alpha) * L >= tau ],  alpha=0.4, tau=0.55.

Faithful to the paper, unlike the legacy gate which used tau=0.80, added an extra
`recommendation == "decompose"` AND-condition, and short-circuited the LLM when H<0.3.
Here the decision is exactly the weighted threshold; alpha and tau come from config.
The full GateDecision (H, L, complexity) is returned so it can be logged and reused by
the later sensitivity study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ..llm.client import LLMClient
from .heuristic import heuristic_score
from .structured import structured_score


@dataclass
class GateDecision:
    decompose: bool
    H: float
    L: float
    complexity: float       # alpha*H + (1-alpha)*L
    alpha: float
    tau: float
    raw: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "decompose": self.decompose,
            "H": round(self.H, 4),
            "L": round(self.L, 4),
            "complexity": round(self.complexity, 4),
            "alpha": self.alpha,
            "tau": self.tau,
            "raw": self.raw,
        }


async def decide(
    question: str,
    client: LLMClient,
    *,
    alpha: float,
    tau: float,
    controller_temperature: float = 0.0,
    controller_max_tokens: int = 256,
) -> GateDecision:
    """Run the gate for one query. complexity >= tau -> decompose; else direct path."""
    H = heuristic_score(question)
    L, raw = await structured_score(
        question,
        client,
        temperature=controller_temperature,
        max_tokens=controller_max_tokens,
    )
    complexity = alpha * H + (1.0 - alpha) * L
    return GateDecision(
        decompose=complexity >= tau,
        H=H,
        L=L,
        complexity=complexity,
        alpha=alpha,
        tau=tau,
        raw=raw,
    )
