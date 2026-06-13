"""Baseline base types.

A baseline takes one question and returns a `BaselineResult` (final answer + a trace of the
intermediate steps, for inspection/debugging — the trace mirrors COMPASS's `CompassResult.trace`
so both can be browsed the same way). Baselines bill every backbone call to the BACKBONE role;
unlike COMPASS they have no separate controller, so the per-query token total is just the
backbone tally (the efficiency table compares that total against COMPASS's controller+backbone).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..llm.client import LLMClient
from ._retrieval_base import Retriever


@dataclass
class BaselineResult:
    answer: str
    trace: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"answer": self.answer, "trace": self.trace}


class Baseline(ABC):
    """One reproduced baseline method. Stateless across queries except the client ledger.

    Subclasses read their hyperparameters from `params` (the config's per-method block) so that
    ALL hyperparameters live in configs/, never hardcoded (project convention). Sampling params
    (temperature/max_tokens) come from the backbone model config and are passed on every call so
    every method generates at the protocol's temperature=0.
    """

    name: str = "baseline"

    def __init__(
        self,
        client: LLMClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        params: Optional[Dict[str, object]] = None,
        retriever: Optional[Retriever] = None,
        dataset: str = "",
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.params = params or {}
        self.retriever = retriever
        # Dataset name (truthfulqa/gsm8k/strategyqa/freshqa). Some methods are dataset-aware in
        # their OFFICIAL code (e.g. Rowen has per-dataset answer templates) — they read this.
        self.dataset = dataset

    @abstractmethod
    async def answer(self, question: str) -> BaselineResult:
        ...
