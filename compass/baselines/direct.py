"""Direct baseline = zero-shot Chain-of-Thought (the paper's 'Direct' baseline, run on the same
backbone). One backbone call: "think step by step, then Answer:". Matches the legacy
DirectAnswering/ChainOfThought intent but at temperature=0 per this protocol."""

from __future__ import annotations

from ..llm import render
from ..llm.client import BACKBONE
from .base import Baseline, BaselineResult


class DirectBaseline(Baseline):
    name = "direct"

    async def answer(self, question: str) -> BaselineResult:
        text = await self.client.chat_text(
            render("baselines/direct", question=question),
            BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return BaselineResult(answer=text, trace={"raw": text})
