"""Self-Refine (Madaan et al. 2023): generate -> self-feedback -> refine, iterated up to
max_iters, stopping early when the model's own feedback is "NO_ISSUES". Same model for generation,
feedback, and refinement (the original uses one frozen LM). No retrieval."""

from __future__ import annotations

from ..llm import render
from ..llm.client import BACKBONE
from .base import Baseline, BaselineResult


class SelfRefineBaseline(Baseline):
    name = "self_refine"

    async def answer(self, question: str) -> BaselineResult:
        # max_attempts=4 is the upstream GSM default (src/gsm/run.py argparse default).
        max_iters = int(self.params.get("max_iters", 4))
        # DETERMINISM DEVIATION: upstream runs feedback at temperature=0.7
        # (GSMFeedback(..., temperature=0.7) in src/gsm/feedback.py); our protocol
        # runs every stage at self.temperature (0.0) so results are reproducible.
        # The two-call (separate feedback then refine) pattern and the natural-language
        # answer format (vs upstream GSM's PAL `def solution()` code) are deliberate,
        # faithful to the commongen/responsegen variant and compatible with our text-QA
        # + LLM judge.
        steps = []
        answer = await self.client.chat_text(
            render("baselines/self_refine_init", question=question),
            BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        steps.append({"stage": "init", "answer": answer})
        for i in range(max_iters):
            feedback = await self.client.chat_text(
                render("baselines/self_refine_feedback", question=question, answer=answer),
                BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
            )
            steps.append({"stage": "feedback", "iter": i, "feedback": feedback})
            # Upstream GSM stops when feedback contains "it is correct"; we ground the
            # equivalent NO_ISSUES sentinel in the few-shot feedback prompt so a 7B
            # model reliably emits it.
            if "NO_ISSUES" in feedback.upper():
                break
            answer = await self.client.chat_text(
                render("baselines/self_refine_refine", question=question, answer=answer, feedback=feedback),
                BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
            )
            steps.append({"stage": "refine", "iter": i, "answer": answer})
        return BaselineResult(answer=answer, trace={"steps": steps})
