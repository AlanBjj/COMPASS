"""TARG (Wang, Wei & Ling, arXiv 2511.09803, 2025): Training-free Adaptive Retrieval Gating.

Faithful to the paper's default ("Margin" gate):
  1. Draft: greedy decoding of a short prefix (k = 20 tokens) from the base prompt, no retrieval.
  2. Score: U = mean_t phi(g_t), g_t = top-1 minus top-2 logit at step t, phi(z) = exp(-z / beta),
     beta = 3. The logit gap equals the log-probability gap (softmax is shift-invariant), so it is
     read from the top-2 logprobs vLLM returns.
  3. Gate: retrieve iff U > tau. tau is calibrated on the dev split for a target retrieval rate
     rho (tau_rho = (1 - rho)-quantile of U), as in the paper.
  4. Answer: the base prompt (no retrieval), or the base prompt plus the top-5 retrieved passages.

Adaptations (documented, not changes to the method): the base prompt is the same zero-shot CoT
prompt as our Direct baseline and the retrieval prompt the shared answer-with-evidence prompt; the
retriever is the shared local snapshot (BM25, top-5) instead of a dense encoder.

Calibration: params `force: never|always` run the two branches unconditionally while recording U,
so scripts/calibrate_gate.py can compose the gated method for any tau from two runs.
"""

from __future__ import annotations

import math

from ..llm import render
from ..llm.client import BACKBONE
from .base import Baseline, BaselineResult
from .retrieval import format_snippets, web_search


def margin_uncertainty(tokens, beta: float) -> float:
    """Mean exp(-(top1 - top2)/beta) over the draft tokens; 1.0 (most uncertain) if no token has
    two alternatives."""
    vals = [math.exp(-(alts[0] - alts[1]) / beta) for _, _, alts in tokens if len(alts) >= 2]
    return sum(vals) / len(vals) if vals else 1.0


class TargBaseline(Baseline):
    name = "targ"

    async def answer(self, question: str) -> BaselineResult:
        k = int(self.params.get("draft_tokens", 20))
        beta = float(self.params.get("beta", 3.0))
        top_k = int(self.params.get("top_k", 5))
        force = self.params.get("force")          # None | "never" | "always"
        tau = self.params.get("tau")
        if force is None and tau is None:
            raise ValueError("targ needs a calibrated tau (or force: never|always for calibration)")

        base_prompt = render("baselines/direct", question=question)
        draft, tokens = await self.client.chat_text_logprobs(
            base_prompt, BACKBONE, temperature=0.0, max_tokens=k, top_logprobs=2)
        u = margin_uncertainty(tokens, beta)
        retrieve = force == "always" or (force is None and u > float(tau))

        snippets = []
        if retrieve and self.retriever is not None:
            snippets = await web_search(self.retriever, question, top_k=top_k)
            prompt = render("baselines/answer_with_evidence", question=question,
                            evidence=format_snippets(snippets))
        else:
            prompt = base_prompt
        text = await self.client.chat_text(prompt, BACKBONE, temperature=self.temperature,
                                           max_tokens=self.max_tokens)
        return BaselineResult(answer=text, trace={
            "uncertainty": u, "retrieved": bool(retrieve), "tau": tau, "force": force,
            "draft": draft, "draft_tokens": len(tokens), "n_snippets": len(snippets)})
