"""DTR (Chen et al., arXiv 2601.03908, 2026): Decide Then Retrieve.

Faithful to the paper and its released code (github.com/ChenWangHKU/DTR, main.py):
  1. Decide: answer once without retrieval (greedy). Uncertainty u = -(1/T) sum_t log p(a_t | q, a_<t)
     over the answer tokens, dropping the final token (the end-of-turn token, as in the code).
  2. If u <= tau, return that initial answer (no second generation).
  3. Otherwise, dual-path retrieval: the model writes a pseudo-context passage for the question;
     the question and the passage each retrieve top-n (n = 5) candidates, pooled (2n at most).
  4. Adaptive information selection (AIS): with s1 / s2 the cosine similarity of a candidate to the
     question / the pseudo-context, score s = cos(theta1 + theta2) = s1*s2 - sqrt(1-s1^2)*sqrt(1-s2^2),
     keep the top-k (k = 3), and answer with them.
     (The paper's formula. The released code has it commented out and ranks by s1 + s2 instead;
     `ais: sum` selects that variant.)

Adaptations (documented, not changes to the method): the answer prompts are the same zero-shot CoT
prompt as our Direct baseline and the shared answer-with-evidence prompt (the paper asks for a single
word or phrase, which does not fit TruthfulQA/GSM8K); each path's candidates come from the shared
local snapshot (BM25) instead of a dense index; the similarities s1/s2 use bge-m3 (the paper uses
bge-large-en-v1.5, not available offline) served by vLLM's embedding endpoint. The pseudo-context
query is capped at `pseudo_query_chars` (default 2000, roughly the 512-token window of the paper's
encoder) instead of the retriever's usual 400-char cap.

Calibration: `force: never|always` run the two branches unconditionally while recording u, so
scripts/calibrate_gate.py can compose the gated method for any tau from two runs (it retrieves
iff u > tau, the same rule as here).
"""

from __future__ import annotations

import math
from typing import List, Optional

from openai import AsyncOpenAI

from ..llm import render
from ..llm.client import BACKBONE
from .base import Baseline, BaselineResult
from .retrieval import LocalPlainRetriever, format_snippets, web_search


def mean_nll(tokens) -> float:
    """-(1/T) sum log p over the answer tokens, dropping the final (end-of-turn) token."""
    lps = [lp for _, lp, _ in tokens]
    if len(lps) > 1:
        lps = lps[:-1]
    return -sum(lps) / len(lps) if lps else 0.0


def ais_score(s1: float, s2: float, mode: str = "angle") -> float:
    s1, s2 = max(-1.0, min(1.0, s1)), max(-1.0, min(1.0, s2))
    if mode == "sum":
        return s1 + s2
    return s1 * s2 - math.sqrt(1 - s1 * s1) * math.sqrt(1 - s2 * s2)


class DtrBaseline(Baseline):
    name = "dtr"
    _embed: Optional[AsyncOpenAI] = None

    async def _embeddings(self, texts: List[str]) -> List[List[float]]:
        if self._embed is None:
            self._embed = AsyncOpenAI(base_url=str(self.params.get("embed_url", "http://localhost:8300/v1")),
                                      api_key="EMPTY", timeout=120.0)
        resp = await self._embed.embeddings.create(model=str(self.params.get("embed_model", "bge-m3")),
                                                   input=texts)
        vecs = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        return [[x / (math.sqrt(sum(v * v for v in vec)) or 1.0) for x in vec] for vec in vecs]

    async def _candidates(self, query: str, n: int, max_chars: Optional[int] = None) -> List[str]:
        if max_chars and isinstance(self.retriever, LocalPlainRetriever):
            evs = await self.retriever.search(query, max_chars=max_chars)
            return [s for ev in evs for s in ev.snippets][:n]
        return await web_search(self.retriever, query, top_k=n)

    async def answer(self, question: str) -> BaselineResult:
        n = int(self.params.get("top_n", 5))
        k = int(self.params.get("top_k", 3))
        mode = str(self.params.get("ais", "angle"))
        force = self.params.get("force")          # None | "never" | "always"
        tau = self.params.get("tau")
        if force is None and tau is None:
            raise ValueError("dtr needs a calibrated tau (or force: never|always for calibration)")

        base_prompt = render("baselines/direct", question=question)
        initial, tokens = await self.client.chat_text_logprobs(
            base_prompt, BACKBONE, temperature=0.0, max_tokens=self.max_tokens)
        u = mean_nll(tokens)
        retrieve = force == "always" or (force is None and u > float(tau))
        trace = {"uncertainty": u, "retrieved": bool(retrieve), "tau": tau, "force": force,
                 "answer_tokens": len(tokens)}
        if not retrieve or self.retriever is None:
            return BaselineResult(answer=initial, trace=trace)

        pseudo = await self.client.chat_text(render("baselines/dtr_pseudo_context", question=question),
                                             BACKBONE, temperature=0.0, max_tokens=self.max_tokens)
        from_q = await self._candidates(question, n)
        from_p = await self._candidates(pseudo, n, int(self.params.get("pseudo_query_chars", 2000)))
        pool: List[str] = []
        for s in from_q + from_p:
            if s not in pool:
                pool.append(s)
        trace.update({"pseudo_context": pseudo, "n_candidates": len(pool)})
        if not pool:
            # Nothing retrieved on either path: keep the initial answer (no evidence to add).
            trace["n_snippets"] = 0
            return BaselineResult(answer=initial, trace=trace)

        vecs = await self._embeddings([question, pseudo] + pool)
        qv, pv, dvs = vecs[0], vecs[1], vecs[2:]
        scored = []
        for s, dv in zip(pool, dvs):
            s1 = sum(a * b for a, b in zip(dv, qv))
            s2 = sum(a * b for a, b in zip(dv, pv))
            scored.append((ais_score(s1, s2, mode), s))
        scored.sort(key=lambda x: -x[0])
        snippets = [s for _, s in scored[:k]]
        prompt = render("baselines/answer_with_evidence", question=question,
                        evidence=format_snippets(snippets))
        text = await self.client.chat_text(prompt, BACKBONE, temperature=self.temperature,
                                           max_tokens=self.max_tokens)
        trace.update({"initial_answer": initial, "n_snippets": len(snippets),
                      "ais_scores": [round(x, 4) for x, _ in scored[:k]],
                      "from_query": sum(1 for s in snippets if s in from_q)})
        return BaselineResult(answer=text, trace=trace)
