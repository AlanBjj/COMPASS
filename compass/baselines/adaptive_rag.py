"""Adaptive-RAG (Jeong et al. 2024): a complexity classifier routes each query to the cheapest
adequate strategy — A: no retrieval (direct CoT), B: single-step retrieve-then-read, C: multi-step
iterative retrieval (IRCoT).

Faithful to the official repo (github.com/starsuzi/Adaptive-RAG @ 0c88670), with two infra swaps
that are NOT reproducible here and are annotated inline:

  * Complexity classifier. The official method trains a t5-large classifier on SILVER labels
    (derived from whether single-/multi-/no-retrieval pipelines answer correctly on
    HotpotQA/MuSiQue/NQ/etc.). That trained model is not reproducible in this environment, so we
    use the standard TRAINING-FREE adaptation: prompt the backbone LM itself to emit the A/B/C
    label, with a few-shot prompt tightened to the silver-label semantics. A 7B zero-shot router
    is noisier than the trained classifier, hence the few-shot examples and the default-to-B on
    unparseable output.
  * Retrieval. The official method uses Elasticsearch BM25 over a fixed Wikipedia / HotpotQA
    corpus. Here we reuse the shared cached Serper open-web retriever (see retrieval.py).

The IRCoT loop (class C) follows StepByStepCOTGenParticipant in commaqa/inference/ircot.py:
generate ONE reasoning sentence per step, then auto-retrieve using that generated sentence as the
query (query_source = "question_or_last_generated_sentence": the question on the first step, the
last generated sentence thereafter), append the snippets to the accumulated evidence, and repeat.
Termination: the generated text contains "answer is: ..." (extracted via the upstream regex
`.* answer is:? (.*)` with the trailing full-stop removed) OR after max_num_sentences steps.
There is no model-emitted SEARCH line — retrieval is driven by the generated reasoning itself.
"""

from __future__ import annotations

import re

from ..llm import render
from ..llm.client import BACKBONE
from .base import Baseline, BaselineResult
from .retrieval import format_snippets, web_search

# Upstream answer marker / extractor (commaqa/inference/ircot.py: answer_extractor_regex).
_ANSWER_RE = re.compile(r".* answer is:?\s*(.*)", flags=re.IGNORECASE)


def _split_first_sentence(text: str) -> str:
    """Keep only the first sentence of a generation (upstream uses spaCy sentence splitting to
    enforce one-sentence-per-step; we approximate with a regex on sentence-final punctuation)."""
    text = text.strip()
    if not text:
        return ""
    m = re.search(r"(.+?[.!?])(\s|$)", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text


class AdaptiveRagBaseline(Baseline):
    name = "adaptive_rag"

    async def _direct(self, question: str) -> str:
        return await self.client.chat_text(
            render("baselines/adaptive_direct", question=question),
            BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
        )

    async def _single(self, question: str) -> str:
        # Class B: single retrieve-then-read. top_k matches upstream oneR bm25_retrieval_count=5.
        top_k = int(self.params.get("single_top_k", 5))
        query = await self.client.chat_text(
            render("baselines/adaptive_make_query", question=question),
            BACKBONE, temperature=self.temperature, max_tokens=64,
        )
        q = query.strip().splitlines()[0] if query.strip() else question
        snippets = await web_search(self.retriever, q, top_k=top_k)
        return await self.client.chat_text(
            render("baselines/adaptive_answer_with_evidence",
                   question=question, evidence=format_snippets(snippets)),
            BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
        )

    async def _ircot(self, question: str, trace: dict) -> str:
        # Class C: IRCoT. max_num_sentences and ircot_top_k match upstream ircot config
        # (max_num_sentences=10, bm25_retrieval_count=6).
        max_steps = int(self.params.get("max_num_sentences", 10))
        top_k = int(self.params.get("ircot_top_k", 6))

        evidence: list[str] = []
        sentences: list[str] = []
        answer: str | None = None

        for _ in range(max_steps):
            # Retrieve BEFORE generating this step's sentence, using query_source =
            # question_or_last_generated_sentence (question on step 1, else last sentence).
            query = sentences[-1] if sentences else question
            snippets = await web_search(self.retriever, query, top_k=top_k)
            for s in snippets:
                if s not in evidence:
                    evidence.append(s)

            out = await self.client.chat_text(
                render("baselines/adaptive_ircot_step", question=question,
                       evidence=format_snippets(evidence) if evidence else "(none yet)",
                       reasoning=" ".join(sentences) if sentences else ""),
                BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
            )
            sentence = _split_first_sentence(out)
            if not sentence:
                break
            sentences.append(sentence)

            m = _ANSWER_RE.match(sentence)
            if m:
                answer = m.group(1).strip()
                if answer.endswith("."):  # upstream remove_last_fullstop=True
                    answer = answer[:-1].strip()
                break

        trace["ircot_steps"] = len(sentences)
        trace["ircot_reasoning"] = " ".join(sentences)
        trace["ircot_evidence_count"] = len(evidence)
        # Prefer the extracted answer span; fall back to the full reasoning chain.
        return answer if answer else " ".join(sentences)

    async def answer(self, question: str) -> BaselineResult:
        cls = (await self.client.chat_text(
            render("baselines/adaptive_classify", question=question),
            BACKBONE, temperature=self.temperature, max_tokens=8,
        )).strip().upper()
        # Default-to-B on unparseable output (a single retrieval is the safe middle choice).
        label = next((c for c in cls if c in "ABC"), "B")
        trace = {"class_raw": cls, "class": label}

        if self.retriever is None:
            # No retriever configured: every route degrades to direct CoT.
            trace["retriever"] = "none -> direct"
            answer = await self._direct(question)
        elif label == "A":
            answer = await self._direct(question)
        elif label == "B":
            answer = await self._single(question)
        else:
            answer = await self._ircot(question, trace)
        return BaselineResult(answer=answer, trace=trace)
