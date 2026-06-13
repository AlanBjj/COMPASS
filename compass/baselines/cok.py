"""Chain-of-Knowledge (Li et al. 2024) — faithful port to one backbone + cached web retrieval.

Faithful to the official repo (CoK @ dfb85cf). The defining mechanism of CoK is a
self-consistency (SC) GATE that decides whether knowledge adapting is needed at all, plus a
TWO-RATIONALE, STEP-BY-STEP edit loop. Concretely:

  Stage I  — reasoning preparation:
             (a) domain selection (one call; kept for fidelity even though all domains
                 collapse to a single web retriever here),
             (b) CoT WITH SELF-CONSISTENCY: sample `sc_n` rationales at `sc_temperature`,
                 keep the first `sc_keep`, extract the text after the marker "The answer is",
                 majority-vote -> cot_sc_answer, sc_score = count(majority)/kept. The first
                 sample whose answer == majority supplies rationale_1 / rationale_2
                 (split on "First," / "Second," / "The answer is").

  SC GATE  — if sc_score >= threshold, RETURN the SC answer directly (no retrieval, no edit).
             Only when sc_score < threshold do we run Stage II/III.

  Stage II — dynamic knowledge adapting, STEP BY STEP:
             generate a search query FROM rationale_1 (verify-question prompt: "write a
             question that asks about the answer to the overall question", rationale as the
             Answer), web_search it, then EDIT rationale_1 with the sentence-level edit prompt
             (Sentence/Knowledge/Edited sentence). REGENERATE rationale_2 conditioned on the
             edited rationale_1, then retrieve + edit rationale_2 the same way.

  Stage III— consolidation: feed edited rationale_1 + rationale_2 back into the reasoning
             demo with the "The answer is " tail -> final answer.

DEVIATIONS FROM OFFICIAL (annotated): the official CoK retrieves from MULTIPLE heterogeneous
sources (Wikidata SPARQL, Wikipedia, per-domain corpora) selected by the domain step; here all
domains collapse to ONE cached Serper web search (shared baseline retriever, R3-reproducible).
A single open-source backbone replaces GPT-3.5/text-davinci. The official answer marker
"The answer is" is preserved (NOT "Answer:"). When no retriever is configured, Stage II/III
degrade to self-verification editing with empty external knowledge (annotated in the trace)."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import List, Tuple

from ..llm import render
from ..llm.client import BACKBONE
from .base import Baseline, BaselineResult
from .retrieval import format_snippets, web_search

ANSWER_MARKER = "The answer is"


def _extract_answer(text: str) -> str | None:
    """Return the lowercased text after the official marker, or None if absent."""
    if ANSWER_MARKER in text:
        return text.split(ANSWER_MARKER, 1)[1].strip().rstrip(".").strip().lower()
    return None


def _split_rationales(text: str) -> Tuple[str, str]:
    """Split a 'First, ... Second, ... The answer is ...' sample into (rationale_1, rationale_2).

    Mirrors the official hotpotqa_parser split; tolerant to missing markers."""
    r1, r2 = "", ""
    if "Second, " in text:
        head, tail = text.split("Second, ", 1)
        r1 = head.split("First, ", 1)[-1].strip()
        r2 = tail.split(ANSWER_MARKER, 1)[0].strip()
    else:
        r1 = text.split("First, ", 1)[-1].split(ANSWER_MARKER, 1)[0].strip()
    return r1, r2


class CoKBaseline(Baseline):
    name = "cok"

    async def answer(self, question: str) -> BaselineResult:
        trace: dict = {}
        # Hyperparameters (config-driven; report keys+defaults).
        threshold = float(self.params.get("threshold", 0.5))
        sc_n = int(self.params.get("sc_n", 10))
        sc_keep = int(self.params.get("sc_keep", 5))
        sc_temperature = float(self.params.get("sc_temperature", 0.7))
        top_k = int(self.params.get("top_k", 5))
        edit_max_tokens = int(self.params.get("edit_max_tokens", 256))

        # ---- Stage I(a): domain selection (kept for fidelity; collapses to one retriever). ----
        domains = await self.client.chat_text(
            render("baselines/cok_domain", question=question),
            BACKBONE, system="You are a helpful assistant.",
            temperature=0.0, max_tokens=256,
        )
        domains = domains.strip().splitlines()[0].strip() if domains.strip() else ""
        trace["domains"] = domains

        # ---- Stage I(b): CoT with self-consistency. ----
        cot_prompt = render("baselines/cok_reason", question=question, tail="")
        samples: List[str] = await asyncio.gather(
            *[
                self.client.chat_text(
                    cot_prompt, BACKBONE, system="You are a helpful assistant.",
                    temperature=sc_temperature, max_tokens=self.max_tokens,
                )
                for _ in range(sc_n)
            ]
        )

        # Keep the first sc_keep samples that contain a parseable answer (official: take [:5]).
        parsed: List[Tuple[str, str]] = []  # (full_sample_text, lowercased_answer)
        for s in samples:
            ans = _extract_answer(s)
            if ans is not None:
                parsed.append((s, ans))
            if len(parsed) >= sc_keep:
                break

        trace["sc_n"] = sc_n
        trace["sc_keep"] = sc_keep
        trace["sc_kept"] = len(parsed)
        trace["sc_threshold"] = threshold

        if not parsed:
            # No sample produced the marker; fall back to the first raw sample as the answer.
            trace["sc_gate"] = "no_parseable_answer"
            answer = samples[0].strip() if samples else ""
            trace["final"] = answer
            return BaselineResult(answer=answer, trace=trace)

        answers = [a for _, a in parsed]
        counts = Counter(answers)
        majority_answer, majority_count = counts.most_common(1)[0]
        sc_score = majority_count / len(parsed)
        trace["sc_score"] = sc_score
        trace["cot_sc_answer"] = majority_answer

        # rationales come from the first sample whose answer == majority.
        sc_sample = next(s for s, a in parsed if a == majority_answer)
        rationale_1, rationale_2 = _split_rationales(sc_sample)
        trace["cot_sc_rationales"] = [rationale_1, rationale_2]

        # ---- SC GATE: high consistency -> answer directly, no retrieval/editing. ----
        if sc_score >= threshold:
            trace["sc_gate"] = "skip_ka (sc_score >= threshold)"
            trace["final"] = majority_answer
            return BaselineResult(answer=majority_answer, trace=trace)

        trace["sc_gate"] = "run_ka (sc_score < threshold)"

        # ---- Stage II: dynamic knowledge adapting, step by step. ----
        if self.retriever is None:
            trace["retrieval"] = "(no retriever configured; self-verification edit, no external knowledge)"

        # rationale_1: make query from rationale, retrieve, edit.
        edited_r1, k1 = await self._edit_rationale(question, rationale_1, top_k, edit_max_tokens)
        trace["rationale_1_knowledge"] = k1
        trace["edited_rationale_1"] = edited_r1

        # Regenerate rationale_2 conditioned on edited rationale_1, strip the answer tail.
        new_r2_raw = await self.client.chat_text(
            render("baselines/cok_reason", question=question, tail=f"First, {edited_r1} Second, "),
            BACKBONE, system="You are a helpful assistant.",
            temperature=0.0, max_tokens=self.max_tokens,
        )
        new_rationale_2 = new_r2_raw.split(ANSWER_MARKER, 1)[0].strip()
        trace["new_rationale_2"] = new_rationale_2

        # rationale_2: make query, retrieve, edit.
        edited_r2, k2 = await self._edit_rationale(question, new_rationale_2, top_k, edit_max_tokens)
        trace["rationale_2_knowledge"] = k2
        trace["edited_rationale_2"] = edited_r2

        # ---- Stage III: answer consolidation. ----
        consolidation = await self.client.chat_text(
            render(
                "baselines/cok_reason",
                question=question,
                tail=f"First, {edited_r1} Second, {edited_r2} {ANSWER_MARKER} ",
            ),
            BACKBONE, system="You are a helpful assistant.",
            temperature=0.0, max_tokens=edit_max_tokens,
        )
        final = consolidation.strip()
        # The completion continues after "The answer is "; keep just that segment if echoed.
        if ANSWER_MARKER in final:
            final = final.split(ANSWER_MARKER, 1)[1].strip()
        final = final.split("\n", 1)[0].strip().rstrip(".").strip()
        trace["final"] = final
        return BaselineResult(answer=final, trace=trace)

    async def _edit_rationale(
        self, question: str, rationale: str, top_k: int, edit_max_tokens: int
    ) -> Tuple[str, str]:
        """Generate a query from the rationale, retrieve knowledge, and sentence-edit it.

        Returns (edited_rationale, knowledge_text)."""
        knowledge = ""
        if self.retriever is not None and rationale:
            query = await self.client.chat_text(
                render("baselines/cok_make_query", question=question, answer=rationale),
                BACKBONE, system="You are a helpful assistant.",
                temperature=0.0, max_tokens=64,
            )
            query = query.strip().splitlines()[0].strip() if query.strip() else rationale
            snippets = await web_search(self.retriever, query, top_k=top_k)
            knowledge = " ".join(snippets) if snippets else ""

        edited = await self.client.chat_text(
            render("baselines/cok_edit", sentence=rationale, knowledge=knowledge),
            BACKBONE, system="You are a helpful assistant.",
            temperature=0.0, max_tokens=edit_max_tokens,
        )
        edited = edited.strip().split("\n", 1)[0].strip()
        # If the model echoed the scaffold, keep the text after the marker.
        if "Edited sentence:" in edited:
            edited = edited.split("Edited sentence:", 1)[1].strip()
        return (edited or rationale), (knowledge or "(no external knowledge)")
