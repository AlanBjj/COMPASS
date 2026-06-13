"""Rowen (Ding et al. 2024): cross-lingual consistency-based hallucination detection with
conditional retrieval-augmented repair.

Faithful to the official repo (github.com/jiangshdd/Rowen @ commit 7fd0454), specifically the
TruthfulQA `reduce_hallucination_pipleline` in run_truthfulqa.py and prompt_template.py.

Pipeline (mode="language", the paper's signature cross-language detector):
  1. Two-stage CoT on the question: a long evaluative CoT answer (TRUTHFULQA_INITIAL_ANSWER_TEMPLATE)
     followed by a SECOND turn that extracts a short, direct answer (TRUTHFULQA_FINAL_ANSWER_TEMPLATE).
     Keep both `original_long_answer` and `original_short_answer`.
  2. Generate k semantically-equivalent perturbations of the question in ONE call at
     temperature 1.0 (SEMANTICALLY_EQUIVALENT_PERTERBATIONS_TEMPLATE), strip "N." prefixes.
  3. CROSS-LANGUAGE consistency: for each perturbation, (a) answer it in EN (short answer),
     (b) translate the perturbation to Chinese with the same backbone, (c) answer the ZH
     question with a Chinese CoT, (d) judge EN-answer vs ZH-answer equivalence with the official
     CROSS_CONSISTENCY_CHECK_TEMPLATE (True/False). score = (#True) / k, a fraction in [0, 1].
  4. If score >= threshold (default 0.5) -> keep the original short answer (NO retrieval).
     Otherwise retrieve evidence over the perturbed queries and repair.
  5. Repair: Serper-search each perturbed query (keep ~floor(k/2) snippets each), concatenate as
     evidence, then ONE repair call conditioned on question + long + short + evidence
     (TRUTHFULQA_REPAIR_HALLUCINATION_TEMPLATE).

Deviation from official (annotated in the trace): the official `hybrid`/`model` modes add a
cross-MODEL consistency axis using a second model (qwen-max). With a single open-source backbone
that axis is not reproducible, so we run mode="language" (alpha=0) and DO NOT fake cross-model by
re-sampling the same model. The official `is_supported` True/False parser is ported verbatim.
All hyperparameters come from config."""

from __future__ import annotations

import asyncio
import re
import string

from ..llm import render
from ..llm.client import BACKBONE, build_messages
from .base import Baseline, BaselineResult


def _remove_prefix(text: str) -> str:
    """Port of official utils.remove_prefix: strip a leading 'N. ' enumeration prefix."""
    return re.sub(r"^\d+\.\s", "", text)


def _is_supported(generated_answer: str) -> bool:
    """Port of official utils.is_supported: parse a True/False judgement leniently."""
    generated_answer = generated_answer.lower()
    if "true" in generated_answer or "false" in generated_answer:
        if "true" in generated_answer and "false" not in generated_answer:
            return True
        elif "false" in generated_answer and "true" not in generated_answer:
            return False
        else:
            return generated_answer.index("true") > generated_answer.index("false")
    return all(
        keyword
        not in generated_answer.translate(str.maketrans("", "", string.punctuation)).split()
        for keyword in ["not", "cannot", "unknown", "information"]
    )


class RowenBaseline(Baseline):
    name = "rowen"

    # Official Rowen ships per-DATASET answer templates and only covers TruthfulQA & StrategyQA
    # (run_truthfulqa.py / run_strategyqa.py). We route by dataset so each gets the right template
    # instead of forcing the TruthfulQA "evaluate the validity of the assumption" prompt onto all
    # (audit HIGH-7). truthfulqa/freshqa -> the official TruthfulQA templates (closest fit; FreshQA
    # is an EXTENSION — Rowen has no FreshQA template, so we use the factual TruthfulQA-style one).
    # strategyqa -> the official StrategyQA templates (True/False format). gsm8k -> a neutral CoT
    # set (EXTENSION — Rowen has no math template; the premise-checking prompt is wrong for math).
    _DS_PREFIX = {"truthfulqa": "tqa", "freshqa": "tqa", "strategyqa": "sqa", "gsm8k": "gen"}
    _TQA_NAMES = {"initial": "rowen_initial_answer", "final": "rowen_final_answer", "repair": "rowen_repair"}

    def _pname(self, kind: str, *, zh: bool = False) -> str:
        prefix = self._DS_PREFIX.get(self.dataset, "tqa")
        base = self._TQA_NAMES[kind] if prefix == "tqa" else f"rowen_{prefix}_{kind}"
        if zh and kind in ("initial", "final"):
            base += "_zh"
        return "baselines/" + base

    async def _cot(self, question: str, *, use_chinese: bool = False) -> tuple[str, str]:
        """Two-stage CoT: long evaluative answer, then a short direct answer in a second turn.
        Mirrors the official (a)synchronous chain_of_thought_reasoning multi-turn structure."""
        init_prompt = render(self._pname("initial", zh=use_chinese), question=question)
        long_answer = await self.client.chat_text(
            init_prompt, BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        final_prompt = render(self._pname("final", zh=use_chinese), question=question)
        # Second turn carries the long answer as assistant context (official multi-turn message list).
        messages = build_messages(init_prompt)
        messages.append({"role": "assistant", "content": long_answer})
        messages.append({"role": "user", "content": final_prompt})
        short_answer = await self.client.chat(
            messages, BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return long_answer, short_answer

    async def answer(self, question: str) -> BaselineResult:
        k = int(self.params.get("num_paraphrases", 6))
        threshold = float(self.params.get("consistency_threshold", 0.5))  # fraction in [0,1]
        pert_temp = float(self.params.get("perturbation_temperature", 1.0))
        mode = str(self.params.get("mode", "language"))
        top_k = int(self.params.get("top_k", 5))
        trace: dict = {"mode": mode, "cross_model_axis": "omitted (single backbone)"}

        # 1. Two-stage CoT on the original question.
        original_long_answer, original_short_answer = await self._cot(question)
        trace["original_long_answer"] = original_long_answer
        trace["original_short_answer"] = original_short_answer

        # 2. Semantically-equivalent perturbations (one call, temperature 1.0).
        pert_block = await self.client.chat_text(
            render("baselines/rowen_perturbations", question=question, k=k),
            BACKBONE, temperature=pert_temp, max_tokens=self.max_tokens,
        )
        perturbated_queries = [
            _remove_prefix(line).strip()
            for line in pert_block.split("\n")
            if line.strip()
        ][:k]
        trace["perturbated_queries"] = perturbated_queries

        if not perturbated_queries:
            trace["retrieved"] = False
            trace["consistency_score"] = None
            trace["note"] = "no perturbations parsed; keeping original short answer"
            return BaselineResult(answer=original_short_answer, trace=trace)

        # 3. Cross-language consistency: EN answer vs ZH answer per perturbation.
        chinese_perturbated_queries = await asyncio.gather(*[
            self.client.chat_text(
                render("baselines/rowen_translate", text=q),
                BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
            )
            for q in perturbated_queries
        ])
        # EN short answers and ZH short answers (each via two-stage CoT), run concurrently.
        en_cots = await asyncio.gather(*[self._cot(q) for q in perturbated_queries])
        zh_cots = await asyncio.gather(*[
            self._cot(zh, use_chinese=True) for zh in chinese_perturbated_queries
        ])
        perturbated_answers = [short for _, short in en_cots]
        target_perturbated_answers = [short for _, short in zh_cots]
        trace["chinese_perturbated_queries"] = chinese_perturbated_queries
        trace["perturbated_answers"] = perturbated_answers
        trace["target_perturbated_answers"] = target_perturbated_answers

        check_raw = await asyncio.gather(*[
            self.client.chat_text(
                render("baselines/rowen_cross_consistency", q=q, a1=en, a2=zh),
                BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
            )
            for q, en, zh in zip(perturbated_queries, perturbated_answers, target_perturbated_answers)
        ])
        consistency_flags = [_is_supported(out.lower()) for out in check_raw]
        consistency_score = sum(consistency_flags) * 1.0 / len(consistency_flags)
        trace["consistency_flags"] = consistency_flags
        trace["consistency_score"] = consistency_score  # fraction in [0,1]

        # 4. Trigger: high consistency -> keep original short answer, no retrieval.
        if consistency_score >= threshold:
            trace["retrieved"] = False
            trace["retrieve_decision"] = f"score {consistency_score:.3f} >= threshold {threshold} -> keep original"
            return BaselineResult(answer=original_short_answer, trace=trace)

        trace["retrieve_decision"] = f"score {consistency_score:.3f} < threshold {threshold} -> retrieve + repair"

        if self.retriever is None:
            trace["retrieved"] = False
            trace["note"] = "retriever is None; cannot repair, returning original short answer"
            return BaselineResult(answer=original_short_answer, trace=trace)

        # 5. Retrieve over perturbed queries and repair.
        from .retrieval import web_search  # local import: optional retrieval dependency

        per_query_snippets = await asyncio.gather(*[
            web_search(self.retriever, q, top_k=top_k) for q in perturbated_queries
        ])
        keep = max(1, k // 2)  # official keeps ~floor(k/2) snippets per query
        retrieved_evidences = []
        for q, snippets in zip(perturbated_queries, per_query_snippets):
            for snippet in snippets[:keep]:
                retrieved_evidences.append(q.rstrip("?") + "? " + snippet)
        trace["retrieved_evidences"] = retrieved_evidences

        repaired_answer = await self.client.chat_text(
            render(
                self._pname("repair"),
                question=question,
                initial_long_answer=original_long_answer,
                initial_short_answer=original_short_answer,
                evidences="\n".join(retrieved_evidences),
            ),
            BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        trace["retrieved"] = True
        trace["repaired_answer"] = repaired_answer
        return BaselineResult(answer=repaired_answer, trace=trace)
