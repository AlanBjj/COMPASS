"""COMPASS main orchestration — Algorithm 1.

Wires the five stages into one entry point, with the two DirectGen fallbacks from the paper's
algorithm: (1) gate says simple -> direct; (2) all sub-answers low quality -> direct. Replaces
the legacy dual-entry mess (main_experiment vs run_my_method) with a single, dataset-agnostic
pipeline. controller (gate/decompose/scoring) and backbone (answering/fusion) are the SAME
vLLM model but billed separately via the client's per-role ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import re

from .aggregation import trace_aggregate
from .decompose import FACT, MATH, SubQuery, decompose
from .decompose.decompose import _looks_arithmetic
from .fusion import fuse, is_low_quality, score_subanswers
from .gate import decide
from .gate.heuristic import _COMPARISON, _RECENCY, _matches_any
from .llm import render
from .llm.client import BACKBONE, LLMClient
from .solvers import (
    CachedRetriever,
    Retriever,
    SerperRetriever,
    SnapshotCache,
    route,
    solve_general,
    solve_math,
)

# Gate self-knowledge dimension that flags "needs external verification". When the gate
# routes a query to the parametric direct path (complexity < tau) but this dimension is at
# its MAX, the query is a low-structural-complexity fact that nonetheless depends on
# external/long-tail/time-varying knowledge (e.g. "Who invented the Perceptron?"). Such
# queries must be grounded in retrieved evidence, not answered from parameters.
_EXTERNAL_DEPENDENCY = "external_dependency"
_EXTERNAL_DEPENDENCY_MAX = 2

# Leading tokens that mark a yes/no-style question (auxiliary verbs + copulas). Such questions
# ("Were muskets used in the Pacific War?") are reasoning/verification questions, NOT fact
# lookups asking for a specific value. Whole-question pure retrieval is wrong for them: the
# retrieved snippets either mislead or come back empty, destroying the commonsense reasoning
# they need. They must fall back to normal gate routing (parametric direct CoT, or decompose).
_YES_NO_LEADING_TOKENS = frozenset(
    {
        "is", "are", "was", "were", "am", "be", "being", "been",
        "can", "could", "would", "should", "shall", "will", "may", "might", "must",
        "do", "does", "did",
        "has", "have", "had",
    }
)


def _is_yes_no_question(q: str) -> bool:
    """True if the question's leading word is an auxiliary verb or copula, i.e. it reads as a
    yes/no (polar) question rather than a wh-style fact lookup. Case-insensitive; strips leading
    punctuation/whitespace before inspecting the first token. wh-fact-lookups (who/when/where/
    which/what/how many/name/list ...) start with a content word, so they return False and stay
    eligible for direct_retrieval."""
    if not q:
        return False
    first = q.strip().split(maxsplit=1)
    if not first:
        return False
    token = first[0].strip("\"'`.,;:!?()[]{}").lower()
    return token in _YES_NO_LEADING_TOKENS


def _is_commonsense_yes_no(q: str) -> bool:
    """True for a binary-verdict (yes/no) question that turns on COMMONSENSE / world-model
    reasoning rather than a specific external or time-varying fact.

    Such questions ("Could a silverfish reach the top of the Empire State Building?", "Can a human
    survive a month without water?") are answered correctly by direct chain-of-thought over the
    model's own world model. Routing them into retrieval/decompose hurts: retrieval injects an
    "answer from the evidence" frame and, when no snippet speaks to the (often hypothetical)
    premise, the model defers to absent evidence ("the evidence does not state ...") and
    hallucinates a non-answer; decomposition slices a single binary verdict into disconnected
    sub-questions and loses the holistic judgment.

    This is a routing-SHAPE rule, not a dataset constant. We keep a yes/no question on direct CoT
    only when BOTH external-fact signals are ABSENT:
      - no recency / date-sensitive cue (_RECENCY) — a polar question about the CURRENT/LATEST
        state of the world ("Is X currently the largest ...?") genuinely needs fresh retrieval;
      - no comparison / multi-hop bridging structure (_COMPARISON) — an entity-bridging polar
        question ("Was the founder of X born in the same country as the designer of Y?") is a
        genuine multi-hop that must decompose.
    Anything with a recency cue or a bridging/comparison structure is NOT treated as commonsense
    and keeps its normal gate routing, so genuine fresh-fact and multi-hop questions are untouched.
    """
    if not _is_yes_no_question(q):
        return False
    ql = (q or "").lower()
    if _matches_any(_RECENCY, ql):
        return False
    if _matches_any(_COMPARISON, ql):
        return False
    return True


# Presupposition / misconception shapes: a "why is X <claimed-property>" loaded question, or a
# request for the single best / guaranteed / sure / proven cure-or-fix. These typically embed a
# FALSE premise (the property does not hold, or no such guaranteed remedy exists). Decomposition
# would dutifully answer each slice and so PROPAGATE the false premise; the direct+verify path,
# by contrast, lets prompts/verify.txt detect and correct the misconception / false premise. So
# when a question matches one of these shapes we keep it on direct CoT (which is always followed
# by _verify) instead of decomposing. These are general linguistic shapes, not dataset phrasings.
_PRESUPPOSITION = [
    r"^\s*why\s+(is|are|was|were|does|do|did|can|can't|cannot)\b",
    r"\b(best|guaranteed|surefire|sure[- ]?fire|proven|only)\b[^?]*\b(cure|remedy|fix|way|method|"
    r"solution|treatment)\b",
    r"\bcure\s+for\b",
]


def _is_presupposition_shaped(q: str) -> bool:
    """True if the question is misconception / false-premise shaped (see _PRESUPPOSITION). Such
    questions are better served by direct generation + verify (which can reject the premise) than
    by decomposition (which would answer the loaded sub-parts and entrench the premise)."""
    ql = (q or "").lower()
    return _matches_any(_PRESUPPOSITION, ql)


def _external_dependency(gate: Dict[str, object]) -> Optional[int]:
    """Read the gate's external_dependency raw score (0/1/2) out of trace['gate'], or None
    if the structured gate call failed to parse (raw holds an error marker instead of scores)."""
    raw = gate.get("raw")
    if not isinstance(raw, dict):
        return None
    scores = raw.get("scores")
    if not isinstance(scores, dict):
        return None
    val = scores.get(_EXTERNAL_DEPENDENCY)
    return int(val) if val is not None else None


def _homogeneous_type(subs: List[SubQuery]) -> Optional[str]:
    """Adaptive Decomposition: if every sub-query carries the SAME type, the query is
    homogeneous and should NOT be sliced — return that single type. If the sub-query types are
    mixed (heterogeneous), return None so the caller keeps the decompose -> score -> fuse path.

    Slicing a homogeneous query (e.g. a pure multi-step arithmetic problem decomposed into all
    Math sub-questions) breaks the connected reasoning chain; such queries are better answered by
    one whole-question expert call. Heterogeneous queries (e.g. Fact+Math) genuinely benefit
    from per-type routing and fusion.
    """
    types = {s.type for s in subs}
    return next(iter(types)) if len(types) == 1 else None


@dataclass
class CompassResult:
    answer: str
    path: str                         # "direct" | "direct_retrieval" | "decompose" | "homogeneous_<type>" | "low_quality_fallback"
    trace: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"answer": self.answer, "path": self.path, "trace": self.trace}


class Compass:
    """Runs the COMPASS pipeline for a single query. Stateless across queries except the
    client's token ledger (which accumulates cost over the whole run)."""

    def __init__(self, client: LLMClient, config: dict, retriever: Optional[Retriever] = None):
        self.client = client
        self.cfg = config
        self.retriever = retriever
        self.ctrl = config["models"]["controller"]   # temperature / max_tokens
        self.back = config["models"]["backbone"]
        self.gate_cfg = config["gate"]
        self.dec_cfg = config["decompose"]
        self.sc_cfg = config["scoring"]
        # TRACE (Type-Routed Adaptive Consistency) — optional. Absent / enabled:false -> single
        # sample everywhere (unchanged behaviour). All knobs (k, temperature, per-type flags) live
        # in configs/<run>.yaml under `trace:`; nothing is hardcoded here.
        self.trace_cfg = config.get("trace") or {}

    async def _direct(self, q: str) -> str:
        return await solve_general(
            q, q, self.client, temperature=self.back["temperature"], max_tokens=self.back["max_tokens"]
        )

    async def _verify(self, q: str, draft: str) -> str:
        """Lightweight ONE-PASS self-verification (R6 robustness). Applied to EVERY path —
        direct / homogeneous_* / direct_retrieval AND the decompose/fuse path. (Earlier the
        decompose path was skipped on the assumption that its internal consistency check covered
        the same ground; in practice the consistency/fusion stage has no misconception or
        false-premise defense, so a decompose-routed misconception trap passed through unchecked.
        verify.txt defaults to keep-draft, so adding it here cannot degrade good fused answers and
        restores the misconception/false-premise correction on this path too.) This is a single
        backbone critique-and-revise call that defaults to keeping the draft and only edits it on a
        clear, general failure mode (misconception trap, yes/no verdict-vs-reasoning mismatch,
        off-question/self-contradiction). It deliberately avoids rowen-style multi-sampling to
        preserve the cost advantage: exactly one extra backbone call, never more.

        Guardrails are enforced by prompts/verify.txt (keep numbers verbatim, no spurious yes/no
        prefix, no passive abstention). If the call fails or comes back empty, the original draft
        is returned unchanged so verification can never make an answer worse."""
        prompt = render("verify", question=q, draft=draft)
        try:
            revised = await self.client.chat_text(
                prompt, BACKBONE,
                temperature=self.back["temperature"], max_tokens=self.back["max_tokens"],
            )
        except Exception:  # noqa: BLE001 - verification must never break the pipeline
            return draft
        return revised if revised.strip() else draft

    async def _direct_retrieval(self, q: str) -> str:
        """ED-driven whole-question RAG: send the entire question (not a sub-query) to the Fact
        expert. route() applies the configured retriever and the numeric-leak guard, and falls
        back to direct generation if no retriever is configured."""
        return await route(
            SubQuery(question=q, type=FACT), self.client, main_q=q, retriever=self.retriever,
            temperature=self.back["temperature"], max_tokens=self.back["max_tokens"],
        )

    def _trace_qtype(self, q: str, path: str, homo: Optional[str] = None) -> Optional[str]:
        """Decide which COMPASS type drives TRACE aggregation for a finished answer, or None if the
        path is out of TRACE v1 scope.

        v1 scope (config-gated per type, see configs `trace:`):
          * MATH — math-type answers: a homogeneous_Math route, or any direct/decompose answer to an
            arithmetic question. -> math operator (number majority vote).
          * REASONING — non-retrieval General/Logic answers: direct / decompose / low_quality
            fallback answers that are NOT arithmetic and NOT fact-routed. -> reasoning operator
            (yes/no verdict vote, or open-ended stance vote).
        Out of scope in v1 (single sample): the retrieval / fact paths (direct_retrieval, and any
        homogeneous_Fact route) — TRACE is not applied there.
        """
        if not self.trace_cfg.get("enabled"):
            return None
        # Retrieval / fact paths are out of scope for v1.
        if path == "direct_retrieval" or path == f"homogeneous_{FACT}":
            return None

        math_on = bool(self.trace_cfg.get("math", True))
        reasoning_on = bool(self.trace_cfg.get("reasoning", True))

        # A homogeneous route already carries its single type.
        if homo is not None:
            if homo == MATH:
                return MATH if math_on else None
            # General / Logic homogeneous route -> reasoning operator.
            return homo if reasoning_on else None

        # direct / decompose / low_quality_fallback: classify by arithmetic shape of the question.
        if _looks_arithmetic(q):
            return MATH if math_on else None
        return "General" if reasoning_on else None  # any non-math type -> reasoning operator

    async def _maybe_trace(
        self, q: str, base_answer: str, path: str, trace: Dict[str, object],
        homo: Optional[str] = None,
    ) -> str:
        """Apply TRACE after the base answer if the path/type is in scope; otherwise return the base
        answer unchanged. Records what fired (operator, sample count, votes) into the run trace."""
        qtype = self._trace_qtype(q, path, homo=homo)
        if qtype is None:
            return base_answer

        # Bind the solver that produced this answer, at sampling temperature, into a no-arg closure.
        sample_temp = float(self.trace_cfg.get("temperature", 0.7))

        if qtype == MATH:
            async def solver_fn() -> str:
                return await solve_math(
                    q, q, self.client, temperature=sample_temp, max_tokens=self.back["max_tokens"],
                )
        else:
            async def solver_fn() -> str:
                return await solve_general(
                    q, q, self.client, temperature=sample_temp, max_tokens=self.back["max_tokens"],
                )

        result = await trace_aggregate(
            q, base_answer, qtype, solver_fn, self.client, self.trace_cfg
        )
        trace["trace_consistency"] = result.as_dict()
        return result.answer

    async def answer(self, q: str) -> CompassResult:
        trace: Dict[str, object] = {}

        # Step 1 — Hybrid Decomposition Gate
        decision = await decide(
            q, self.client,
            alpha=self.gate_cfg["alpha"], tau=self.gate_cfg["tau"],
            controller_temperature=self.ctrl["temperature"],
            controller_max_tokens=self.ctrl["max_tokens"],
        )
        gate = decision.as_dict()
        trace["gate"] = gate
        if not decision.decompose:
            # Decouple retrieval from structural complexity (Self-RAG / Adaptive-RAG style):
            # the gate says "simple" (don't decompose), but if its self-knowledge dimension
            # external_dependency is at MAX, the question needs external verification. Route the
            # WHOLE question to the Fact/retrieval expert instead of answering from parameters.
            # ED 0/1 -> normal parametric direct; math/reasoning (ED=0) stay parametric/CoT.
            #
            # But whole-question pure retrieval is only appropriate for FACT-LOOKUP questions
            # (wh-style, asking for a specific value). yes/no-style (polar) questions are
            # reasoning/verification tasks: retrieval mangles or empties the evidence and
            # destroys the commonsense reasoning they need, so they must NOT take this shortcut
            # — they fall back to normal gate routing (parametric direct CoT here, since the
            # gate already said "don't decompose").
            if (
                _external_dependency(gate) == _EXTERNAL_DEPENDENCY_MAX
                and not _is_yes_no_question(q)
            ):
                draft = await self._direct_retrieval(q)
                return CompassResult(await self._verify(q, draft), "direct_retrieval", trace)
            base = await self._verify(q, await self._direct(q))
            return CompassResult(await self._maybe_trace(q, base, "direct", trace), "direct", trace)

        # Routing-shape override (applies only when the gate WANTED to decompose). Two general
        # question shapes are better answered by direct CoT + verify than by decomposition:
        #   (1) commonsense yes/no questions — a single binary verdict over the model's world
        #       model. Decomposing slices the verdict into disconnected parts; routing the parts
        #       through the Fact/retrieval expert injects an "answer from evidence" frame that, on
        #       these (often hypothetical) premises, has no matching snippet and makes the model
        #       defer to absent evidence and hallucinate a non-answer. Direct CoT answers them
        #       holistically; _verify still corrects any verdict/reasoning mismatch.
        #   (2) presupposition / misconception-shaped questions — decomposition would answer the
        #       loaded sub-parts and entrench a false premise, whereas direct+verify lets
        #       verify.txt reject the premise / correct the misconception.
        # Both are routing-SHAPE rules, not dataset constants; genuine multi-hop (comparison /
        # entity-bridging) and genuine fresh-fact (recency-cued) yes/no questions are explicitly
        # excluded by _is_commonsense_yes_no, so they still decompose / retrieve as before.
        if _is_commonsense_yes_no(q) or _is_presupposition_shaped(q):
            base = await self._verify(q, await self._direct(q))
            return CompassResult(await self._maybe_trace(q, base, "direct", trace), "direct", trace)

        # Step 2 — Sub-queries Decomposition
        subs: List[SubQuery] = await decompose(
            q, self.client,
            min_subqueries=self.dec_cfg["min_subqueries"],
            max_subqueries=self.dec_cfg["max_subqueries"],
            temperature=self.ctrl["temperature"], max_tokens=self.ctrl["max_tokens"],
        )
        trace["subqueries"] = [s.as_dict() for s in subs]

        # Step 2.5 — Adaptive Decomposition (paper-faithful refinement of "decompose complex
        # queries"): decompose only HETEROGENEOUS complex queries. If the decomposition is
        # homogeneous (all sub-queries share one type), route the WHOLE original question to that
        # one type's expert in a single call — preserving the connected reasoning chain instead of
        # slicing it. Heterogeneous (mixed-type) queries fall through to the per-type
        # solve -> score -> fuse path below, unchanged.
        homo = _homogeneous_type(subs)
        if homo is not None:
            whole = SubQuery(question=q, type=homo)
            answer = await route(
                whole, self.client, main_q=q, retriever=self.retriever,
                temperature=self.back["temperature"], max_tokens=self.back["max_tokens"],
            )
            base = await self._verify(q, answer)
            path = f"homogeneous_{homo}"
            return CompassResult(await self._maybe_trace(q, base, path, trace, homo=homo), path, trace)

        # Step 3 — Type-Aware Answering
        pairs: List[Tuple[SubQuery, str]] = []
        for sq in subs:
            a = await route(
                sq, self.client, main_q=q, retriever=self.retriever,
                temperature=self.back["temperature"], max_tokens=self.back["max_tokens"],
            )
            pairs.append((sq, a))

        # Step 4 — Sub-answer Quality Scoring (+ low-quality fallback)
        scored = await score_subanswers(
            q, pairs, self.client,
            acc_weight=self.sc_cfg["acc_weight"], rel_weight=self.sc_cfg["rel_weight"],
            temperature=self.ctrl["temperature"], max_tokens=self.ctrl["max_tokens"],
        )
        trace["scored"] = [s.as_dict() for s in scored]
        if is_low_quality(scored, self.sc_cfg["low_quality_threshold"]):
            base = await self._direct(q)
            return CompassResult(
                await self._maybe_trace(q, base, "low_quality_fallback", trace),
                "low_quality_fallback", trace,
            )

        # Step 5 — Confidence-Aware Fusion
        answer = await fuse(
            q, scored, self.client,
            temperature=self.back["temperature"], max_tokens=self.back["max_tokens"],
        )
        base = await self._verify(q, answer)
        return CompassResult(await self._maybe_trace(q, base, "decompose", trace), "decompose", trace)


def build_compass(config: dict) -> Compass:
    """Construct a Compass from a parsed config dict. The controller and backbone share one
    vLLM endpoint/model; the RAG retriever (live Serper + snapshot cache) is built only if a
    `rag` section is present."""
    m = config["models"]["controller"]  # controller and backbone share endpoint+model
    client = LLMClient(
        base_url=m["base_url"], model=m["model"],
        default_temperature=m.get("temperature", 0.0),
        default_max_tokens=m.get("max_tokens", 1024),
        concurrency=config.get("client", {}).get("concurrency", 8),
        retries=config.get("client", {}).get("retries", 3),
    )
    retriever: Optional[Retriever] = None
    rag = config.get("rag")
    if rag:
        retriever = CachedRetriever(
            SerperRetriever(rag["source_weights"]), SnapshotCache(rag["cache_dir"])
        )
    return Compass(client, config, retriever)
