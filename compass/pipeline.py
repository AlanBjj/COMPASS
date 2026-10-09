"""COMPASS orchestration (Algorithm 1 of the paper).

Wires the hybrid gate, the planner, the typed operators, and the type-routed consistency layer into
one dataset-agnostic entry point. The controller role (gate, decomposition, scoring, fusion, CPC) and
the backbone role (answers, verification, TRACE samples) are billed separately via a shared per-role
ledger; by default they are the same served model, but they can point to different endpoints (see
build_compass).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import re

from .aggregation import trace_aggregate
from .decompose import FACT, GENERAL, MATH, SubQuery, decompose
from .decompose.decompose import LOGIC, _looks_arithmetic
from .fusion import fuse, is_low_quality, score_subanswers
from .gate import decide
from .gate.heuristic import _COMPARISON, _RECENCY, _matches_any
from .llm import render
from .llm.client import BACKBONE, DualLLMClient, LLMClient, TokenLedger
from .solvers import (
    CachedRetriever,
    Retriever,
    SerperRetriever,
    SnapshotCache,
    route,
    solve_general,
    solve_math,
    solve_rag,
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


_WORD = re.compile(r"[A-Za-z][A-Za-z.'\-]*")


def _names_entity(q: str) -> bool:
    """True if the question names a specific entity: a capitalized word anywhere after the first
    word (proper nouns, titles, acronyms; the pronoun "I" excluded).

    A question about a NAMED entity ("When was Dartmouth Law School founded?") asks for a specific,
    often time-varying fact that the backbone's parametric memory may not hold; a GENERIC question
    ("What happens if you crack your knuckles?") turns on common knowledge, where retrieved web text
    tends to carry the popular misconception instead. The gate's external_dependency score does not
    separate the two, since the backbone rates most named-entity facts as stable (score 0). This is a
    surface shape test, not a dataset constant."""
    words = _WORD.findall(q or "")
    return any(w[0].isupper() and w != "I" for w in words[1:])


# Cues that a fact question is TIME-SENSITIVE (the recency test R(q)): the gate's _RECENCY cues plus
# forward-looking ("next", "upcoming") and running-total ("so far", "to date") cues. Such questions are
# answered from the evidence as a whole: decomposing them sends parts to the parametric solvers, whose
# memory is stale by construction.
_TIME_SENSITIVE = _RECENCY + [r"\b(next|upcoming|so far|to date|newest)\b"]


def _is_time_sensitive(q: str) -> bool:
    return _matches_any(_TIME_SENSITIVE, (q or "").lower())


# Whole-question arithmetic word problems, the test A(q). `_looks_arithmetic` (a numeric cue plus any
# digit) is tuned for short sub-queries; on whole questions it misses problems written with number
# words ("a one hour yoga class ... three times as long") or asking for a time ("what is the latest
# time ...?"), and it fires on fact questions that merely contain a number ("the highest death rate
# per 100,000 in 2023"). A word problem states at least two quantities (digits or number words;
# years are not quantities) and asks for a numeric result; an entity-seeking which/who/where/when
# question is not one.
_NUMBER_WORDS = (r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|"
                 r"thirty|forty|fifty|hundred|thousand|half|twice|thrice|double|triple|dozen|quarter")
_QUANTITY = re.compile(rf"\d[\d,]*(?:\.\d+)?|\b(?:{_NUMBER_WORDS})\b", re.IGNORECASE)
_YEAR = re.compile(r"^(1[5-9]|20)\d\d$")
_NUMERIC_ASK = re.compile(
    r"how many|how much|how long|how old|how far|how fast|what time|latest time|earliest time|"
    r"what (?:fraction|percent|percentage)|\btotal\b|\baltogether\b|\bin all\b|\bleft\b|\bremain|"
    r"\bdifference\b|\bcalculate\b|\bcompute\b|\bfind the\b|\bamount\b|\bprice\b|\bcost\b|\bchange\b|"
    r"\bprofit\b|\bearn",
    re.IGNORECASE,
)
_ENTITY_ASK = re.compile(r"^\s*(which|who|whom|whose|where|when)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")  # "the U.S. had" is not a sentence break


def _is_arithmetic_problem(q: str) -> bool:
    q = q or ""
    # The asked sentence is the last one ("When the water is cold Ray swims ... How much longer ...?"
    # opens with "When" but asks "how much").
    sentences = [x for x in _SENTENCE_END.split(q.strip()) if x.strip()]
    asked = sentences[-1] if sentences else q
    if _ENTITY_ASK.match(asked) or not _NUMERIC_ASK.search(q):
        return False
    quantities = [m.group(0) for m in _QUANTITY.finditer(q)
                  if not _YEAR.match(m.group(0).replace(",", ""))]
    return len(quantities) >= 2


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
    path: str                         # "direct" | "direct_retrieval" | "recency_retrieval" | "fact_retrieval" | "decompose" | "homogeneous_<type>" | "low_quality_fallback"
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
        # Planner and consistency rules (configs `rules:` block). Every switch defaults to off, so a
        # config without the block runs the bare pipeline; the configs in configs/ switch on the full
        # method, and each ablation of the paper turns one rule off (compass.run.ABLATIONS).
        #   entity_retrieval   — simple wh-questions that name an entity are answered from evidence;
        #   math_no_split      — word problems that decompose only into Math/Logic parts are solved
        #                        whole by the math solver;
        #   yesno_facts        — yes/no questions list the implicit facts before the verdict
        #                        (prompts/solver_yesno.txt);
        #   cpc_on_retrieval   — CPC also checks the answers of retrieval plans, against the evidence;
        #   recency_retrieval  — time-sensitive non-yes/no questions take the recency plan;
        #   fallback_retrieval — the low-quality fallback answers from evidence when the gate saw an
        #                        external dependency;
        #   arith_fix          — the whole-question arithmetic test A(q) drives the retrieval, math,
        #                        and TRACE routing decisions;
        #   no_cpc_recency     — answers of the recency plan are exempt from CPC;
        #   fact_no_split      — complex questions that ask for a fact take the fact plan
        #                        (whole-question retrieval) instead of decomposition.
        # Two further options are off in all reported runs: rag_knowledge_fallback (the retrieval
        # prompt may fall back to a confidently known stable fact, prompts/solver_rag_knowledge.txt)
        # and grounded_audit (the premise audit on retrieval plans treats the evidence as authoritative).
        self.rules = config.get("rules") or {}
        self.rag_prompt = "solver_rag_knowledge" if self.rules.get("rag_knowledge_fallback") else "solver_rag"

    def _wants_retrieval(self, q: str, gate: Dict[str, object]) -> bool:
        """Direct-plan retrieval decision E(q): never for yes/no questions; yes when the gate's
        external_dependency is at its maximum, or (entity_retrieval) for wh-questions that name a
        specific entity, unless they are misconception/presupposition-shaped or arithmetic."""
        if _is_yes_no_question(q):
            return False
        if _external_dependency(gate) == _EXTERNAL_DEPENDENCY_MAX:
            return True
        return bool(
            self.rules.get("entity_retrieval")
            and _names_entity(q)
            and not _is_presupposition_shaped(q)
            and not _looks_arithmetic(q)
            and not (self.rules.get("arith_fix") and _is_arithmetic_problem(q))
        )

    def _math_whole(self, q: str, subs: List[SubQuery], gate: Dict[str, object]) -> bool:
        """math_no_split: an arithmetic problem that needs no external knowledge and decomposes only into
        Math/Logic parts is one connected calculation; answer it whole with the math solver
        (and TRACE's number vote) rather than slicing and fusing it."""
        if not self.rules.get("math_no_split"):
            return False
        types = {s.type for s in subs}
        return (
            MATH in types
            and types <= {MATH, LOGIC}
            and _external_dependency(gate) == 0
            and (_looks_arithmetic(q) or (self.rules.get("arith_fix") and _is_arithmetic_problem(q)))
        )

    def _time_sensitive_fact(self, q: str) -> bool:
        """Recency plan test: a time-sensitive fact question (not yes/no, not a word problem)."""
        return bool(
            self.rules.get("recency_retrieval")
            and self.retriever is not None
            and not _is_yes_no_question(q)
            and _is_time_sensitive(q)
            and not _is_arithmetic_problem(q)
            and not _looks_arithmetic(q)
        )

    async def _general(self, q: str, temperature: Optional[float] = None) -> str:
        """Whole-question parametric answer (direct plan, fallback, and TRACE reasoning samples).
        With yesno_facts, yes/no questions use the implicit-facts prompt."""
        prompt = "solver_general"
        if self.rules.get("yesno_facts") and _is_yes_no_question(q):
            prompt = "solver_yesno"
        return await solve_general(
            q, q, self.client,
            temperature=self.back["temperature"] if temperature is None else temperature,
            max_tokens=self.back["max_tokens"], prompt_name=prompt,
        )

    async def _direct(self, q: str) -> str:
        return await self._general(q)

    async def _verify(self, q: str, draft: str) -> str:
        """One-pass self-verification, applied to every plan except the direct fallback, including
        the decompose-and-fuse plan, whose fusion stage has no misconception or false-premise check of
        its own. A single backbone critique-and-revise call that defaults to keeping the draft and only
        edits it on a clear, general failure (misconception trap, yes/no verdict that contradicts the
        reasoning, off-question answer, self-contradiction): exactly one extra backbone call.

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

    async def _direct_retrieval(self, q: str, evidence_sink: Optional[List[str]] = None) -> str:
        """ED-driven whole-question RAG: send the entire question (not a sub-query) to the Fact
        expert. route() applies the configured retriever and the numeric-leak guard, and falls
        back to direct generation if no retriever is configured. With arith_fix, the whole-question
        arithmetic test decides instead, so a fact question that merely contains a number ("per
        100,000 in 2023") is not sent to the math solver."""
        if self.rules.get("arith_fix") and self.retriever is not None and not _is_arithmetic_problem(q):
            return await solve_rag(
                q, q, self.client, self.retriever,
                temperature=self.back["temperature"], max_tokens=self.back["max_tokens"],
                prompt_name=self.rag_prompt, evidence_sink=evidence_sink,
            )
        return await route(
            SubQuery(question=q, type=FACT), self.client, main_q=q, retriever=self.retriever,
            temperature=self.back["temperature"], max_tokens=self.back["max_tokens"],
            rag_prompt=self.rag_prompt, evidence_sink=evidence_sink,
        )

    async def _retrieval_answer(self, q: str, path: str, trace: Dict[str, object]) -> "CompassResult":
        """Whole-question retrieval answer + verify; with cpc_on_retrieval, CPC checks it against the
        evidence, except on the recency plan when no_cpc_recency is set (its questions ask about a
        current fact that the model's training data predates)."""
        evidence: List[str] = []
        base = await self._verify(q, await self._direct_retrieval(q, evidence))
        if self.rules.get("cpc_on_retrieval") and not (path == "recency_retrieval" and self.rules.get("no_cpc_recency")):
            base = await self._grounded_cpc(q, base, trace, "\n".join(evidence))
        return CompassResult(base, path, trace)

    async def _grounded_cpc(self, q: str, base: str, trace: Dict[str, object], evidence: str) -> str:
        """CPC on a retrieval-plan answer, with the retrieved evidence in the premise audit: retrieved
        web text often states the popular misconception a question is built on. yes/no and arithmetic
        questions are left to their own operators (they never reach whole-question retrieval)."""
        if not self.trace_cfg.get("enabled") or _is_yes_no_question(q) or _is_arithmetic_problem(q):
            return base
        cfg = dict(self.trace_cfg)
        cfg["evidence"] = evidence[: int(self.rules.get("cpc_evidence_chars", 6000))]
        if self.rules.get("grounded_audit"):   # option, off in all reported runs
            cfg["audit_prompt"] = "cpc_premise_audit_grounded"

        async def solver_fn() -> str:  # CPC does not sample; kept for the common signature
            return await self._general(q, temperature=float(self.trace_cfg.get("temperature", 0.7)))

        result = await trace_aggregate(q, base, GENERAL, solver_fn, self.client, cfg)
        trace["trace_consistency"] = result.as_dict()
        return result.answer

    def _trace_qtype(self, q: str, path: str, homo: Optional[str] = None) -> Optional[str]:
        """Decide which answer type drives the consistency layer for a finished answer, or None if
        the plan is out of scope (config-gated per type, see configs `trace:`):
          * MATH — math-type answers: a whole-Math plan, or any direct/decompose answer to an
            arithmetic question -> numeric vote (TRACE).
          * REASONING — non-retrieval General/Logic answers (direct, decompose, low-quality
            fallback) that are not arithmetic -> yes/no verdict vote (TRACE), or CPC for
            open-ended answers.
        Retrieval plans are checked in _retrieval_answer instead.
        """
        if not self.trace_cfg.get("enabled"):
            return None
        # Retrieval plans are checked in _retrieval_answer.
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
        if _looks_arithmetic(q) or (self.rules.get("arith_fix") and _is_arithmetic_problem(q)):
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
                return await self._general(q, temperature=sample_temp)

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
        if self._time_sensitive_fact(q):
            return await self._retrieval_answer(q, "recency_retrieval", trace)
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
            if self._wants_retrieval(q, gate):
                return await self._retrieval_answer(q, "direct_retrieval", trace)
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

        # Fact plan (fact_no_split): a question that depends on external knowledge (not yes/no, not a
        # word problem) asks for a fact; answering it whole from retrieved evidence keeps the
        # question's own constraints together, where decomposition answers the parts separately.
        if (
            self.rules.get("fact_no_split") and self.retriever is not None
            and not _is_yes_no_question(q) and not _is_arithmetic_problem(q) and not _looks_arithmetic(q)
            and (_external_dependency(gate) or 0) >= 1
        ):
            return await self._retrieval_answer(q, "fact_retrieval", trace)

        # Step 2 — Sub-queries Decomposition
        subs: List[SubQuery] = await decompose(
            q, self.client,
            min_subqueries=self.dec_cfg["min_subqueries"],
            max_subqueries=self.dec_cfg["max_subqueries"],
            temperature=self.ctrl["temperature"], max_tokens=self.ctrl["max_tokens"],
        )
        trace["subqueries"] = [s.as_dict() for s in subs]

        # Step 2.5 — whole-question routing: decompose only HETEROGENEOUS complex queries. If the decomposition is
        # homogeneous (all sub-queries share one type), route the WHOLE original question to that
        # one type's expert in a single call — preserving the connected reasoning chain instead of
        # slicing it. Heterogeneous (mixed-type) queries fall through to the per-type
        # solve -> score -> fuse path below, unchanged.
        homo = _homogeneous_type(subs)
        if homo is None and self._math_whole(q, subs, gate):
            homo = MATH
        if homo == MATH and self.rules.get("arith_fix") and not _is_arithmetic_problem(q):
            # arith_fix: the decomposer typed every part Math, but the question is not a word problem
            # (e.g. "Which state had the highest death rate per 100,000 in 2023?"): answer it as a
            # fact question rather than forcing a number out of the math solver.
            homo = FACT if self.retriever is not None else GENERAL
        if homo == FACT and self.rules.get("cpc_on_retrieval"):
            return await self._retrieval_answer(q, f"homogeneous_{FACT}", trace)
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
            if (
                self.rules.get("fallback_retrieval") and self.retriever is not None
                and not _is_yes_no_question(q) and not _is_arithmetic_problem(q)
                and (_external_dependency(gate) or 0) >= 1
            ):
                # fallback_retrieval: the sub-answers were unusable and the question depends on external
                # knowledge, so fall back to whole-question retrieval, not to parametric memory.
                return await self._retrieval_answer(q, "low_quality_fallback", trace)
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


def _make_client(m: dict, client_cfg: dict, ledger: TokenLedger, system_note: Optional[str] = None) -> LLMClient:
    """Build one LLMClient from a models.<role> config block. If the block sets
    `enable_thinking: false` (Qwen3 thinking models), suppress the <think> trace server-side and
    strip any residual block client-side so the structured controller parsers stay intact."""
    extra_body = None
    strip_thinking = False
    if m.get("enable_thinking") is False:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        strip_thinking = True
    return LLMClient(
        base_url=m["base_url"], model=m["model"],
        default_temperature=m.get("temperature", 0.0),
        default_max_tokens=m.get("max_tokens", 1024),
        concurrency=client_cfg.get("concurrency", 8),
        retries=client_cfg.get("retries", 3),
        ledger=ledger, extra_body=extra_body, strip_thinking=strip_thinking,
        system_note=system_note,
    )


def date_note(config: dict) -> Optional[str]:
    """Evaluation setting for time-sensitive benchmarks: `run.current_date` (e.g. "April 2026" for the
    FreshQA version in use) becomes "Today's date is April 2026." on every model call. It is part of
    the benchmark setup, shared by all methods (baselines included), not a COMPASS component."""
    d = (config.get("run") or {}).get("current_date")
    return f"Today's date is {d}." if d else None


def build_compass(config: dict) -> Compass:
    """Construct a Compass from a parsed config dict.

    Controller (gate/decompose/scoring) and backbone (answering/RAG/fusion/verify) are billed
    separately via a shared per-role TokenLedger. When `models.controller` and `models.backbone`
    name the SAME endpoint+model (the default), a single LLMClient serves both roles. When they
    differ (e.g. a larger controller in front of a smaller backbone), a DualLLMClient routes each role
    to its own endpoint. The retriever is built only if a `rag` section is present: BM25 over the
    frozen snapshot (`backend: local_snapshot`, as in the paper) or live web search with a cache."""
    mc = config["models"]["controller"]
    mb = config["models"]["backbone"]
    client_cfg = config.get("client", {})
    ledger = TokenLedger()
    same = (
        mc["base_url"] == mb["base_url"]
        and mc["model"] == mb["model"]
        and mc.get("enable_thinking") == mb.get("enable_thinking")
    )
    note = date_note(config)
    if same:
        client = _make_client(mc, client_cfg, ledger, note)
    else:
        client = DualLLMClient(
            _make_client(mc, client_cfg, ledger, note),
            _make_client(mb, client_cfg, ledger, note),
            ledger,
        )
    retriever: Optional[Retriever] = None
    rag = config.get("rag")
    if rag:
        if rag.get("backend") == "local_snapshot":
            # BM25 over the frozen web-search snapshot (deterministic, so no per-query cache).
            from .retrieval_local import LocalSourceRetriever, QueryRewriteRetriever

            retriever = LocalSourceRetriever(
                rag["source_weights"], rag["corpus"], rag["index_dir"],
                per_source_k=int(rag.get("per_source_k", 5)),
            )
            qr = rag.get("query_rewrite") or {}
            if qr.get("enabled"):
                retriever = QueryRewriteRetriever(
                    retriever, client,
                    n_queries=int(qr.get("n_queries", 3)),
                    per_source_k=int(qr.get("per_source_k", rag.get("per_source_k", 5))),
                    temperature=mc.get("temperature", 0.0),
                )
        else:
            retriever = CachedRetriever(
                SerperRetriever(rag["source_weights"]), SnapshotCache(rag["cache_dir"])
            )
    return Compass(client, config, retriever)
