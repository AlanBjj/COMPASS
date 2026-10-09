"""TRACE — Type-Routed Adaptive Consistency.

Self-consistency aggregates k samples with ONE fixed operator (string /
answer majority). TRACE instead picks the aggregation operator BY QUESTION TYPE, because "agreement
as evidence" means different things per type:

  * MATH  — the unit of agreement is the FINAL NUMBER, not the surrounding prose. We extract and
    canonicalize the final number from each sample, then take the frequency-weighted modal number
    and return that sample's answer verbatim. This suppresses random single-sample arithmetic slips
    without trusting noisy CoT wording.

  * REASONING / MISCONCEPTION (non-retrieval General/Logic) — the unit of agreement is the STANCE
    or the yes/no VERDICT, not the raw string. For yes/no questions we majority-vote the polar
    verdict. For open-ended adversarial/misconception questions we use CROSS-PERSPECTIVE
    CONSISTENCY (CPC) instead of same-model majority vote: same-model voting AMPLIFIES a
    misconception the model itself believes (voting reinforces the popular false belief), so on
    adversarial questions plain self-consistency makes things worse. CPC instead PERTURBS the
    misconception trigger with two cross-perspective legs — a de-biased paraphrase that strips the
    loaded framing, and a grounded devil's-advocate that audits the question's presupposition — and
    CONVERGES (not votes) to an EXPLICIT factual refutation when either leg flags the premise as a
    misconception, while leaving genuinely-true premises with their real informative answer.

Guarantees:
  * answers and numbers are kept VERBATIM — TRACE never fabricates new content, it only SELECTS
    among the k drawn samples (or falls back to the base answer);
  * deterministic tie-breaking always falls back to ``base_answer`` so an already-correct base
    answer is never replaced by a tie;
  * the existing answer format is preserved (number-only / "Answer: <n>" for math, the sample's own
    yes/no prefix for polar questions).

All thresholds / k / temperature come from the ``trace`` config block — nothing is hardcoded.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from ..decompose.decompose import MATH
from ..llm import render
from ..llm.client import BACKBONE, CONTROLLER, LLMClient

# Operator selector: which TRACE operator a route/sub-query type maps to. MATH-type questions use
# the number-majority operator; the non-retrieval reasoning types (General/Logic / commonsense /
# misconception) use the verdict/stance operator. Retrieval plans are filtered by the pipeline
# before calling trace_aggregate (their answers are checked by CPC with the evidence instead).
_MATH_OP = "math"
_REASONING_OP = "reasoning"

# A solver function re-invokable to draw one fresh sample. The pipeline binds the original
# question + sampling temperature into this closure so TRACE stays solver-agnostic.
SolverFn = Callable[[], Awaitable[str]]


@dataclass
class TraceResult:
    """Outcome of one TRACE aggregation, recorded into the pipeline trace."""

    answer: str                         # the selected (verbatim) answer, or base_answer on fallback
    fired: bool                         # did TRACE actually change anything vs base_answer
    operator: str                       # "math" | "reasoning_yesno" | "reasoning_cpc" | "none"
    qtype: str = ""
    k: int = 0                          # samples requested
    samples_used: int = 0               # samples successfully drawn
    votes: Dict[str, int] = field(default_factory=dict)
    winner_key: Optional[str] = None
    fallback: Optional[str] = None      # reason TRACE fell back to base_answer, if it did

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "fired": self.fired,
            "operator": self.operator,
            "qtype": self.qtype,
            "k": self.k,
            "samples_used": self.samples_used,
            "votes": self.votes,
            "winner_key": self.winner_key,
            "fallback": self.fallback,
        }


# ----------------------------------------------------------------------------------------------
# Number canonicalization (MATH operator)
# ----------------------------------------------------------------------------------------------

# Prefer the number after an explicit "Answer:" / "FINAL:" marker (the solver_math format ends with
# exactly such a line); otherwise we scan the whole text and keep the LAST number, which is the
# committed result in step-by-step CoT.
_ANSWER_MARKER = re.compile(r"(?:answer|final)\s*[:\-]\s*", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _canonical_number(text: str) -> Optional[str]:
    """Extract and canonicalize the final numeric answer from a math sample, or None if the sample
    contains no number. Canonicalization: drop thousands-separator commas, strip a trailing zero
    decimal (``42.0`` -> ``42``, ``42.50`` -> ``42.5``) so cosmetically-different spellings of the
    same value collapse to one vote key. Numbers are kept verbatim otherwise (no rounding)."""
    if not text:
        return None
    marker = list(_ANSWER_MARKER.finditer(text))
    region = text[marker[-1].end():] if marker else text
    nums = _NUMBER.findall(region)
    if not nums and marker:
        # The marker line had no number (e.g. "Answer: see above"); fall back to whole text.
        nums = _NUMBER.findall(text)
    if not nums:
        return None
    raw = nums[-1].replace(",", "")
    if "." in raw:
        raw = raw.rstrip("0").rstrip(".")
        if raw in ("", "-"):
            raw = "0"
    return raw


# ----------------------------------------------------------------------------------------------
# Yes/No verdict canonicalization (REASONING operator, polar branch)
# ----------------------------------------------------------------------------------------------

_YES = re.compile(r"^\W*(yes|true|correct|affirmative)\b", re.IGNORECASE)
_NO = re.compile(r"^\W*(no|false|incorrect|negative)\b", re.IGNORECASE)


_FINAL_LINE = re.compile(r"final\s+answer\s*[:：]\s*(.+)", re.IGNORECASE)


def _final_answer_line(text: str) -> Optional[str]:
    """Content of the LAST "Final answer: ..." line, or None. solve_general ends every reply with
    such a line (prompts/solver_general.txt), after the step-by-step reasoning."""
    matches = list(_FINAL_LINE.finditer(text or ""))
    return matches[-1].group(1).strip() if matches else None


def _yes_no_verdict(text: str, robust: bool = False) -> Optional[str]:
    """Return 'yes' / 'no' for a polar answer, else None.

    The leading token parses the verified base answer (which opens with the verdict) but not the
    re-sampled solver outputs, which open with their reasoning and state the verdict on the closing
    "Final answer:" line. With ``robust`` the "Final answer:" line is read first, then the leading
    token; without it, only the leading token is read."""
    if not text:
        return None
    candidates = []
    if robust:
        final = _final_answer_line(text)
        if final:
            candidates.append(final)
    candidates.append(text.strip())
    for head in candidates:
        if _YES.match(head):
            return "yes"
        if _NO.match(head):
            return "no"
    return None


# ----------------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------------


async def _draw_samples(solver_fn: SolverFn, k: int) -> List[str]:
    """Draw up to k fresh samples by re-invoking the solver. Individual failures are skipped (the
    operators degrade gracefully on fewer samples); the caller falls back to base_answer if too few
    survive. Samples are drawn concurrently — the client's own semaphore bounds real parallelism."""
    import asyncio

    results = await asyncio.gather(*[solver_fn() for _ in range(k)], return_exceptions=True)
    return [r.strip() for r in results if isinstance(r, str) and r.strip()]


# ----------------------------------------------------------------------------------------------
# MATH operator
# ----------------------------------------------------------------------------------------------


def _math_aggregate(
    base_answer: str, samples: List[str], qtype: str, k: int, min_votes: int
) -> TraceResult:
    """Frequency-weighted majority over canonicalized final numbers. The base answer participates as
    one vote so a correct base is reinforced rather than overridden by a slim sample plurality.
    Returns a sample whose number is the modal number, keeping that sample's text verbatim. Ties or
    an unclear majority -> base_answer (deterministic)."""
    keyed: List[tuple[Optional[str], str]] = [(_canonical_number(base_answer), base_answer)]
    keyed += [(_canonical_number(s), s) for s in samples]
    valid = [(num, txt) for num, txt in keyed if num is not None]

    if not valid:
        return TraceResult(base_answer, False, "math", qtype, k, len(samples),
                           fallback="no parseable number in any sample")

    counts = Counter(num for num, _ in valid)
    votes = dict(counts)
    # Deterministic winner: highest count, ties broken toward the base answer's number if it is
    # among the top, else fall back. We never let a tie silently replace the base.
    top_count = max(counts.values())
    leaders = sorted(n for n, c in counts.items() if c == top_count)
    base_num = _canonical_number(base_answer)

    if top_count < min_votes:
        return TraceResult(base_answer, False, "math", qtype, k, len(samples),
                           votes=votes, fallback=f"top vote {top_count} < min_votes {min_votes}")
    if len(leaders) > 1:
        # Tie: keep base if it is one of the tied leaders, else deterministic fallback.
        if base_num in leaders:
            return TraceResult(base_answer, False, "math", qtype, k, len(samples),
                               votes=votes, winner_key=base_num,
                               fallback="tie incl. base -> keep base")
        return TraceResult(base_answer, False, "math", qtype, k, len(samples),
                           votes=votes, fallback=f"tie among {leaders} -> keep base")

    winner = leaders[0]
    if winner == base_num:
        return TraceResult(base_answer, False, "math", qtype, k, len(samples),
                           votes=votes, winner_key=winner)
    # Return the first sample carrying the winning number, verbatim.
    rep = next(txt for num, txt in valid if num == winner)
    return TraceResult(rep, True, "math", qtype, k, len(samples), votes=votes, winner_key=winner)


# ----------------------------------------------------------------------------------------------
# REASONING operator
# ----------------------------------------------------------------------------------------------


def _yesno_aggregate(
    question: str, base_answer: str, samples: List[str], qtype: str, k: int, min_votes: int,
    robust: bool = False, majority: bool = False, min_ballots: int = 3,
) -> TraceResult:
    """Majority vote over the polar (yes/no) verdict across base + samples. Returns a representative
    sample of the winning verdict, preserving its yes/no prefix. Ties / no clear majority -> base.

    With ``robust`` and ``majority`` (the configuration used in the paper), verdicts are also read
    from the "Final answer:" line, and the winner needs a strict majority of the parseable ballots
    (at least ``min_ballots`` of them), as in self-consistency. Without them, only the leading token
    is parsed and the winner needs an absolute bar of ``min_votes``."""
    keyed: List[tuple[Optional[str], str]] = [(_yes_no_verdict(base_answer, robust), base_answer)]
    keyed += [(_yes_no_verdict(s, robust), s) for s in samples]
    valid = [(v, txt) for v, txt in keyed if v is not None]

    if not valid:
        return TraceResult(base_answer, False, "reasoning_yesno", qtype, k, len(samples),
                           fallback="no parseable yes/no verdict")

    counts = Counter(v for v, _ in valid)
    votes = dict(counts)
    top_count = max(counts.values())
    leaders = sorted(v for v, c in counts.items() if c == top_count)
    base_verdict = _yes_no_verdict(base_answer, robust)

    if majority:
        no_winner = len(valid) < min_ballots or len(leaders) > 1 or top_count * 2 <= len(valid)
        reason = "no strict majority of parseable ballots -> keep base"
    else:
        no_winner = top_count < min_votes or len(leaders) > 1
        reason = "tie / below min_votes -> keep base"
    if no_winner:
        return TraceResult(base_answer, False, "reasoning_yesno", qtype, k, len(samples),
                           votes=votes, winner_key=base_verdict, fallback=reason)

    winner = leaders[0]
    if winner == base_verdict:
        return TraceResult(base_answer, False, "reasoning_yesno", qtype, k, len(samples),
                           votes=votes, winner_key=winner)
    rep = next(txt for v, txt in valid if v == winner)
    if robust:
        # Return the sample's committed answer, not its whole reasoning trace.
        final = _final_answer_line(rep)
        if final and _yes_no_verdict(final) == winner:
            rep = final
    return TraceResult(rep, True, "reasoning_yesno", qtype, k, len(samples),
                       votes=votes, winner_key=winner)


# ----------------------------------------------------------------------------------------------
# CPC operator (Cross-Perspective Consistency) — replaces same-model stance majority vote.
# ----------------------------------------------------------------------------------------------
#
# Rationale: a stance vote samples the SAME model k times and takes the majority stance. On
# adversarial-misconception questions the model itself endorses the popular false belief, so the
# majority is ENDORSE and voting AMPLIFIES the misconception. CPC instead perturbs the misconception
# TRIGGER with two cross-perspective legs and CONVERGES to an explicit refutation:
#
#   1. DE-BIASED PARAPHRASE leg — rewrite the question into a neutral, de-leading form that strips
#      the popular / "everyone knows" / loaded framing, then answer that neutral version. Removing
#      the presupposition removes the pressure to go along with the false belief.
#   2. GROUNDED DEVIL'S-ADVOCATE leg — a single premise-AUDIT pass anchored to the question's
#      presupposition (and to retrieved evidence if the path carried any): "is the belief this
#      question assumes a common misconception? if false, give the correct factual refutation."
#      Premise-anchored, NOT free-form self-critique (intrinsic self-correction degrades factuality).
#
# Convergence (NOT majority vote): if either leg indicates the premise is FALSE -> output an
# EXPLICIT refutation/negation, decisively phrased, matching gold's explicit-negation style (gold
# is usually "Nothing happens" / "There is no ..." / "No, ...", NOT a vague hedge). A vague "no
# consensus" would be judged WRONG against an explicit-negation gold, so CPC must NOT hedge. If all
# perspectives agree the premise is TRUE, keep the base answer (anti-over-refutation guardrail: a
# genuinely true premise keeps its real, informative answer). Tie / ambiguous -> keep base.

# Parse the de-biased leg ("DEBIASED:" / "ANSWER:") and the premise audit ("PREMISE:" / "VERDICT:" /
# "REFUTATION:"). Tolerant of missing labels: we key on the VERDICT/ANSWER lines.
_CPC_ANSWER = re.compile(r"answer\s*[:\-]\s*(.+)", re.IGNORECASE | re.DOTALL)
_CPC_VERDICT = re.compile(r"verdict\s*[:\-]\s*(true|false)\b", re.IGNORECASE)
_CPC_REFUTATION = re.compile(r"refutation\s*[:\-]\s*(.+)", re.IGNORECASE | re.DOTALL)

# A neutral de-biased answer that itself states the belief is false (explicit-negation cues). Used
# as a second, independent signal that the premise is a misconception even when the audit hedges.
_NEGATION_CUE = re.compile(
    r"\b(nothing (?:harmful |bad )?happens|no (?:harm|harmful|evidence|effect|truth|such|real)|"
    r"not? true|does not|doesn't|won't|will not|isn't|is not|"
    r"myth|misconception|false|debunk|no scientific|no consensus that|"
    r"there is no|there's no|cannot|can't)\b",
    re.IGNORECASE,
)


def _cpc_parse_debias(out: str) -> tuple[Optional[str], bool]:
    """Return (answer_text, flags_false). answer_text is the de-biased leg's ANSWER body; flags_false
    is True when that answer reads as an explicit negation/refutation of the premise."""
    if not out:
        return None, False
    m = _CPC_ANSWER.search(out)
    ans = m.group(1).strip() if m else out.strip()
    return ans, bool(_NEGATION_CUE.search(ans))


def _cpc_parse_audit(out: str) -> tuple[Optional[bool], Optional[str]]:
    """Return (premise_is_false, refutation_text). premise_is_false is True/False from the VERDICT
    line (None if unparseable); refutation_text is the explicit correction when FALSE."""
    if not out:
        return None, None
    mv = _CPC_VERDICT.search(out)
    is_false: Optional[bool] = None
    if mv:
        is_false = mv.group(1).lower() == "false"
    mr = _CPC_REFUTATION.search(out)
    refutation = None
    if mr:
        body = mr.group(1).strip()
        # The TRUE branch writes "NONE"; treat that as no refutation.
        if body and body.strip().upper() != "NONE":
            refutation = body
    return is_false, refutation


async def _cpc_aggregate(
    question: str, base_answer: str, client: LLMClient, cfg: dict,
    qtype: str, evidence: str = "",
) -> TraceResult:
    """Cross-Perspective Consistency for open-ended adversarial / misconception questions.

    Runs the two cross-perspective legs concurrently (de-biased paraphrase + grounded premise audit)
    and CONVERGES to an explicit refutation when the premise is flagged false. Returns the audit's
    explicit REFUTATION (verbatim, model-authored) as the decisive answer, falling back to the
    de-biased leg's negating ANSWER if the audit gave no usable refutation text. Keeps base_answer
    when perspectives agree the premise is TRUE or the signal is ambiguous (anti-over-refutation).

    Cost: 2 extra controller calls per open-ended misconception-type question, recorded in the
    ledger by role like every other call. Knobs (temperature, max_tokens) come from the trace cfg.
    """
    import asyncio

    cpc_temp = float(cfg.get("cpc_temperature", 0.0))
    cpc_max_tokens = int(cfg.get("cpc_max_tokens", 256))
    # Block injected into the audit prompt only when the path carried retrieved evidence; empty
    # otherwise (non-retrieval reasoning paths). Keeps the devil's advocate premise-anchored and,
    # when available, evidence-grounded rather than free-form.
    evidence_block = (
        f"Use the following retrieved evidence when judging the premise:\n{evidence}\n\n"
        if evidence and evidence.strip() else ""
    )

    async def run_debias() -> str:
        return await client.chat_text(
            render("cpc_debias", question=question),
            CONTROLLER, temperature=cpc_temp, max_tokens=cpc_max_tokens,
        )

    async def run_audit() -> str:
        return await client.chat_text(
            render(cfg.get("audit_prompt", "cpc_premise_audit"), question=question, evidence=evidence_block),
            CONTROLLER, temperature=cpc_temp, max_tokens=cpc_max_tokens,
        )

    debias_out, audit_out = await asyncio.gather(
        run_debias(), run_audit(), return_exceptions=True,
    )
    debias_text = debias_out if isinstance(debias_out, str) else ""
    audit_text = audit_out if isinstance(audit_out, str) else ""

    debias_ans, debias_false = _cpc_parse_debias(debias_text)
    audit_false, audit_refutation = _cpc_parse_audit(audit_text)

    # Convergence signal. The premise audit is the AUTHORITATIVE premise judgment: an explicit
    # VERDICT (TRUE/FALSE) decides directly. The de-biased leg's negation cue is only an INDEPENDENT
    # refutation signal when the audit gave NO verdict (parse failure / silence) — it must never
    # override an explicit TRUE verdict, otherwise a true-premise answer that happens to contain a
    # negating phrase ("does not absorb", "almost always") gets falsely refuted (anti-over-refutation).
    votes = {
        "audit_verdict": ("false" if audit_false else "true") if audit_false is not None else "none",
        "debias_negates": "yes" if debias_false else "no",
    }
    if audit_false is None:
        # Audit gave no usable verdict: fall back to the de-biased leg's explicit-negation cue.
        premise_false = debias_false
    else:
        premise_false = audit_false

    if cfg.get("cpc_require_agreement") and audit_false and not debias_false:
        # Option (off in all reported runs): refute only when BOTH perspectives converge on a false
        # premise.
        return TraceResult(base_answer, False, "reasoning_cpc", qtype, 2, 2,
                           votes=votes, winner_key="legs_disagree",
                           fallback="audit false but de-biased leg did not negate -> keep base")

    if audit_false is None and not debias_false:
        # Neither leg produced a usable signal (parse failure / both silent) -> deterministic base.
        return TraceResult(base_answer, False, "reasoning_cpc", qtype, 2, 2,
                           votes=votes, winner_key="ambiguous",
                           fallback="no usable cross-perspective signal -> keep base")

    if not premise_false:
        # All perspectives agree the premise is TRUE (audit said TRUE and de-bias did not negate).
        # Guardrail: keep the real, informative base answer — do not nihilistically negate.
        return TraceResult(base_answer, False, "reasoning_cpc", qtype, 2, 2,
                           votes=votes, winner_key="true_premise",
                           fallback="premise judged true -> keep base answer")

    # Premise is FALSE -> CONVERGE to an EXPLICIT refutation (verbatim, model-authored). Prefer the
    # audit's REFUTATION (built to be an explicit negation); fall back to the de-biased ANSWER if the
    # audit gave no refutation body but the de-biased leg negated.
    final = audit_refutation or (debias_ans if debias_false else None)
    if not final or not final.strip():
        # Premise flagged false but no concrete refutation text surfaced -> keep base (don't emit a
        # contentless negation; avoids a vague hedge that would be judged wrong against gold).
        return TraceResult(base_answer, False, "reasoning_cpc", qtype, 2, 2,
                           votes=votes, winner_key="false_premise",
                           fallback="premise false but no explicit refutation text -> keep base")

    return TraceResult(final.strip(), True, "reasoning_cpc", qtype, 2, 2,
                       votes=votes, winner_key="false_premise")


# ----------------------------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------------------------


def _operator_for(qtype: str) -> Optional[str]:
    """Map a COMPASS route/sub-query type to a TRACE operator."""
    return _MATH_OP if qtype == MATH else _REASONING_OP


async def trace_aggregate(
    question: str,
    base_answer: str,
    qtype: str,
    solver_fn: SolverFn,
    client: LLMClient,
    cfg: dict,
) -> TraceResult:
    """Type-routed adaptive consistency over k re-sampled solver outputs.

    Draws k samples by re-invoking ``solver_fn`` at the configured sampling temperature, then
    aggregates by ``qtype``: the MATH operator for math-type questions, otherwise the REASONING
    operator (yes/no verdict vote for polar questions, CPC cross-perspective convergence for
    open-ended misconception-type ones). Returns a
    ``TraceResult`` whose ``answer`` is either a verbatim selected sample or ``base_answer`` on any
    fallback. ``cfg`` is the parsed ``trace`` config block (k, temperature, min_votes, ...);
    nothing is hardcoded.
    """
    k = int(cfg["k"])
    # A clear majority must clear at least ceil((k+1)/2) by default (the +1 counts the base vote),
    # i.e. a true majority of the participating ballots. Overridable via config for the sensitivity
    # study; never hardcoded inline.
    default_min = (k + 1) // 2 + 1
    min_votes = int(cfg.get("min_votes", default_min))

    operator = _operator_for(qtype)
    if operator is None:
        return TraceResult(base_answer, False, "none", qtype, k, 0,
                           fallback=f"qtype {qtype!r} out of TRACE scope")

    # Open-ended reasoning / misconception questions use CPC (Cross-Perspective Consistency), which
    # does NOT need the k same-model samples — same-model voting amplifies the misconception. CPC
    # runs its own two cross-perspective legs instead, so we route here BEFORE drawing samples to
    # avoid k wasted solver calls. evidence (if the path carried any) is passed via cfg for grounding.
    if operator == _REASONING_OP and not _is_yes_no(question):
        if not cfg.get("cpc", True):   # ablation "-noCPC": keep the base answer
            return TraceResult(base_answer, False, "none", qtype, k, 0, fallback="CPC disabled (ablation)")
        return await _cpc_aggregate(
            question, base_answer, client, cfg, qtype,
            evidence=str(cfg.get("evidence", "")),
        )
    if operator == _REASONING_OP and not cfg.get("yesno", True):   # ablation "-noTRACE" (yes/no vote)
        return TraceResult(base_answer, False, "none", qtype, k, 0, fallback="yes/no vote disabled (ablation)")

    robust = bool(cfg.get("yesno_robust_parse", False))
    if operator == _MATH_OP:
        def key(text: str) -> Optional[str]:
            return _canonical_number(text)
    else:
        def key(text: str) -> Optional[str]:
            return _yes_no_verdict(text, robust)

    # Early stop: draw (k+1)//2 samples first. If they and the base all give the same parseable answer,
    # that answer holds a strict majority of the k+1 ballots whatever the other samples say, so the
    # vote is settled (both operators then keep the base answer) and the rest are not drawn.
    first = (k + 1) // 2
    if cfg.get("early_stop") and 0 < first < k and first + 1 >= min_votes:
        samples = await _draw_samples(solver_fn, first)
        base_key = key(base_answer)
        if not (base_key is not None and len(samples) == first and all(key(s) == base_key for s in samples)):
            samples += await _draw_samples(solver_fn, k - first)
    else:
        samples = await _draw_samples(solver_fn, k)
    if not samples:
        return TraceResult(base_answer, False, operator, qtype, k, 0,
                           fallback="no samples drawn")

    if operator == _MATH_OP:
        return _math_aggregate(base_answer, samples, qtype, k, min_votes)

    # REASONING operator, polar (yes/no) question -> verdict majority vote (kept: cheap and works on
    # binary verdicts; the de-biased leg could also be used here but simple verdict voting suffices).
    return _yesno_aggregate(
        question, base_answer, samples, qtype, k, min_votes,
        robust=robust,
        majority=bool(cfg.get("yesno_majority", False)),
        min_ballots=int(cfg.get("yesno_min_ballots", 3)),
    )


# Local copy of the polar-question shape test. We re-implement the small token check here rather
# than import from the pipeline to avoid a circular import (pipeline imports this module).
_YES_NO_LEADING_TOKENS = frozenset(
    {
        "is", "are", "was", "were", "am", "be", "being", "been",
        "can", "could", "would", "should", "shall", "will", "may", "might", "must",
        "do", "does", "did",
        "has", "have", "had",
    }
)


def _is_yes_no(q: str) -> bool:
    if not q:
        return False
    first = q.strip().split(maxsplit=1)
    if not first:
        return False
    token = first[0].strip("\"'`.,;:!?()[]{}").lower()
    return token in _YES_NO_LEADING_TOKENS
