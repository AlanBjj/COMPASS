"""Self-Consistency (Wang et al. 2022): sample k Chain-of-Thought paths, then take the majority
vote over their extracted final answers. This is the *vanilla* self-consistency baseline — ONE
fixed aggregation operator (answer-string majority) applied uniformly to every question type.

It is the natural reference point for COMPASS's TRACE component, which differs by ONE variable:
TRACE keeps the same k samples but picks the aggregation operator BY QUESTION TYPE (number-modal
vote for math, polar-verdict vote for yes/no, cross-perspective convergence for open-ended
misconception) instead of one global string vote. Holding the prompt, k, and sampling temperature
identical to TRACE isolates "global vote vs type-routed operator" as the only difference.

Sampling temperature deviates from the protocol's backbone temperature=0: self-consistency is
defined by diverse sampled reasoning paths and is degenerate at greedy decoding (all k samples
identical). We therefore draw the k paths at the configured `sample_temperature` (default 0.7,
the value used by Wang et al. 2022 and by COMPASS's own TRACE block), while the rest of the
protocol is unchanged. The final voted answer is the one judged; all k backbone calls are billed
to the per-query token total.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import List

from ..llm import render
from ..llm.client import BACKBONE
from .base import Baseline, BaselineResult
from .util import extract_answer


def _vote_key(text: str) -> str:
    """Normalize a sample's extracted final answer into a vote key: take the text after the
    'Answer:' marker, lowercase, collapse whitespace, and strip surrounding punctuation so that
    cosmetically-different spellings of the same answer collapse to one vote."""
    ans = extract_answer(text)
    ans = ans.strip().lower()
    ans = re.sub(r"\s+", " ", ans)
    return ans.strip(" \t\r\n.,:;!?\"'`()[]{}")


class SelfConsistencyBaseline(Baseline):
    name = "self_consistency"

    async def answer(self, question: str) -> BaselineResult:
        # k and the sampling temperature live in the config methods block (project convention:
        # no hyperparameter hardcoded). Defaults match Wang et al. 2022 / the COMPASS TRACE block.
        k = int(self.params.get("num_samples", 5))
        sample_temp = float(self.params.get("sample_temperature", 0.7))

        prompt = render("baselines/direct", question=question)
        drawn = await asyncio.gather(
            *[
                self.client.chat_text(
                    prompt, BACKBONE, temperature=sample_temp, max_tokens=self.max_tokens,
                )
                for _ in range(k)
            ],
            return_exceptions=True,
        )
        samples: List[str] = [s.strip() for s in drawn if isinstance(s, str) and s.strip()]
        if not samples:
            return BaselineResult(answer="", trace={"error": "no samples drawn", "k": k})

        # Global majority vote over the normalized final answers. Ties (and the all-empty-key case)
        # break deterministically toward the FIRST sample carrying a top-count key, in sample order,
        # so the result is reproducible given the drawn samples.
        keys = [_vote_key(s) for s in samples]
        counts = Counter(keys)
        top_count = max(counts.values())
        winner_key = next(key for key in keys if counts[key] == top_count)
        # Return the first sample whose key won, VERBATIM (full CoT + "Answer:" line), so the judge
        # extracts the final answer exactly as it does for every other method.
        answer = next(s for s, key in zip(samples, keys) if key == winner_key)

        return BaselineResult(
            answer=answer,
            trace={
                "k": k,
                "sample_temperature": sample_temp,
                "samples_used": len(samples),
                "votes": dict(counts),
                "winner_key": winner_key,
                "samples": samples,
            },
        )
