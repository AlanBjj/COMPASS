"""ReAct (Yao et al. 2023): interleaved Thought / Action / Observation reasoning.

Faithful to the official repo (ysymyth/ReAct @ 6bdb3a1, hotpotqa.ipynb + prompts_naive.json
key `webthink_simple6`): the 3-action instruction block and the 6-shot HotpotQA exemplars are
copied VERBATIM into prompts/baselines/react.txt. The loop runs up to 7 steps (range(1, 8)),
each step is one LLM call that produces `Thought i:` + `Action i:`, and observations come ONLY
from the environment — we truncate the model output at the first "\nObservation" so it cannot
hallucinate its own observations (standard ReAct harness behavior). The numbered
`Thought i / Action i / Observation i` scratchpad format matches the original.

Annotated deviations from the official repo (infra differences, not method changes):
- Search[entity]: the original hits a Wikipedia API; we route to the shared cached web
  retriever (Serper-backed web_search + format_snippets) and use the snippets as the Observation.
- Lookup[keyword]: the original returns the next sentence containing the keyword in the
  CURRENT Wikipedia passage (in-page lookup). With web snippets there is no persistent
  "current passage", so we APPROXIMATE Lookup[keyword] as a web_search(keyword) — a deviation.
- Forced final: the original force-emits Finish[] (an EMPTY answer) when it overflows the step
  budget. We instead synthesize a best-effort final answer from the scratchpad via
  react_final.txt — a deliberately stronger baseline; annotated here as a deviation.
"""

from __future__ import annotations

import re

from ..llm import render
from ..llm.client import BACKBONE
from .base import Baseline, BaselineResult
from .retrieval import format_snippets, web_search

# Accept optional step numbers: "Action 3: Search[...]" or "Action: Finish[...]".
_ACTION = re.compile(r"Action\s*\d*:\s*(Search|Lookup|Finish)\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)


class ReActBaseline(Baseline):
    name = "react"

    async def answer(self, question: str) -> BaselineResult:
        max_steps = int(self.params.get("max_steps", 7))
        top_k = int(self.params.get("top_k", 5))
        scratchpad, steps = "", []

        for i in range(1, max_steps + 1):
            # Cue the model for this numbered step, mirroring the original `prompt + f"Thought {i}:"`.
            scratchpad += f"Thought {i}:"
            out = await self.client.chat_text(
                render("baselines/react", question=question, scratchpad=scratchpad),
                BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
            )
            # Observations come only from the env: cut anything the model wrote past its Action.
            out = re.split(r"\nObservation", out, maxsplit=1)[0].strip()
            scratchpad += " " + out + "\n"

            m = _ACTION.search(out)
            if not m:
                # Mirror the env: unparseable action -> "Invalid action" observation, keep going.
                steps.append({"step": i, "out": out, "action": None})
                scratchpad += f"Observation {i}: Invalid action. Use Search[entity], Lookup[keyword], or Finish[answer].\n"
                continue

            kind, arg = m.group(1).lower(), m.group(2).strip()
            steps.append({"step": i, "out": out, "action": kind, "arg": arg})

            if kind == "finish":
                return BaselineResult(answer=arg, trace={"steps": steps, "scratchpad": scratchpad})

            # Search and Lookup both route to web search (Lookup is an approximation; see module docstring).
            if self.retriever is not None:
                snippets = await web_search(self.retriever, arg, top_k=top_k)
                obs = format_snippets(snippets)
            else:
                obs = "No retrieval backend available."
            scratchpad += f"Observation {i}: {obs}\n"

        # Overflowed the step budget. The original force-returns Finish[] (empty); we instead
        # synthesize a best-effort answer from the scratchpad (annotated deviation, stronger baseline).
        final = await self.client.chat_text(
            render("baselines/react_final", question=question, scratchpad=scratchpad),
            BACKBONE, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return BaselineResult(
            answer=final.strip(),
            trace={"steps": steps, "scratchpad": scratchpad, "forced_final": True},
        )
