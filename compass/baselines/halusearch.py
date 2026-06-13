"""HaluSearch (ACL 2025 Findings; "Think More, Hallucinate Less: Mitigating Hallucinations via
Dual Process of Fast and Slow Thinking", arXiv:2501.01306).

PROVENANCE
==========
Official code is NOT released, so this is a faithful reimplementation from the paper.

FAITHFUL to the paper:
- Genuine Monte Carlo Tree Search over reasoning steps. A node = one generated reasoning SENTENCE
  (step); the root is the question only. Each iteration runs the four MCTS phases:
    SELECT    descend from the root, at each level picking the child with the highest
              UCT(s) = V(s) + w * sqrt(ln N(parent) / N(s)),  w = 0.4.
    EXPAND    from the selected node, generate K children (candidate next sentences) at the
              expansion temperature (0.9), so candidates are diverse.
    SIMULATE  for a freshly expanded node, sample `m` short rollout continuations and score EACH
              with the reward prompt; the node value V is the MEAN reward over the m rollouts.
    BACKPROP  push the reward up the path with the running-mean update
              V_new = (V_old * N_old + r) / N_new  and increment visit counts.
- Reward is a 1-5 hallucination-RISK rating of the FUTURE CONTINUATION (1 = no risk ... 5 = very
  high risk), ported from the paper's rubric (halusearch_verify.txt). We convert risk to a QUALITY
  value  V = 6 - risk  so that higher is better for UCT / selection / the gamma switch (this also
  resolves the paper's scale-direction ambiguity: everything downstream is on a 1-5 quality scale).
- Hierarchical (dual-process) system switch:
    * step-level: if the parent's value >= gamma (the path is already reliable) expand FAST with a
      single child; otherwise expand SLOW with K children.
    * instance-level: a cheap confidence gate (one direct answer + a self-rated reliability score)
      decides System 1 (return the direct answer, no search) vs System 2 (run full MCTS).
- Final answer: take the greedy highest-value root->terminal path, then consolidate it with
  halusearch_final.txt.
- Termination: max iterations M reached, OR a node reaches quality >= r_th, OR a path hits a
  "FINAL:" marker (terminal) or max_depth.

ANNOTATED DEVIATIONS (the only departures from the paper):
- Trained generative reward model  ->  zero-shot self-eval on the SAME backbone (the verify prompt).
- Trained fast/slow switch classifier  ->  a value-vs-gamma heuristic (the paper trains a classifier
  that merely imitates this reliability rule, so the heuristic is the rule the classifier learns).
- Search budgets REDUCED for tractability: K=3, m=2, M=8 here vs the paper's K=10, m=5, M=20. Full
  budgets are several thousand 7B calls per question, which is impractical for a baseline sweep; the
  search algorithm itself is unchanged. All other values match the paper (w=0.4, temp=0.9,
  reward 1-5, gamma in {3,4,5}).

All hyperparameters are read from config (self.params), never hardcoded.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..llm import render
from ..llm.client import BACKBONE
from .base import Baseline, BaselineResult
from .util import parse_int_score


@dataclass
class _Node:
    """One reasoning step in the search tree. `text` is the step's sentence ("" for the root)."""

    text: str
    parent: Optional["_Node"] = None
    depth: int = 0
    visits: int = 0          # N(s)
    value: float = 0.0       # V(s), running mean reward on the 1-5 QUALITY scale (higher=better)
    children: List["_Node"] = field(default_factory=list)
    terminal: bool = False   # path ended ("FINAL:" marker or max depth)

    def prefix(self) -> str:
        """Reasoning accumulated from the root down to (and including) this node."""
        steps, n = [], self
        while n is not None and n.parent is not None:  # skip the root (question-only)
            steps.append(n.text)
            n = n.parent
        return "\n".join(reversed(steps))

    def is_expanded(self) -> bool:
        return bool(self.children) or self.terminal


class HaluSearchBaseline(Baseline):
    name = "halusearch"

    # ---- reward (zero-shot self-eval RM substitute): 1-5 continuation risk -> 1-5 quality ----
    async def _reward(self, question: str, prefix: str, continuation: str) -> float:
        raw = await self.client.chat_text(
            render(
                "baselines/halusearch_verify",
                question=question,
                prefix=prefix or "(none yet)",
                candidate=continuation,
            ),
            BACKBONE,
            temperature=self.temperature,
            max_tokens=8,
        )
        risk = parse_int_score(raw, 1, 5, default=3)  # default = moderate risk
        return 6.0 - float(risk)  # quality = 6 - risk  (1..5, higher is better)

    # ---- EXPAND: generate one candidate next step (sentence) ----
    async def _gen_step(self, question: str, prefix: str, temperature: float) -> str:
        return await self.client.chat_text(
            render("baselines/halusearch_step", question=question, prefix=prefix or "(none yet)"),
            BACKBONE,
            temperature=temperature,
            max_tokens=256,
        )

    async def answer(self, question: str) -> BaselineResult:
        # --- config (faithful reduced-MCTS defaults; paper K=10/m=5/M=20) ---
        M = int(self.params.get("iterations_M", 8))
        K = int(self.params.get("children_K", 3))
        m = int(self.params.get("rollouts_m", 2))
        w = float(self.params.get("uct_w", 0.4))
        gamma = float(self.params.get("gamma", 4.0))          # on the 1-5 quality scale
        r_th = float(self.params.get("r_th", 4.0))            # early-stop quality threshold
        expand_t = float(self.params.get("expand_temperature", 0.9))
        max_depth = int(self.params.get("max_depth", 6))
        instance_switch = bool(self.params.get("instance_switch", True))

        trace: dict = {
            "config": {
                "iterations_M": M, "children_K": K, "rollouts_m": m, "uct_w": w,
                "gamma": gamma, "r_th": r_th, "expand_temperature": expand_t,
                "max_depth": max_depth, "instance_switch": instance_switch,
            }
        }

        # ===== INSTANCE-LEVEL DUAL-PROCESS GATE (System 1 vs System 2) =====
        # Cheap confidence probe: one direct answer + self-rated risk. Low risk -> trust System 1.
        if instance_switch:
            gate_raw = await self.client.chat_text(
                render("baselines/halusearch_gate", question=question),
                BACKBONE,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            # Split the gate output into the ANSWER (chain-of-thought + "Answer:") and the trailing
            # "Confidence:" meta-line. Confidence drives routing ONLY; the System-1 answer is the
            # CoT answer itself (NOT the bare meta-line — the earlier bug returned gate_raw verbatim,
            # so System-1 math answers carried no reasoning and were wrong, audit BLOCKER-5).
            parts = re.split(r"\n?\s*Confidence\s*:", gate_raw, maxsplit=1)
            sys1_answer = parts[0].strip()
            gate_risk = parse_int_score(parts[1], 1, 5, default=3) if len(parts) > 1 else 3
            gate_quality = 6.0 - float(gate_risk)
            trace["instance_gate"] = {"quality": gate_quality, "raw": gate_raw}
            if gate_quality >= gamma:
                # System 1 (fast thinking): return the CoT answer, skip the search.
                trace["system"] = 1
                return BaselineResult(answer=sys1_answer or gate_raw, trace=trace)
        trace["system"] = 2

        # ===== SYSTEM 2: MCTS over reasoning steps =====
        root = _Node(text="")
        root.visits = 1
        best_quality = 0.0

        def uct(child: _Node, parent: _Node) -> float:
            if child.visits == 0:
                return float("inf")  # force exploration of unvisited children first
            return child.value + w * math.sqrt(math.log(parent.visits) / child.visits)

        for it in range(M):
            # --- SELECT: descend to a not-yet-expanded (or terminal) node ---
            node = root
            while node.is_expanded() and node.children:
                node = max(node.children, key=lambda c: uct(c, node))
            if node.terminal:
                break  # greedy frontier is terminal; nothing left to expand productively

            # --- step-level fast/slow switch: reliable parent -> FAST (1 child), else SLOW (K) ---
            slow = node.parent is None or node.value < gamma
            n_children = K if slow else 1
            prefix = node.prefix()

            cands = await asyncio.gather(
                *[self._gen_step(question, prefix, expand_t) for _ in range(n_children)]
            )

            # --- EXPAND + SIMULATE each new child ---
            for c in cands:
                c = (c or "").strip()
                if not c:
                    continue
                child = _Node(text=c, parent=node, depth=node.depth + 1)
                child.terminal = ("FINAL:" in c.upper()) or (child.depth >= max_depth)
                node.children.append(child)

                child_prefix = child.prefix()
                if child.terminal:
                    # no further rollout possible; score the step itself as its own continuation
                    rewards = [await self._reward(question, prefix, c)]
                else:
                    rollouts = await asyncio.gather(
                        *[self._gen_step(question, child_prefix, expand_t) for _ in range(m)]
                    )
                    rewards = await asyncio.gather(
                        *[self._reward(question, child_prefix, r or c) for r in rollouts]
                    )
                child.value = sum(rewards) / max(1, len(rewards))  # SIMULATE: mean rollout reward

                # --- BACKPROP: running-mean update up the path ---
                r = child.value
                n = child
                while n is not None:
                    n.visits += 1
                    n.value = (n.value * (n.visits - 1) + r) / n.visits
                    n = n.parent

                best_quality = max(best_quality, child.value)

            # --- TERMINATION: a node reached the quality threshold ---
            if best_quality >= r_th:
                trace["stopped_early"] = {"iteration": it, "best_quality": best_quality}
                break

        # ===== FINAL ANSWER: greedy highest-value root->terminal path =====
        node, path = root, []
        while node.children:
            node = max(node.children, key=lambda c: c.value)
            path.append(node.text)
        reasoning = "\n".join(path)

        final = await self.client.chat_text(
            render("baselines/halusearch_final", question=question, reasoning=reasoning or "(none)"),
            BACKBONE,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # tree stats for the trace
        total_nodes = 0
        max_d = 0
        stack = [root]
        while stack:
            n = stack.pop()
            total_nodes += 1
            max_d = max(max_d, n.depth)
            stack.extend(n.children)
        trace.update({
            "best_path": path,
            "reasoning": reasoning,
            "tree": {
                "total_nodes": total_nodes - 1,          # exclude the question-only root
                "max_depth": max_d,
                "root_visits": root.visits,
                "best_path_quality": node.value if path else 0.0,
                "best_quality_seen": best_quality,
            },
        })
        return BaselineResult(answer=final, trace=trace)
