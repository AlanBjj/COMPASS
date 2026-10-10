"""How much the gate decision depends on the nine rule weights, recomputed offline from the stored gate scores.

    python scripts/gate_weight_check.py --results results --out paper_tables/

Every COMPASS record stores the gate's rule score H(q) and model score L(q). The rules are deterministic functions of the
question text, so H can be recomputed under other weights (the script first checks that the stored H is reproduced exactly),
while L is kept. The complex-branch indicator delta(q) = [alpha*H + (1-alpha)*L >= tau] is then recomputed, and the table
reports the share of test questions (three runs pooled) whose delta changes. Questions on the recency plan are counted as
unchanged, because that plan is selected before delta is read. A question whose delta does not change keeps its plan, so the
share bounds how many answers a weight change could affect. For comparison, the same share is computed for the tau and alpha
values of the sensitivity runs. Writes gateweights_<backbone>_test.tex (rows: perturbations; columns: the four benchmarks).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import statistics as st

DATASETS = ["truthfulqa", "gsm8k", "strategyqa", "freshqa"]
BACKBONES = ["qwen3-8b", "llama-3.1-8b", "ministral-8b"]
ALPHA, TAU = 0.4, 0.55
N_DRAWS, LO, HI = 1000, 0.15, 0.30

_spec = importlib.util.spec_from_file_location(
    "heuristic", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "compass", "gate", "heuristic.py"))
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)   # the gate's own rules and weights, without importing the LLM client
W0 = [h._W_MULTI_STEP, h._W_LOGIC, h._W_TYPE_MIX, h._W_MANY_ENTITIES, h._W_MANY_NUMBERS, h._W_GOAL_CONSTRAINT,
      h._W_COMPARISON, h._W_MANY_NAMED, h._W_RECENCY]


def fired(question):
    """Which of the nine rules fire, in the order of heuristic_score (rho_1 .. rho_9)."""
    q = (question or "").lower()
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", q)
    goal = bool(re.search(r"\b(maximize|minimize|optimal|best|cheapest|fastest)\b", q))
    constraint = bool(re.search(r"\b(constraint|subject to|without|at most|at least|within)\b", q))
    proper = {w for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", question or "")}
    return [h._matches_any(h._MULTI_STEP, q), h._matches_any(h._LOGIC, q), bool(numbers and h._matches_any(h._FACTUAL, q)),
            sum(q.count(s) for s in h._ENTITY_SEPARATORS) >= 2, len(numbers) >= 3, goal and constraint,
            h._matches_any(h._COMPARISON, q), len(proper) >= 3, h._matches_any(h._RECENCY, q)]


def score(f, w):
    """H in the engine's floating-point order (rule by rule, then clipped), so that the default weights reproduce delta exactly."""
    total = 0.0
    for fi, wi in zip(f, w):
        if fi:
            total += wi
    return max(0.0, min(1.0, total))


def load(root, ds):
    """(rules fired, L, stored delta, on the recency plan) for every record of the three test runs."""
    items = []
    for rep in (1, 2, 3):
        path = os.path.join(root, f"compass_{ds}_test_rep{rep}.json")
        for r in json.load(open(path))["records"]:
            g = (r.get("trace") or {}).get("gate")
            if not g:
                continue
            f = fired(r["question"])
            assert abs(score(f, W0) - g["H"]) < 1e-9, (path, r["id"], g["H"], score(f, W0))
            assert (ALPHA * g["H"] + (1.0 - ALPHA) * g["L"] >= TAU) == g["decompose"], (path, r["id"])
            items.append((f, g["L"], g["decompose"], r.get("path") == "recency_retrieval"))
    return items


def changed(items, w=W0, alpha=ALPHA, tau=TAU):
    """% of the items whose delta changes under weights w and gate parameters alpha, tau."""
    n = sum(1 for f, L, d, rec in items if not rec and ((alpha * score(f, w) + (1.0 - alpha) * L >= tau) != d))
    return 100.0 * n / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_tables")
    args = ap.parse_args()
    rng = random.Random(0)
    draws = [[rng.uniform(LO, HI) for _ in W0] for _ in range(N_DRAWS)]
    mean_w = [st.mean(W0)] * len(W0)
    for bb in BACKBONES:
        data = {ds: load(os.path.join(args.results, bb), ds) for ds in DATASETS}
        rows = [(f"Equal weights ({st.mean(W0):.2f} each)", {ds: changed(it, mean_w) for ds, it in data.items()})]
        rand = {ds: sorted(changed(it, w) for w in draws) for ds, it in data.items()}
        rows.append((f"Random weights in $[{LO:.2f},{HI:.2f}]$, median", {ds: st.median(v) for ds, v in rand.items()}))
        rows.append((f"Random weights in $[{LO:.2f},{HI:.2f}]$, maximum", {ds: v[-1] for ds, v in rand.items()}))
        rows.append(("\\midrule", None))
        for tau in (0.50, 0.60):
            rows.append((f"$\\tau={tau:.2f}$ (default weights)", {ds: changed(it, tau=tau) for ds, it in data.items()}))
        for alpha in (0.3, 0.5):
            rows.append((f"$\\alpha={alpha:.1f}$ (default weights)", {ds: changed(it, alpha=alpha) for ds, it in data.items()}))
        lines = [label if vals is None else " & ".join([label] + [f"{vals[ds]:.1f}" for ds in DATASETS]) + " \\\\"
                 for label, vals in rows]
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"gateweights_{bb}_test.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(bb, "n =", {ds: len(it) for ds, it in data.items()})
        print("\n".join("  " + line for line in lines))


if __name__ == "__main__":
    main()
