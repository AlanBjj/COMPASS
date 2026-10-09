"""Judge validation: does the local judge (Qwen3-32B) agree with the human annotators?

Judges the 160 human-annotated items (human_eval/) with the local judge, using the same judge prompt
and reference formatting as every run (compass.eval.judge), then reports agreement and Cohen's kappa
against (a) the three annotators' majority label and (b) the labels of a commercial API judge stored
in human_eval/_ground_truth.json.

    python scripts/judge_calibration.py --config configs/qwen3-8b/truthfulqa.yaml
(only the config's `models.judge` block is used)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from collections import Counter

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from compass.datasets import load_dataset  # noqa: E402
from compass.eval import judge_all  # noqa: E402
from compass.llm.client import LLMClient  # noqa: E402

HE = "human_eval"
PATHS = {
    "truthfulqa": "data/truthfulqa/TruthfulQA.json",
    "gsm8k": "data/gsm8k/test.jsonl",
    "strategyqa": "data/strategyqa/strategyqa.json",
    "freshqa": "data/freshqa/freshqa.json",
}


def cohen_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/judge_calibration.json")
    args = ap.parse_args()
    jm = yaml.safe_load(open(args.config))["models"]["judge"]
    no_think = jm.get("enable_thinking") is False
    client = LLMClient(base_url=jm["base_url"], model=jm["model"], api_key="EMPTY",
                       default_temperature=jm.get("temperature", 0.0),
                       default_max_tokens=jm.get("max_tokens", 512), concurrency=16,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}} if no_think else None,
                       strip_thinking=no_think)

    truth = json.load(open(f"{HE}/_ground_truth.json"))
    labels = {}
    answers = {}
    for a in (1, 2, 3):
        for r in csv.DictReader(open(f"{HE}/annotator_{a}.csv")):
            labels.setdefault(r["seq"], []).append(1 if r["is_hallucination(yes/no)"].strip().lower() == "yes" else 0)
            answers[r["seq"]] = r["model_answer"]
    human = {s: int(Counter(v).most_common(1)[0][0]) for s, v in labels.items()}

    by_ds = {}
    for ds, path in PATHS.items():
        exs = load_dataset(ds, path, None, split="all", dev_size=100, seed=0)
        by_ds[ds] = {str(e.id): e for e in exs}
    seqs = sorted(truth)
    examples = [by_ds[truth[s]["dataset"]][str(truth[s]["id"])] for s in seqs]
    judgments = await judge_all(examples, [answers[s] for s in seqs], client)

    local = [j.hallucination for j in judgments]
    ok = [i for i, v in enumerate(local) if v in (0, 1)]
    h = [human[seqs[i]] for i in ok]
    api = [int(truth[seqs[i]]["judge_hallucination"]) for i in ok]
    nw = [local[i] for i in ok]
    rows = [
        ("local judge vs human majority", sum(x == y for x, y in zip(nw, h)) / len(ok), cohen_kappa(nw, h)),
        ("API judge vs human majority", sum(x == y for x, y in zip(api, h)) / len(ok), cohen_kappa(api, h)),
        ("local judge vs API judge", sum(x == y for x, y in zip(nw, api)) / len(ok), cohen_kappa(nw, api)),
    ]
    print(f"judge={client.model}  parsed {len(ok)}/{len(seqs)}")
    print(f"{'comparison':32s} {'agree':>7s} {'kappa':>7s}")
    for name, agree, k in rows:
        print(f"{name:32s} {100 * agree:6.1f}% {k:7.3f}")
    per_ds = Counter()
    for i in ok:
        ds = truth[seqs[i]]["dataset"]
        per_ds[(ds, nw[ok.index(i)] == h[ok.index(i)])] += 1
    for ds in PATHS:
        n = per_ds[(ds, True)] + per_ds[(ds, False)]
        if n:
            print(f"  {ds:11s} local-vs-human agree {100 * per_ds[(ds, True)] / n:5.1f}% (n={n})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"judge": client.model, "rows": rows,
               "items": [{"seq": seqs[i], "local": local[i], "api": int(truth[seqs[i]]["judge_hallucination"]),
                          "human": human[seqs[i]]} for i in range(len(seqs))]},
              open(args.out, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
