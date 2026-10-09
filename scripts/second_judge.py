"""Second-judge check: re-judge a stratified sample of the test answers with a judge from another model
family, and compare it with the main judge (Qwen3-32B) and with the human annotators.

Sample: from the first test run, 24 answers per (backbone, method, benchmark) cell, i.e. 3 x 7 x 4 x 24 =
2,016 answers (seed 20261009), plus the 160 human-annotated items of human_eval/. The second judge gets
exactly what the main judge gets: the same prompt, system message, reference block, and parser
(compass.eval.judge), with reference answers reloaded from the dataset by id as in judge_generated.py.

The paper used DeepSeek-V4-Pro through its OpenAI-compatible API, non-thinking, temperature 0:

    export DEEPSEEK_API_KEY=...
    python scripts/second_judge.py --results results            # judge (resumes) and analyze
    python scripts/second_judge.py --results results --analyze-only

The labels reported in the paper are in human_eval/second_judge.jsonl (the default --out); labels already
in --out are kept, so running the first command on a fresh checkout only analyzes them.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from compass.datasets import load_dataset  # noqa: E402
from compass.eval import judge_all  # noqa: E402
from compass.llm.client import LLMClient  # noqa: E402

BACKBONES = ["qwen3-8b", "llama-3.1-8b", "ministral-8b"]
METHODS = ["direct", "self_consistency", "react", "adaptive_rag", "targ", "dtr", "compass"]
DATASETS = ["truthfulqa", "gsm8k", "strategyqa", "freshqa"]
PATHS = {
    "truthfulqa": "data/truthfulqa/TruthfulQA.json",
    "gsm8k": "data/gsm8k/test.jsonl",
    "strategyqa": "data/strategyqa/strategyqa.json",
    "freshqa": "data/freshqa/freshqa.json",
}
HE = "human_eval"


def run_file(results: str, bb: str, method: str, ds: str) -> str:
    if method == "compass":
        return f"{results}/{bb}/compass_{ds}_test_rep1.json"
    return f"{results}/{bb}/baselines/{method}_{ds}_test_r1.json"


def build_items(results: str, per_cell: int, seed: int) -> list:
    """The stratified sample (in a fixed order, so the seed reproduces it) and the human-annotated items."""
    rng = random.Random(seed)
    items = []
    for bb in BACKBONES:
        for m in METHODS:
            for ds in DATASETS:
                for r in rng.sample(json.load(open(run_file(results, bb, m, ds)))["records"], per_cell):
                    items.append({"key": f"{bb}|{m}|{ds}|{r['id']}", "item": "sample", "backbone": bb,
                                  "method": m, "dataset": ds, "id": str(r["id"]), "answer": r["answer"] or "",
                                  "main_judge": r["judgment"]["hallucination"]})
    truth = json.load(open(f"{HE}/_ground_truth.json"))
    labels, answers = {}, {}
    for a in (1, 2, 3):
        for r in csv.DictReader(open(f"{HE}/annotator_{a}.csv")):
            labels.setdefault(r["seq"], []).append(1 if r["is_hallucination(yes/no)"].strip().lower() == "yes" else 0)
            answers[r["seq"]] = r["model_answer"]
    for seq in sorted(truth):
        t = truth[seq]
        items.append({"key": f"human|{seq}", "item": "human", "seq": seq, "dataset": t["dataset"],
                      "id": str(t["id"]), "answer": answers[seq],
                      "human_majority": int(Counter(labels[seq]).most_common(1)[0][0]),
                      "api_judge": int(t["judge_hallucination"])})
    return items


async def judge(items: list, args) -> None:
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["key"] for line in open(args.out)}
    todo = [it for it in items if it["key"] not in done]
    print(f"{len(items)} items, {len(done)} already judged, {len(todo)} to judge")
    if not todo:
        return
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"set {args.api_key_env} to judge the remaining items")
    client = LLMClient(base_url=args.base_url, model=args.model, api_key=api_key, default_temperature=0.0,
                       default_max_tokens=512, concurrency=args.concurrency,
                       extra_body={"thinking": {"type": "disabled"}})
    by_ds = {ds: {str(e.id): e for e in load_dataset(ds, path, None, split="all", dev_size=100, seed=0)}
             for ds, path in PATHS.items()}
    for start in range(0, len(todo), 200):  # append in batches, so an interrupted run resumes
        batch = todo[start:start + 200]
        js = await judge_all([by_ds[it["dataset"]][it["id"]] for it in batch], [it["answer"] for it in batch], client)
        with open(args.out, "a") as fh:
            for it, j in zip(batch, js):
                row = {k: v for k, v in it.items() if k != "answer"}
                row.update({"second_judge": j.hallucination, "second_accuracy": j.accuracy, "second_raw": j.raw[:300]})
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {min(start + 200, len(todo))}/{len(todo)}")


def kappa(a: list, b: list) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(pairs: list) -> str:
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    return f"{100 * sum(x == y for x, y in pairs) / len(pairs):5.1f}%  kappa {kappa(a, b):.3f}  (n={len(pairs)})"


def kendall_tau(x: list, y: list) -> float:
    n, s = len(x), 0
    for i in range(n):
        for j in range(i + 1, n):
            d = (x[i] - x[j]) * (y[i] - y[j])
            s += (d > 0) - (d < 0)
    return s / (n * (n - 1) / 2)


def analyze(path: str) -> None:
    rows = [json.loads(line) for line in open(path)]
    ok = [r for r in rows if r["second_judge"] in (0, 1)]
    print(f"\n{len(rows)} labels, {len(rows) - len(ok)} unparseable")
    human = [r for r in ok if r["item"] == "human"]
    print("\nHuman-annotated items: agreement with the annotators' majority")
    print("  second judge  ", agreement([(r["second_judge"], r["human_majority"]) for r in human]))
    for ds in DATASETS:
        print(f"    {ds:11s} ", agreement([(r["second_judge"], r["human_majority"]) for r in human if r["dataset"] == ds]))
    sample = [r for r in ok if r["item"] == "sample" and r["main_judge"] in (0, 1)]
    print("\nStratified sample: second judge vs. main judge")
    print("  all           ", agreement([(r["second_judge"], r["main_judge"]) for r in sample]))
    for key in ("dataset", "backbone", "method"):
        for v in sorted({r[key] for r in sample}):
            print(f"    {v:16s}", agreement([(r["second_judge"], r["main_judge"]) for r in sample if r[key] == v]))
    rng = random.Random(0)

    def shift(sub: list) -> str:
        d = [r["second_judge"] - r["main_judge"] for r in sub]
        n = len(d)
        boots = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000))
        hr_main = 100 * sum(r["main_judge"] for r in sub) / n
        hr_second = 100 * sum(r["second_judge"] for r in sub) / n
        return (f"HR main {hr_main:5.1f}  second {hr_second:5.1f}  shift {hr_second - hr_main:+5.1f} "
                f"[{boots[99]:+.1f}, {boots[3899]:+.1f}]")
    print("\nHR shift when the second judge replaces the main judge (points, 95% bootstrap interval)")
    for key in ("backbone", "method"):
        for v in sorted({r[key] for r in sample}):
            print(f"    {v:16s}", shift([r for r in sample if r[key] == v]))
    print("\nMethod ranking per backbone by sample HR (Kendall's tau between the two judges)")
    for bb in BACKBONES:
        hr = {j: [100 * sum(r[j] for r in sample if r["backbone"] == bb and r["method"] == m)
                  / sum(1 for r in sample if r["backbone"] == bb and r["method"] == m) for m in METHODS]
              for j in ("main_judge", "second_judge")}
        order = {j: [m for _, m in sorted(zip(hr[j], METHODS))] for j in hr}
        print(f"  {bb:13s} tau {kendall_tau(hr['main_judge'], hr['second_judge']):.2f}  "
              f"lowest HR: main {order['main_judge'][0]}, second {order['second_judge'][0]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=f"{HE}/second_judge.jsonl")
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--per-cell", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        asyncio.run(judge(build_items(args.results, args.per_cell, args.seed), args))
    analyze(args.out)


if __name__ == "__main__":
    main()
