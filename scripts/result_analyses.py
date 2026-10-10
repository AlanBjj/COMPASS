"""Analyses behind the main-results text and the parameter checks, computed from the judged result files.

    python scripts/result_analyses.py --results results --data data

Prints, per backbone (test split, three runs pooled):
  - TruthfulQA: how often CPC runs and rejects the premise, and the HR of COMPASS and TARG on the rejected questions;
  - FreshQA: HR of COMPASS, TARG, and DTR on false- and true-premise questions, CPC rejection rates for both, the rejection
    rate by the year the answer last changed, and the HR when a true premise is rejected;
  - GSM8K: share of numeric votes without a strict majority, and the HR of COMPASS and Self-Consistency with and without one;
and, over all backbones:
  - the scoring weight lambda: share of scored decompositions whose sub-answer order or low-quality fallback changes when
    Q = lambda * s_acc + (1 - lambda) * s_rel is recomputed for other values (the ratings are stored per sub-answer);
  - the low-quality fallback: number of test answers that used it;
  - the CPC evidence limit: length of the formatted snapshot snippets (tag "[source | weight | date] " + text).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
from collections import defaultdict

BACKBONES = ["qwen3-8b", "llama-3.1-8b", "ministral-8b"]
DATASETS = ["truthfulqa", "gsm8k", "strategyqa", "freshqa"]
RECENT = {"2024", "2025", "2026"}


def records(path):
    return {r["id"]: r for r in json.load(open(path))["records"]}


def runs(results, bb, ds, method):
    for rep in (1, 2, 3):
        if method == "compass":
            yield records(os.path.join(results, bb, f"compass_{ds}_test_rep{rep}.json"))
        else:
            yield records(os.path.join(results, bb, "baselines", f"{method}_{ds}_test_r{rep}.json"))


def pct(xs):
    return 100 * st.mean(xs) if xs else float("nan")


def cpc(r):
    """(CPC ran, premise rejected) for a COMPASS record."""
    tc = (r.get("trace") or {}).get("trace_consistency") or {}
    ran = tc.get("operator") == "reasoning_cpc"
    return ran, ran and tc.get("winner_key") == "false_premise"


def hall(r):
    return r["judgment"]["hallucination"]


def truthfulqa(results, bb):
    o = defaultdict(list)
    for c, t in zip(runs(results, bb, "truthfulqa", "compass"), runs(results, bb, "truthfulqa", "targ")):
        for i, r in c.items():
            ran, rej = cpc(r)
            o["all"].append(ran)
            if ran:
                o["rej"].append(rej)
            if rej:
                o["c"].append(hall(r))
                o["t"].append(hall(t[i]))
    print(f"  TruthfulQA: CPC runs on {pct(o['all']):.0f}% of answers and rejects the premise in {pct(o['rej']):.1f}% of them;"
          f" HR on rejected questions: COMPASS {pct(o['c']):.1f}, TARG {pct(o['t']):.1f}")


def freshqa(results, data, bb):
    fq = {x["id"]: x for x in json.load(open(os.path.join(data, "freshqa", "freshqa.json")))}
    o = defaultdict(list)
    for c, t, d in zip(*(runs(results, bb, "freshqa", m) for m in ("compass", "targ", "dtr"))):
        for i, r in c.items():
            true_premise = fq[i]["false_premise"] == "FALSE"
            key = "true" if true_premise else "false"
            o[key + "_c"].append(hall(r)); o[key + "_t"].append(hall(t[i])); o[key + "_d"].append(hall(d[i]))
            ran, rej = cpc(r)
            if not ran:
                continue
            o[key + "_rej"].append(rej)
            if true_premise:
                o["true_rej_recent" if fq[i]["effective_year"] in RECENT else "true_rej_earlier"].append(rej)
                if rej:
                    o["rej_c"].append(hall(r)); o["rej_t"].append(hall(t[i]))
    print(f"  FreshQA false premise (n={len(o['false_c']) // 3}): HR COMPASS {pct(o['false_c']):.1f}, TARG {pct(o['false_t']):.1f},"
          f" DTR {pct(o['false_d']):.1f}; CPC rejects {pct(o['false_rej']):.1f}% of the false premises it checks")
    print(f"  FreshQA true premise (n={len(o['true_c']) // 3}): HR COMPASS {pct(o['true_c']):.1f}, TARG {pct(o['true_t']):.1f},"
          f" DTR {pct(o['true_d']):.1f}; CPC rejects {pct(o['true_rej']):.1f}% of the true premises it checks"
          f" ({pct(o['true_rej_recent']):.1f}% if the answer changed in 2024 or later, {pct(o['true_rej_earlier']):.1f}% otherwise);"
          f" HR when a true premise is rejected: COMPASS {pct(o['rej_c']):.1f}, TARG {pct(o['rej_t']):.1f}")


def gsm8k(results, bb):
    o = defaultdict(list)
    for c, sc in zip(runs(results, bb, "gsm8k", "compass"), runs(results, bb, "gsm8k", "self_consistency")):
        for i, r in c.items():
            tc = (r.get("trace") or {}).get("trace_consistency") or {}
            if tc.get("operator") != "math":
                continue
            split = "min_votes" in str(tc.get("fallback") or "")
            o["split"].append(split)
            key = "split" if split else "majority"
            o[key + "_c"].append(hall(r)); o[key + "_sc"].append(hall(sc[i]))
    print(f"  GSM8K: {pct(o['split']):.1f}% of the numeric votes have no strict majority; HR with a majority: COMPASS"
          f" {pct(o['majority_c']):.1f}, Self-Consistency {pct(o['majority_sc']):.1f}; without: COMPASS {pct(o['split_c']):.1f},"
          f" Self-Consistency {pct(o['split_sc']):.1f}")


def lambda_check(results):
    def order(scored, lam):
        q = [lam * s["s_acc"] + (1 - lam) * s["s_rel"] for s in scored]
        return [i for _, i in sorted((-v, i) for i, v in enumerate(q))], max(q) < 2
    counts = defaultdict(lambda: defaultdict(int))
    for bb in BACKBONES:
        for ds in DATASETS:
            for run in runs(results, bb, ds, "compass"):
                for r in run.values():
                    scored = (r.get("trace") or {}).get("scored")
                    if not scored:
                        continue
                    base = order(scored, 0.6)
                    counts[bb]["n"] += 1
                    for lam in (0.4, 0.5, 0.7, 0.8):
                        o, fb = order(scored, lam)
                        counts[bb][f"order{lam}"] += o != base[0]
                        counts[bb][f"fallback{lam}"] += fb != base[1]
    for bb, c in counts.items():
        print(f"  lambda, {bb} ({c['n']} scored decompositions): order changes "
              + ", ".join(f"{lam}: {100 * c[f'order{lam}'] / c['n']:.1f}%" for lam in (0.4, 0.5, 0.7, 0.8))
              + "; fallback changes " + ", ".join(str(c[f"fallback{lam}"]) for lam in (0.4, 0.5, 0.7, 0.8)))


def fallback_count(results):
    n = used = 0
    for bb in BACKBONES:
        for ds in DATASETS:
            for run in runs(results, bb, ds, "compass"):
                for r in run.values():
                    n += 1
                    used += r.get("path") == "low_quality_fallback"
    print(f"  low-quality fallback: {used} of {n:,} test answers")


def evidence_length(data):
    lengths = []
    for line in open(os.path.join(data, "snapshot_corpus", "corpus.jsonl")):
        d = json.loads(line)
        tag = f"[{d['source']} | 0.45 | {d['date']}] " if d.get("date") else f"[{d['source']} | 0.45] "
        lengths.append(len(tag) + len(d["text"]) + 1)
    lengths.sort()
    p99 = lengths[int(0.99 * len(lengths))]
    print(f"  formatted snippet length: median {lengths[len(lengths) // 2]}, 99th percentile {p99};"
          f" 18 such snippets: {18 * p99} characters")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    for bb in BACKBONES:
        print(bb)
        truthfulqa(args.results, bb)
        freshqa(args.results, args.data, bb)
        gsm8k(args.results, bb)
    print("all backbones")
    lambda_check(args.results)
    fallback_count(args.results)
    evidence_length(args.data)


if __name__ == "__main__":
    main()
