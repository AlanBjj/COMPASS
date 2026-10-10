"""Do snippets dated after the FreshQA reference date reach the methods, and do they account for the FreshQA results?

    python scripts/freshqa_dates.py --results results --data data

The snapshot was collected in June 2026 without temporal filtering, after the FreshQA reference date (April 21, 2026).
This script replays the snapshot retrieval for the 500 FreshQA test questions with the same BM25 index:
  - "question, whole snapshot, top 5": the retrieval of the question by the baselines that search the whole snapshot
    with it, such as TARG (exact);
  - "COMPASS, query u0 = q": the question searched in each of the three partitions (top 5 each); the rewritten queries
    u1..um are not stored, so they are not replayed.
A snippet counts as later than the reference date if its date is after April 21, 2026, or if it carries a relative
date of at most six weeks ("3 days ago", "2 weeks ago", "1 month ago"), which the June collection places after it.
The script then compares the HR of COMPASS and TARG (three runs) on the questions whose COMPASS retrieval returns at
least one later snippet and on the other questions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics as st
from datetime import datetime

REF = datetime(2026, 4, 21)
BACKBONES = ["qwen3-8b", "llama-3.1-8b", "ministral-8b"]

_spec = importlib.util.spec_from_file_location(
    "snapshot_index", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "compass", "snapshot_index.py"))
snapshot_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snapshot_index)   # BM25 only, without importing the LLM client


def later(date):
    """True if the snippet's date is after the reference date (absolute or relative date)."""
    d = (date or "").strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(d, fmt) > REF
        except ValueError:
            pass
    if re.search(r"\b(hour|hours|day|days)\s+ago", d):
        return True
    m = re.search(r"(\d+)\s+weeks?\s+ago", d)
    if m:
        return int(m.group(1)) <= 6
    m = re.search(r"(\d+)\s+months?\s+ago", d)
    return bool(m) and int(m.group(1)) <= 1


def hr(path, ids):
    recs = {r["id"]: r for r in json.load(open(path))["records"]}
    return [recs[i]["judgment"]["hallucination"] for i in ids if i in recs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    corpus = os.path.join(args.data, "snapshot_corpus")
    index = snapshot_index.SnapshotIndex(os.path.join(corpus, "corpus.jsonl"), os.path.join(corpus, "bm25"))
    questions = [x for x in json.load(open(os.path.join(args.data, "freshqa", "freshqa.json"))) if x["split"] == "TEST"]
    replays = {
        "question, whole snapshot, top 5": lambda q: index.search(q, "all", 5),
        "COMPASS, query u0 = q": lambda q: [h for g in snapshot_index.GROUPS for h in index.search(q, g, 5)],
    }
    with_later = {}
    for name, retrieve in replays.items():
        flags, hit = [], set()
        for x in questions:
            f = [later(h.get("date")) for h in retrieve(x["question"])]
            flags += f
            if any(f):
                hit.add(x["id"])
        with_later[name] = hit
        print(f"{name}: {100 * sum(flags) / len(flags):.1f}% of {len(flags)} retrieved snippets are later than the"
              f" reference date; {len(hit)} of {len(questions)} questions retrieve at least one")
    a = with_later["COMPASS, query u0 = q"]
    b = {x["id"] for x in questions} - a
    print("HR of COMPASS minus HR of TARG (points), on questions whose COMPASS retrieval returns later snippets / the others:")
    for bb in BACKBONES:
        diff = {}
        for name, ids in (("later", a), ("other", b)):
            c, t = [], []
            for rep in (1, 2, 3):
                c += hr(os.path.join(args.results, bb, f"compass_freshqa_test_rep{rep}.json"), ids)
                t += hr(os.path.join(args.results, bb, "baselines", f"targ_freshqa_test_r{rep}.json"), ids)
            diff[name] = 100 * (st.mean(c) - st.mean(t))
        print(f"  {bb}: {diff['later']:+.1f} (n={len(a)}) / {diff['other']:+.1f} (n={len(b)})")


if __name__ == "__main__":
    main()
