"""Aggregate and print metrics for a run: `python -m compass.eval.report results/<run_id>`.

Accepts a results directory or a single results JSON (the structure compass/run.py writes:
{meta, metrics, records}). Prints one row per result file: HR / Accuracy / avg tokens / latency.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple


def load_results(path: str) -> List[Tuple[str, dict]]:
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".json")]
    else:
        files = [path]
    out = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                out.append((f, json.load(fh)))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {f}: {e}")
    return out


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def print_report(results: List[Tuple[str, dict]]) -> None:
    header = ["dataset", "method", "HR%↓", "Acc↑", "avg_tok", "lat_s", "n"]
    print(" | ".join(f"{h:>10}" for h in header))
    print("-" * (13 * len(header)))
    for _, r in results:
        meta, m = r.get("meta", {}), r.get("metrics", {})
        row = [
            meta.get("dataset", "?"), meta.get("method", "?"),
            _fmt(m.get("hallucination_rate")), _fmt(m.get("accuracy")),
            _fmt(m.get("avg_tokens_per_query")), _fmt(m.get("avg_latency_s")),
            _fmt(m.get("n_scored")),
        ]
        print(" | ".join(f"{str(c):>10}" for c in row))


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate COMPASS run metrics")
    ap.add_argument("path", help="results directory or a results JSON file")
    args = ap.parse_args()
    results = load_results(args.path)
    if not results:
        print(f"no results found at {args.path}")
        return
    print_report(results)


if __name__ == "__main__":
    main()
