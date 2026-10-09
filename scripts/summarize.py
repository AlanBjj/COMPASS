"""Main-table preview: COMPASS (full method and ablations) and baselines side by side (one screen).

    python scripts/summarize.py --dir results/qwen3-8b --split test \
        --compass full --baselines direct self_consistency react adaptive_rag targ dtr

Per method and dataset: HR mean±std over the finished repeats (population std, as in the paper),
then the average over the four datasets (computed per repeat index, so it needs the same number of
repeats on every dataset; otherwise the mean of the per-dataset means is shown with "*"), and the
average tokens per query. COMPASS files: <dir>/compass_<ds>[_<ablation>]_<split>_rep<k>.json; baseline
files: <dir>/baselines/<method>_<ds>_<split>_r<k>.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st

DATASETS = ["truthfulqa", "gsm8k", "strategyqa", "freshqa"]


def load(paths):
    out = []
    for p in paths:
        d = json.load(open(p))
        h = [r["judgment"]["hallucination"] for r in d["records"]
             if (r.get("judgment") or {}).get("hallucination") in (0, 1)]
        if len(h) != len(d["records"]):
            print(f"  note: {os.path.basename(p)} has {len(d['records']) - len(h)} unjudged records")
        out.append({"hr": 100 * sum(h) / len(h) if h else None,
                    "tok": (d.get("metrics") or {}).get("avg_tokens_per_query")})
    return out


def files(root, ds, name, baseline, split="dev"):
    if baseline:
        pat = re.compile(rf"^{re.escape(name)}_{ds}_{split}_r(\d+)\.json$")
        cands = glob.glob(os.path.join(root, "baselines", "*.json"))
    else:
        stem = f"compass_{ds}" + ("" if name == "full" else f"_{name}")
        pat = re.compile(rf"^{re.escape(stem)}_{split}_rep(\d+)\.json$")
        cands = glob.glob(os.path.join(root, "*.json"))
    return sorted(f for f in cands if pat.match(os.path.basename(f)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/qwen3-8b")
    ap.add_argument("--compass", nargs="*", default=["full"], help="'full' and/or ablation names")
    ap.add_argument("--baselines", nargs="*",
                    default=["direct", "self_consistency", "react", "adaptive_rag", "targ", "dtr"])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    args = ap.parse_args()

    rows = [("COMPASS" if v == "full" else f"COMPASS-{v}", v, False) for v in args.compass] + [(b, b, True) for b in args.baselines]
    print(f"{args.split.upper()} main-table preview  ({args.dir}; HR %, lower is better; mean±std over repeats [n])")
    print(f"{'method':22s}" + "".join(f"{ds:>17s}" for ds in DATASETS) + f"{'AVERAGE':>14s}{'tok/q':>8s}")
    for label, name, baseline in rows:
        cells, per_ds, toks = [], {}, []
        for ds in DATASETS:
            runs = load(files(args.dir, ds, name, baseline, args.split))
            per_ds[ds] = runs
            hrs = [r["hr"] for r in runs if r["hr"] is not None]
            toks += [r["tok"] for r in runs if r["tok"]]
            cells.append(f"{st.mean(hrs):6.1f}±{st.pstdev(hrs):4.1f} [{len(hrs)}]" if hrs else f"{'-':>16s}")
        nrep = [len(per_ds[ds]) for ds in DATASETS]
        if min(nrep) and len(set(nrep)) == 1:
            avgs = [st.mean(per_ds[ds][k]["hr"] for ds in DATASETS) for k in range(nrep[0])]
            avg = f"{st.mean(avgs):6.1f}±{st.pstdev(avgs):4.1f}"
        elif min(nrep):
            avg = f"{st.mean(st.mean(r['hr'] for r in per_ds[ds]) for ds in DATASETS):6.1f}*    "
        else:
            avg = f"{'-':>11s}"
        tok = f"{st.mean(toks):8.0f}" if toks else f"{'-':>8s}"
        print(f"{label:22s}" + "".join(f"{c:>17s}" for c in cells) + f"{avg:>14s}{tok}")


if __name__ == "__main__":
    main()
