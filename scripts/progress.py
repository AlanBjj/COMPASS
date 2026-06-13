#!/usr/bin/env python
"""Overall progress of the COMPASS full-test x3 generation sweep.

Reads results/<ds>_<ds>_rep<r>.json (done) / .partial.json (running), sums samples done
vs target, and estimates ETA from the sweep_master.log start time. Run anytime:
    python scripts/progress.py
"""
import glob
import json
import os
import re
from datetime import datetime

DATASETS = {"gsm8k": 1219, "truthfulqa": 717, "strategyqa": 2190, "freshqa": 500}
REPS = [1, 2, 3]
R = "results"

total = sum(DATASETS.values()) * len(REPS)
done = 0
rows = []
for ds, N in DATASETS.items():
    for rep in REPS:
        fin = f"{R}/{ds}_{ds}_rep{rep}.json"
        par = f"{R}/{ds}_{ds}_rep{rep}.partial.json"
        if os.path.exists(fin):
            try:
                n = json.load(open(fin))["meta"]["n"]
            except Exception:
                n = N
            st = "done"
        elif os.path.exists(par):
            try:
                n = json.load(open(par))["meta"]["n"]
            except Exception:
                n = 0
            st = "run "
        else:
            n, st = 0, "wait"
        done += n
        rows.append((f"{ds} rep{rep}", st, n, N))

pct = 100.0 * done / total

elapsed_min = None
try:
    txt = open("/tmp/sweep_master.log").read()
    m = re.search(r"\[(\d{2}):(\d{2}):(\d{2})", txt)
    if m:
        h, mi, s = map(int, m.groups())
        now = datetime.now()
        start = now.replace(hour=h, minute=mi, second=s, microsecond=0)
        d = (now - start).total_seconds() / 60.0
        if d > 0:
            elapsed_min = d
except Exception:
    pass

print(f"\n{'run':16s} {'st':4s} {'samples':>13s}  progress")
print("-" * 60)
for name, st, n, N in rows:
    bar = int(20 * n / N)
    print(f"{name:16s} {st} {n:5d}/{N:<5d}  [{'#'*bar}{'.'*(20-bar)}]")
print("-" * 60)
print(f"TOTAL  {done}/{total}  =  {pct:.1f}%")
if elapsed_min and done > 0:
    rate = done / elapsed_min
    eta = (total - done) / rate if rate > 0 else 0
    print(f"elapsed {elapsed_min:.0f} min | {rate:.0f} samples/min | ETA ~{eta:.0f} min")
else:
    print("(not enough data for ETA yet)")
print()
