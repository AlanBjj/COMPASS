"""Calibrate an uncertainty-gated retrieval baseline (TARG, DTR) on the dev split.

The gated method answers with its retrieval branch iff its uncertainty U > tau. Running the two
branches unconditionally once (<m>_never and <m>_always, both recording U in trace["uncertainty"])
lets us compose the gated method for ANY tau offline: per question, take the "always" judgment if
U > tau, else the "never" judgment. tau is set from a target retrieval rate rho as in TARG
(tau_rho = (1 - rho)-quantile of U on dev).

    python scripts/calibrate_gate.py --method targ --dir results/qwen3-8b/baselines --rep 1 2 3

Prints, per dataset, HR and accuracy for rho in 0, 0.1, ..., 1 and the best rho/tau on dev. With
several runs, tau is the quantile of U pooled over the runs and rho is chosen by the mean HR. The
chosen tau then goes into scripts/make_configs.py (TARG_TAU / DTR_TAU) for the gated runs.
"""

from __future__ import annotations

import argparse
import json
import os

DATASETS = ["truthfulqa", "gsm8k", "strategyqa", "freshqa"]


def load(path):
    return {str(r["id"]): r for r in json.load(open(path))["records"]}


def tok(path):
    return (json.load(open(path)).get("metrics") or {}).get("avg_tokens_per_query")


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = q * (len(xs) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="targ")
    ap.add_argument("--dir", default="results/qwen3-8b/baselines")
    ap.add_argument("--rep", type=int, nargs="+", default=[1],
                    help="one or more runs; with several, one tau per rho is applied to every run and "
                         "rho is chosen by the mean HR over the runs")
    ap.add_argument("--write", action="store_true",
                    help="write the composed gated runs at the chosen tau as <method>_gated_<ds>_dev_r<k>.json "
                         "(read by summarize_main_dev.py; tokens/query estimated from the two branches)")
    args = ap.parse_args()
    taus = {}
    for ds in DATASETS:
        runs = []   # per rep: (never, always, ids, u)
        for rep in args.rep:
            pn = os.path.join(args.dir, f"{args.method}_never_{ds}_dev_r{rep}.json")
            pa = os.path.join(args.dir, f"{args.method}_always_{ds}_dev_r{rep}.json")
            if not (os.path.exists(pn) and os.path.exists(pa)):
                print(f"{ds}: missing {pn if not os.path.exists(pn) else pa}")
                continue
            never, always = load(pn), load(pa)
            ids = sorted(set(never) & set(always))
            # U from both branches (same greedy draft/answer in both runs; take the mean)
            u = {}
            for i in ids:
                us = [r["trace"].get("uncertainty") for r in (never[i], always[i])
                      if isinstance(r.get("trace"), dict) and r["trace"].get("uncertainty") is not None]
                u[i] = sum(us) / len(us) if us else float("inf")
            runs.append((never, always, ids, u, rep, tok(pn), tok(pa)))
        if not runs:
            continue
        pooled = [x for r in runs for x in r[3].values()]
        rows = []
        for k in range(11):
            rho = k / 10
            tau = quantile(pooled, 1 - rho) if 0 < rho < 1 else (float("inf") if rho == 0 else float("-inf"))
            hrs, accs, reals = [], [], []
            for never, always, ids, u, *_ in runs:
                picked = [always[i] if u[i] > tau else never[i] for i in ids]
                hrs.append(100 * sum(r["judgment"]["hallucination"] for r in picked) / len(picked))
                accs.append(sum(r["judgment"]["accuracy"] for r in picked) / len(picked))
                reals.append(sum(1 for i in ids if u[i] > tau) / len(ids))
            rows.append((rho, tau, sum(reals) / len(reals), sum(hrs) / len(hrs), sum(accs) / len(accs), hrs))
        best = min(rows, key=lambda x: (round(x[3], 6), x[2]))   # lowest mean HR, then fewer retrievals
        print(f"\n{ds}  (n={len(runs[0][2])}, runs={len(runs)})")
        for rho, tau, real, hr, acc, hrs in rows:
            mark = "  <- best" if (rho, tau) == (best[0], best[1]) else ""
            per = "  [" + " ".join(f"{h:5.1f}" for h in hrs) + "]" if len(hrs) > 1 else ""
            print(f"  rho={rho:.1f}  tau={tau:8.4f}  retrieved={real:.2f}  HR={hr:5.1f}  Acc={acc:5.1f}{per}{mark}")
        # rho = 0 / 1 as config values: U >= 0 for TARG and DTR, so 1e9 never retrieves and -1.0 always does
        taus[ds] = round(best[1], 6) if abs(best[1]) != float("inf") else (1e9 if best[1] > 0 else -1.0)
        if args.write:
            for never, always, ids, u, rep, tok_n, tok_a in runs:
                tau = best[1]
                picked = []
                for i in ids:
                    r = dict(always[i] if u[i] > tau else never[i])
                    r["trace"] = dict(r.get("trace") or {}, composed_from="always" if u[i] > tau else "never")
                    picked.append(r)
                real = sum(1 for i in ids if u[i] > tau) / len(ids)
                est = (1 - real) * tok_n + real * tok_a if tok_n is not None and tok_a is not None else None
                out = os.path.join(args.dir, f"{args.method}_gated_{ds}_dev_r{rep}.json")
                json.dump({"meta": {"method": f"{args.method}_gated", "dataset": ds, "run_index": rep,
                                    "composed": True, "rho": best[0], "tau": tau, "retrieved": real,
                                    "note": "composed offline from the _never/_always runs at the chosen tau; "
                                            "avg_tokens_per_query is estimated from the two branches"},
                           "metrics": {"avg_tokens_per_query": est},
                           "records": picked}, open(out, "w"), ensure_ascii=False, indent=1)
            print(f"  wrote {args.method}_gated_{ds}_dev_r*.json")
    print(f"\n{args.method.upper()}_TAU = {taus}   # 1e9 = never retrieve (rho 0), -1.0 = always (rho 1)")


if __name__ == "__main__":
    main()
