"""LaTeX tables and figure data for the paper, straight from the judged result files.

    python scripts/paper_tables.py --results results --out paper_tables/

Writes one .tex fragment per table (table rows only; the table environment lives in the paper), e.g.
  main_all_test.tex       HR mean±std / Acc per dataset + average, 7 methods, three backbones
  signif_all_test.tex     paired bootstrap differences against every baseline
  main_grouped_/signif_grouped_test.tex  the same with the benchmarks grouped (knowledge-intensive,
                          reasoning-intensive) and a per-group average
  ablation_qwen3-8b_test.tex  HR (delta vs. full) of the ablations; datasets an ablation was not run on
                          reuse the full model's runs and are marked (component inactive there)
  plans_/retrieval_/consistency_/oracle_/costds_<bb>_test.tex  the analyses of the paper
and the data files of the figures (data/*.dat). Every tinted column is colored on its own scale (tab_tint.py): within each
backbone, the median is white, better values are blue and worse values orange, shaded by rank; the paired-difference tables
use zero as the white point and keep bold for significance.
A cell whose runs are not all judged yet prints \\tbd (a macro the paper defines), so a partial run
never shows up as a number. HR and Acc are recomputed from the per-record judgments; population std
over runs; the average is taken per run over the four datasets, then mean±std over runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tab_tint import tint_fragment   # noqa: E402  (per-column heat-map tints, best bold / second underlined)

DATASETS = ["truthfulqa", "gsm8k", "strategyqa", "freshqa"]
N_TEST = {"truthfulqa": 717, "gsm8k": 1219, "strategyqa": 2190, "freshqa": 500}
BACKBONES = ["qwen3-8b", "llama-3.1-8b", "ministral-8b"]
# COMPASS result files: compass_<ds>[_<ablation>]_<split>_rep<k>.json; FINAL is the full method.
FINAL = "full"
# (label, key, is_baseline); dev uses the composed gated TARG/DTR files
METHODS = [("Direct (CoT)", "direct", True), ("Self-Consistency", "self_consistency", True),
           ("ReAct", "react", True), ("Adaptive-RAG", "adaptive_rag", True),
           ("TARG", "targ", True), ("DTR", "dtr", True), ("COMPASS (ours)", FINAL, False)]
GROUPS = {"direct": "Reasoning only", "react": "Retrieval-augmented", "adaptive_rag": "Adaptive retrieval"}
# (row label, ablation name in compass.run.ABLATIONS)
ABLATIONS = [("$-$TRACE (no voting)", "no-trace"), ("$-$CPC", "no-cpc"),
             ("$-$implicit-fact listing", "no-implicit-facts"), ("$-$CPC on retrieval plans", "no-cpc-on-retrieval"),
             ("$-$recency plan", "no-recency-plan"), ("$-$whole-question arithmetic test", "no-arithmetic-test"),
             ("$-$recency exemption from CPC", "no-recency-exemption"), ("$-$fact plan (decompose instead)", "no-fact-plan")]
ROUTES = [("Direct", ["direct"]), ("Direct + retrieval", ["direct_retrieval"]),
          ("Recency retrieval", ["recency_retrieval"]), ("Whole Math", ["homogeneous_Math"]),
          ("Whole Fact (retrieval)", ["homogeneous_Fact"]), ("Whole General/Logic", ["homogeneous_General", "homogeneous_Logic"]),
          ("Decompose + fuse", ["decompose"]), ("Low-quality fallback", ["low_quality_fallback"]),
          ("Fact retrieval (whole)", ["fact_retrieval"])]   # last, so routes.dat columns p0-p7 keep their order


def run_files(root, ds, key, baseline, split):
    if baseline:
        name = key if split == "test" else {"targ": "targ_gated", "dtr": "dtr_gated"}.get(key, key)
        pat = os.path.join(root, "baselines", f"{name}_{ds}_{split}_r[0-9].json")
    else:
        stem = f"compass_{ds}" + ("" if key == FINAL else f"_{key}")
        pat = os.path.join(root, f"{stem}_{split}_rep[0-9].json")
    return sorted(glob.glob(pat))


def score(path, n_expected):
    d = json.load(open(path))
    recs = d["records"]
    j = [r.get("judgment") or {} for r in recs]
    h = [x.get("hallucination") for x in j]
    if n_expected and len(recs) != n_expected or any(v not in (0, 1) for v in h):
        return None                                   # unfinished or unjudged
    acc = [x.get("accuracy") for x in j if isinstance(x.get("accuracy"), (int, float))]
    return {"hr": 100 * sum(h) / len(h), "acc": st.mean(acc) if acc else None,
            "tok": (d.get("metrics") or {}).get("avg_tokens_per_query"),
            "paths": (d.get("meta") or {}).get("path_counts") or {}}


def cell_runs(root, ds, key, baseline, split, reps=3):
    n = N_TEST[ds] if split == "test" else 100
    runs = [score(p, n) for p in run_files(root, ds, key, baseline, split)]
    runs = [r for r in runs if r]
    return runs if len(runs) >= reps else None


def mstd(xs):
    return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)


def mark_fmt(mark):
    return {"best": "[\\textbf]", "second": "[\\underline]"}.get(mark, "")


def tint(v, ref, mark=None, lower_better=True):
    return f"\\{'hcl' if lower_better else 'hc'}{mark_fmt(mark)}{{{v:.1f}}}{{{ref:.1f}}}"


def ranks_of(vals):
    """{key: 'best'|'second'} for the two lowest values (all keys must have a value)."""
    order = sorted((v, k) for k, v in vals.items())
    return {order[0][1]: "best", order[1][1]: "second"} if len(order) >= 2 else {}


def main_table(root, split):
    """Rows: methods (grouped); columns: HR (tinted vs. Direct, std small) per dataset, average HR,
    average accuracy, tokens/question with an inline bar. Unfinished cells print \\tbd."""
    data = {key: {ds: cell_runs(root, ds, key, b, split) for ds in DATASETS} for _, key, b in METHODS}
    avg = {}
    for key, cells in data.items():
        if all(cells[ds] for ds in DATASETS):
            hrs = [st.mean(cells[ds][i]["hr"] for ds in DATASETS) for i in range(3)]
            accs = [st.mean(cells[ds][i]["acc"] for ds in DATASETS) if all(cells[ds][i]["acc"] is not None for ds in DATASETS)
                    else None for i in range(3)]
            avg[key] = [{"hr": h, "acc": a} for h, a in zip(hrs, accs)]
        else:
            avg[key] = None
    cols = {ds: {k: data[k][ds] for k in data} for ds in DATASETS}
    cols["avg"] = avg
    complete = {c: all(col.values()) for c, col in cols.items()}
    means = {c: {k: st.mean(r["hr"] for r in runs) for k, runs in col.items() if runs} for c, col in cols.items()}
    marks = {c: ranks_of(means[c]) if complete[c] else {} for c in cols}
    toks = {}
    for key, cells in data.items():
        if all(cells[ds] for ds in DATASETS) and all(r["tok"] for ds in DATASETS for r in cells[ds]):
            toks[key] = st.mean(st.mean(r["tok"] for r in cells[ds]) for ds in DATASETS)
    tmax = max(toks.values()) if toks else 1
    lines = []
    for label, key, _ in METHODS:
        if key in GROUPS:
            lines.append(f"\\grp{{8}}{{{GROUPS[key]}}} \\\\")
        if key == FINAL:
            lines.append(f"\\grp{{8}}{{Ours}} \\\\")
        row = [("\\ours " if key == FINAL else "\\base ") + label]
        for c in DATASETS + ["avg"]:
            runs = cols[c][key]
            ref = means[c].get("direct")
            if runs and ref is not None:
                m, sd = mstd([r["hr"] for r in runs])
                row.append(tint(m, ref, marks[c].get(key)) + f"\\sd{{{sd:.1f}}}")
            else:
                row.append("\\tbd")
        acc = [r["acc"] for r in (avg[key] or []) if r["acc"] is not None]
        row.append(f"{st.mean(acc):.1f}" if avg[key] and acc else "\\tbd")
        if key in toks:
            row.append(f"{toks[key] / 1000:.1f}K\\,\\costbar{{{2.6 * toks[key] / tmax:.2f}em}}")
        else:
            row.append("\\tbd")
        lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n", data, avg


def cost_table(data, avg):
    rows = []
    for label, key, _ in METHODS:
        cells = data[key]
        if not all(cells[ds] for ds in DATASETS) or any(r["tok"] is None for ds in DATASETS for r in cells[ds]):
            rows.append((None, label, key))
            continue
        tok = st.mean(st.mean(r["tok"] for r in cells[ds]) for ds in DATASETS)
        rows.append((tok, label, key))
    done = sorted(r for r in rows if r[0] is not None)
    out = []
    for tok, label, key in done + [r for r in rows if r[0] is None]:
        hr = f"{st.mean(r['hr'] for r in avg[key]):.1f}" if avg.get(key) else "\\tbd"
        t = f"{tok:,.0f}".replace(",", "{,}") if tok is not None else "\\tbd"
        pre = "\\rowcolor{gray!15}" if key == FINAL else ""
        out.append(f"{pre}{label} & {t} & {hr} \\\\")
    return "\n".join(out) + "\n"


def ablation_table(root):
    """Full model, then one row per ablation: HR tinted vs. the full model (orange = worse), change in
    small type; datasets an ablation was not run on (no result files: the component fires on fewer than
    0.3% of their questions) reuse the full model's runs (grey). Last column: tokens/question averaged
    over the four datasets."""
    ran = {ab: {ds for ds in DATASETS if run_files(root, ds, ab, False, "test")} for _, ab in ABLATIONS}
    full = {ds: cell_runs(root, ds, FINAL, False, "test") for ds in DATASETS}

    def hr(runs):
        return st.mean(r["hr"] for r in runs)

    def tok(runs):
        return st.mean(r["tok"] for r in runs) if runs and all(r["tok"] is not None for r in runs) else None

    def ktok(ts):
        return f"{st.mean(ts) / 1000:.1f}K" if ts and all(t is not None for t in ts) else "\\tbd"
    fh = {ds: hr(full[ds]) if full[ds] else None for ds in DATASETS}
    favg = st.mean(fh.values()) if all(v is not None for v in fh.values()) else None
    head = ["\\ours \\textbf{Full \\method}"] + [f"\\textbf{{{v:.1f}}}" if v is not None else "\\tbd" for v in fh.values()]
    head.append(f"\\textbf{{{favg:.1f}}}" if favg is not None else "\\tbd")
    head.append(ktok([tok(full[ds]) for ds in DATASETS]) if all(full.values()) else "\\tbd")
    lines = [" & ".join(head) + " \\\\", "\\midrule"]

    def delta(v, ref):
        d = v - ref
        return f"{{\\scriptsize\\,({'+' if d >= 0 else '$-$'}{abs(d):.1f})}}"
    for label, ab in ABLATIONS:
        if not ran[ab]:   # an ablation without any result files is left out
            continue
        row, vals, toks = ["\\base " + label], [], []
        for ds in DATASETS:
            if ds in ran.get(ab, set()):
                runs = cell_runs(root, ds, ab, False, "test")
                toks.append(tok(runs))
                if runs and fh[ds] is not None:
                    v = hr(runs)
                    vals.append(v)
                    row.append(tint(v, fh[ds]) + delta(v, fh[ds]))
                else:
                    vals.append(None)
                    row.append("\\tbd")
            else:
                toks.append(tok(full[ds]))
                vals.append(fh[ds])
                row.append(f"\\reused{{{fh[ds]:.1f}}}" if fh[ds] is not None else "\\tbd")
        if all(v is not None for v in vals) and favg is not None:
            a = st.mean(vals)
            row.append(tint(a, favg) + delta(a, favg))
        else:
            row.append("\\tbd")
        row.append(ktok(toks))
        lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n"


def routes_table(root):
    share = {}
    for ds in DATASETS:
        runs = cell_runs(root, ds, FINAL, False, "test")
        if not runs:
            share[ds] = None
            continue
        tot = sum(sum(r["paths"].values()) for r in runs)
        share[ds] = {name: 100 * sum(r["paths"].get(p, 0) for r in runs for p in ps) / tot for name, ps in ROUTES}
    lines = []
    for name, _ in ROUTES:
        row = [name] + [f"{share[ds][name]:.1f}" if share[ds] else "\\tbd" for ds in DATASETS]
        lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n"


def per_question(root, ds, key, baseline, split="test"):
    """id -> (mean HR over the three runs, path of run 1) for a finished cell, else None."""
    n = N_TEST[ds] if split == "test" else 100
    files = run_files(root, ds, key, baseline, split)
    if len(files) < 3 or any(score(f, n) is None for f in files):
        return None
    hr, path = {}, {}
    for i, f in enumerate(files[:3]):
        for r in json.load(open(f))["records"]:
            hr.setdefault(r["id"], []).append(r["judgment"]["hallucination"])
            if i == 0:
                path[r["id"]] = r.get("path")
    return {k: (100 * st.mean(v), path.get(k)) for k, v in hr.items()}


def sgn(x):
    return f"{'+' if x >= 0 else '$-$'}{abs(x):.1f}"


def signif_table(root, n_boot=10000, seed=0, split="test"):
    """COMPASS minus each baseline: HR difference with a 95% paired-bootstrap CI over questions
    (per-question HR averaged over the three runs). The average column resamples within each dataset
    and averages the four dataset means, matching the main table's macro average."""
    import random
    rng = random.Random(seed)
    ours = {ds: per_question(root, ds, FINAL, False, split) for ds in DATASETS}
    lines = []
    for label, key, b in METHODS:
        if key == FINAL:
            continue
        base = {ds: per_question(root, ds, key, b, split) for ds in DATASETS}
        row = ["\\base " + label]
        diffs = {}
        for ds in DATASETS:
            if not (ours[ds] and base[ds]):
                row.append("\\tbd")
                continue
            ids = sorted(set(ours[ds]) & set(base[ds]))
            diffs[ds] = [ours[ds][i][0] - base[ds][i][0] for i in ids]
        cols = DATASETS + ["avg"]
        for c in cols:
            if c != "avg" and c not in diffs:
                continue
            if c == "avg":
                if len(diffs) < len(DATASETS):
                    row.append("\\tbd")
                    continue
                point = st.mean(st.mean(d) for d in diffs.values())
                boots = []
                for _ in range(n_boot):
                    boots.append(st.mean(st.mean(rng.choices(d, k=len(d))) for d in diffs.values()))
            else:
                d = diffs[c]
                point = st.mean(d)
                boots = [st.mean(rng.choices(d, k=len(d))) for _ in range(n_boot)]
            boots.sort()
            lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
            sig = hi < 0 or lo > 0
            fmt = "[\\mnumbf]" if sig else "[\\mnum]"
            cell = "\\hcl" + fmt + f"{{{point:.1f}}}{{0}}"
            row.append(cell + f"\\ci{{{sgn(lo)}}}{{{sgn(hi)}}}")
        lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n"


PLAN_GROUPS = [("Direct", {"direct"}), ("Retrieval plans", {"direct_retrieval", "recency_retrieval", "homogeneous_Fact",
                                                        "fact_retrieval"}),   # the fact plan answers from evidence
               ("Whole Math", {"homogeneous_Math"}), ("Whole General/Logic", {"homogeneous_General", "homogeneous_Logic"}),
               ("Decompose + fuse", {"decompose"}), ("Low-quality fallback", {"low_quality_fallback"})]


def plans_table(root, split="test"):
    """HR on the same questions, grouped by the plan COMPASS chose in run 1 (all datasets pooled)."""
    ours = {ds: per_question(root, ds, FINAL, False, split) for ds in DATASETS}
    refs = [("direct", True), ("self_consistency", True), ("targ", True)]
    base = {k: {ds: per_question(root, ds, k, b, split) for ds in DATASETS} for k, b in refs}
    if not all(ours.values()) or not all(all(v.values()) for v in base.values()):
        return "\\multicolumn{6}{l}{\\tbd} \\\\\n"
    lines = []
    for name, paths in PLAN_GROUPS:
        ids = [(ds, i) for ds in DATASETS for i, (h, p) in ours[ds].items() if p in paths]
        if not ids:
            continue
        vals = [st.mean(ours[ds][i][0] for ds, i in ids)] + [st.mean(base[k][ds][i][0] for ds, i in ids) for k, _ in refs]
        row = ["\\base " + name, f"{len(ids):,}".replace(",", "{,}")]
        row += [tint(v, v) for v in vals]
        lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n"


RETR_PLANS = {"direct_retrieval", "recency_retrieval", "homogeneous_Fact", "fact_retrieval"}


def retrieval_table(root):
    """% of test questions answered with retrieved evidence, per dataset (mean over runs):
    TARG/DTR (calibrated gate, trace.retrieved), Adaptive-RAG (classes B/C retrieve), COMPASS
    (retrieval plans; the low-quality fallback is not counted)."""
    def rate(key, baseline, ds):
        files = run_files(root, ds, key, baseline, "test")
        n = N_TEST[ds]
        if len(files) < 3 or any(score(f, n) is None for f in files):
            return None
        vals = []
        for f in files[:3]:
            recs = json.load(open(f))["records"]
            if key in ("targ", "dtr"):
                hit = [bool((r.get("trace") or {}).get("retrieved")) for r in recs]
            elif key == "adaptive_rag":
                hit = [(r.get("trace") or {}).get("class") in ("B", "C") for r in recs]
            else:
                hit = [r.get("path") in RETR_PLANS for r in recs]
            vals.append(100 * sum(hit) / len(hit))
        return st.mean(vals)
    lines = []
    for label, key, b in [("Adaptive-RAG", "adaptive_rag", True), ("TARG", "targ", True), ("DTR", "dtr", True),
                          ("\\method (retrieval plans)", FINAL, False)]:
        row = [("\\ours " if key == FINAL else "\\base ") + label]
        row += [f"{v:.1f}" if (v := rate(key, b, ds)) is not None else "\\tbd" for ds in DATASETS]
        lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n"


def cost_by_dataset(root):
    """Tokens/question (K) and LLM calls/question per method and dataset (mean over the three runs)."""
    def cell(key, baseline, ds):
        files = run_files(root, ds, key, baseline, "test")
        n = N_TEST[ds]
        if len(files) < 3 or any(score(f, n) is None for f in files):
            return "\\tbd"
        tok, calls = [], []
        for f in files[:3]:
            m = json.load(open(f)).get("metrics") or {}
            tok.append(m.get("avg_tokens_per_query") or 0)
            calls.append(((m.get("tokens") or {}).get("total_calls") or 0) / n)
        return f"{st.mean(tok) / 1000:.1f}K\\,/\\,{st.mean(calls):.0f}"
    lines = []
    for label, key, b in METHODS:
        lines.append(" & ".join([("\\ours " if key == FINAL else "\\base ") + label] + [cell(key, b, ds) for ds in DATASETS]) + " \\\\")
    return "\n".join(lines) + "\n"


# (parameter, [(value, run-name prefix of the dev runs; None = the default setting)])
SENS = [("$\\tau$", [("0.40", "tau0.40"), ("0.45", "tau0.45"), ("0.50", "tau0.50"), ("0.55", None), ("0.60", "tau0.60")]),
        ("$\\alpha$", [("0.2", "alpha0.2"), ("0.3", "alpha0.3"), ("0.4", None), ("0.5", "alpha0.5"), ("0.6", "alpha0.6")]),
        ("$k$", [("3", "k3"), ("5", None), ("7", "k7")])]


def sens_table(root):
    """Dev sensitivity (Qwen3-8B, 3 runs): HR per dataset, average, tokens/question; the default row
    (None) is the final model's dev runs and is set in bold."""
    def runs(ds, name):
        pat = os.path.join(root, f"compass_{ds}_dev_" + (f"{name}_" if name else "") + "rep[0-9].json")
        rs = [score(f, 100) for f in sorted(glob.glob(pat))]
        rs = [r for r in rs if r]
        return rs if len(rs) >= 3 else None
    lines = []
    for gi, (param, rows) in enumerate(SENS):
        if gi:
            lines.append("\\midrule")
        for val, name in rows:
            cells = [runs(ds, name) for ds in DATASETS]
            row = [param if val == rows[0][0] else "", val]
            row += [f"{st.mean(r['hr'] for r in c):.1f}" if c else "\\tbd" for c in cells]
            if all(cells):
                row.append(f"{st.mean(st.mean(r['hr'] for r in c) for c in cells):.1f}")
                tok = [st.mean(r["tok"] for r in c) for c in cells if all(r["tok"] for r in c)]
                row.append(f"{st.mean(tok) / 1000:.1f}K" if len(tok) == len(cells) else "\\tbd")
            else:
                row += ["\\tbd", "\\tbd"]
            if name is None:
                row = [row[0]] + [f"\\textbf{{{x}}}" for x in row[1:]]
            lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n"


def records_by_id(root, ds, key, baseline, split="test"):
    """id -> list of records over the three runs (finished cells only), else None."""
    n = N_TEST[ds] if split == "test" else 100
    files = run_files(root, ds, key, baseline, split)
    if len(files) < 3 or any(score(f, n) is None for f in files):
        return None
    out = {}
    for f in files[:3]:
        for r in json.load(open(f))["records"]:
            out.setdefault(r["id"], []).append(r)
    return out


GATE_BINS = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 1.01)]


def gate_bins_table(root):
    """Does the gate's complexity estimate track difficulty? Test questions binned by C(q) (run 1 of
    COMPASS, four benchmarks pooled): share, HR of Direct / TARG / COMPASS on the same questions."""
    ours = {ds: records_by_id(root, ds, FINAL, False) for ds in DATASETS}
    direct = {ds: per_question(root, ds, "direct", True) for ds in DATASETS}
    targ = {ds: per_question(root, ds, "targ", True) for ds in DATASETS}
    if not all(ours.values()) or not all(direct.values()) or not all(targ.values()):
        return "\\multicolumn{5}{l}{\\tbd} \\\\\n"
    rows = {b: [] for b in GATE_BINS}
    for ds in DATASETS:
        for i, recs in ours[ds].items():
            c = ((recs[0].get("trace") or {}).get("gate") or {}).get("complexity")
            if c is None:
                continue
            hr = 100 * st.mean(r["judgment"]["hallucination"] for r in recs)
            for b in GATE_BINS:
                if b[0] <= c < b[1]:
                    rows[b].append((hr, direct[ds][i][0], targ[ds][i][0]))
    tot = sum(len(v) for v in rows.values())
    lines = []
    for (lo, hi), v in rows.items():
        if not v:
            continue
        label = f"[{lo:.2f}, {min(hi, 1.0):.2f}{']' if hi > 1 else ')'}"
        lines.append(f"{label} & {100 * len(v) / tot:.1f} & {st.mean(x[1] for x in v):.1f} & "
                     f"{st.mean(x[2] for x in v):.1f} & {st.mean(x[0] for x in v):.1f} \\\\")
    return "\n".join(lines) + "\n"


def oracle_table(root):
    """Headroom of per-question selection: mean over questions of the lowest run-averaged HR among a
    set of methods (an upper bound for choosing among them per question), per dataset and average."""
    keys = [(k, b) for _, k, b in METHODS]
    pq = {k: {ds: per_question(root, ds, k, b) for ds in DATASETS} for k, b in keys}
    if not all(all(v.values()) for v in pq.values()):
        return "\\multicolumn{6}{l}{\\tbd} \\\\\n"
    sets = [("\\ours \\method", [FINAL]),
            ("\\base Best of Direct, TARG", ["direct", "targ"]),
            ("\\base Best of 6 baselines", [k for k, _ in keys if k != FINAL]),
            ("\\base Best of all 7 methods", [k for k, _ in keys])]
    lines = []
    for label, ks in sets:
        vals = []
        for ds in DATASETS:
            ids = set.intersection(*[set(pq[k][ds]) for k in ks])
            vals.append(st.mean(min(pq[k][ds][i][0] for k in ks) for i in ids))
        lines.append(" & ".join([label] + [tint(v, v) for v in vals + [st.mean(vals)]]) + " \\\\")
    return "\n".join(lines) + "\n"


def consistency_table(root):
    """How often each consistency operator runs and changes the answer (COMPASS, test, three runs
    pooled), and the HR of the answers it changed; one row per operator and benchmark it runs on."""
    ours = {ds: records_by_id(root, ds, FINAL, False) for ds in DATASETS}
    if not all(ours.values()):
        return "\\multicolumn{5}{l}{\\tbd} \\\\\n"
    names = {"truthfulqa": "TruthfulQA", "gsm8k": "GSM8K", "strategyqa": "StrategyQA", "freshqa": "FreshQA"}
    ops = [("\\trace numeric", "math"), ("\\trace yes/no", "reasoning_yesno"), ("\\cpc", "reasoning_cpc")]
    lines = []
    for label, op in ops:
        first = True
        for ds in DATASETS:
            recs = [r for v in ours[ds].values() for r in v]
            tc = [(r.get("trace") or {}).get("trace_consistency") or {} for r in recs]
            ran = [(t, r) for t, r in zip(tc, recs) if t.get("operator") == op]
            if len(ran) < 0.01 * len(recs):
                continue
            ch = [r for t, r in ran if t.get("fired")]
            hr = f"{100 * st.mean(r['judgment']['hallucination'] for r in ch):.1f}" if ch else "\\na"
            lines.append(f"{label if first else ''} & {names[ds]} & {100 * len(ran) / len(recs):.1f} & "
                         f"{100 * len(ch) / len(recs):.1f} & {hr} \\\\")
            first = False
    return "\n".join(lines) + "\n"


def nan(x, fmt="{:.2f}"):
    return fmt.format(x) if x is not None else "nan"


def export_figure_data(results, out):
    """Data files for the pgfplots figures (figures/data/*.dat); missing values are written as nan."""
    os.makedirs(out, exist_ok=True)
    root = os.path.join(results, "qwen3-8b")
    # plan shares per dataset (test)
    with open(os.path.join(out, "routes.dat"), "w") as f:
        f.write("ds " + " ".join(f"p{i}" for i in range(len(ROUTES))) + "\n")
        for ds, short in zip(DATASETS, ["TruthfulQA", "GSM8K", "StrategyQA", "FreshQA"]):
            runs = cell_runs(root, ds, FINAL, False, "test")
            if runs:
                tot = sum(sum(r["paths"].values()) for r in runs)
                vals = [100 * sum(r["paths"].get(p, 0) for r in runs for p in ps) / tot for _, ps in ROUTES]
            else:
                vals = [None] * len(ROUTES)
            f.write(short + " " + " ".join(nan(v) for v in vals) + "\n")
    # sensitivity (dev): one file per parameter
    for param, rows in SENS:
        name = {"$\\tau$": "tau", "$\\alpha$": "alpha", "$k$": "k"}[param]
        with open(os.path.join(out, f"sens_{name}.dat"), "w") as f:
            f.write("x hr tok default\n")
            for val, tag in rows:
                cells = []
                for ds in DATASETS:
                    pat = os.path.join(root, f"compass_{ds}_dev_" + (f"{tag}_" if tag else "") + "rep[0-9].json")
                    rs = [score(x, 100) for x in sorted(glob.glob(pat))]
                    rs = [r for r in rs if r]
                    cells.append(rs if len(rs) >= 3 else None)
                hr = st.mean(st.mean(r["hr"] for r in c) for c in cells) if all(cells) else None
                tk = st.mean(st.mean(r["tok"] for r in c) for c in cells) / 1000 if all(cells) else None
                f.write(f"{val} {nan(hr)} {nan(tk)} {1 if tag is None else 0}\n")
    # motivation (dev, Qwen3-8B): the same answer prompt with retrieval never / always (TARG's two branches)
    with open(os.path.join(out, "motivation_retrieval.dat"), "w") as f:
        f.write("x ds never always d delta\n")
        for j, (ds, short) in enumerate(zip(DATASETS, ["TruthfulQA", "GSM8K", "StrategyQA", "FreshQA"])):
            hr = {}
            for br in ("never", "always"):
                rs = [score(x, 100) for x in sorted(glob.glob(os.path.join(root, "baselines", f"targ_{br}_{ds}_dev_r[0-9].json")))]
                rs = [r for r in rs if r]
                hr[br] = st.mean(r["hr"] for r in rs) if len(rs) >= 3 else None
            d = hr["always"] - hr["never"] if None not in hr.values() else None
            dtxt = "nan" if d is None else ("+" if d >= 0 else "$-$") + f"{abs(d):.1f}"
            f.write(f"{j} {short} {nan(hr['never'], '{:.1f}')} {nan(hr['always'], '{:.1f}')} {nan(d, '{:.1f}')} {dtxt}\n")
    # complexity bins (test): share and HR of Direct / TARG / COMPASS
    ours = {ds: records_by_id(root, ds, FINAL, False) for ds in DATASETS}
    direct = {ds: per_question(root, ds, "direct", True) for ds in DATASETS}
    targ = {ds: per_question(root, ds, "targ", True) for ds in DATASETS}
    with open(os.path.join(out, "gatebins.dat"), "w") as f:
        f.write("bin label share direct targ compass\n")
        ok = all(ours.values()) and all(direct.values()) and all(targ.values())
        rows = {b: [] for b in GATE_BINS}
        if ok:
            for ds in DATASETS:
                for i, recs in ours[ds].items():
                    c = ((recs[0].get("trace") or {}).get("gate") or {}).get("complexity")
                    if c is None:
                        continue
                    hr = 100 * st.mean(r["judgment"]["hallucination"] for r in recs)
                    for b in GATE_BINS:
                        if b[0] <= c < b[1]:
                            rows[b].append((hr, direct[ds][i][0], targ[ds][i][0]))
        tot = sum(len(v) for v in rows.values()) or 1
        for j, ((lo, hi), v) in enumerate(rows.items()):
            label = f"{lo:.2f}--{min(hi, 1.0):.2f}"
            if v:
                f.write(f"{j} {label} {100 * len(v) / tot:.1f} {st.mean(x[1] for x in v):.1f} "
                        f"{st.mean(x[2] for x in v):.1f} {st.mean(x[0] for x in v):.1f}\n")
            else:
                f.write(f"{j} {label} nan nan nan nan\n")


BB_NAMES = {"qwen3-8b": "Qwen3-8B", "llama-3.1-8b": "Llama-3.1-8B-Instruct", "ministral-8b": "Ministral-8B-Instruct"}


def main_table_all(results):
    """One main table for the three backbones (test split): a group row per backbone, then the seven
    methods; HR tinted against Direct of the same backbone, best bold / second underlined within the
    backbone; average accuracy; tokens/question with an inline bar on one scale for the whole table."""
    blocks, tmax = [], 1.0
    for bb in BACKBONES:
        root = os.path.join(results, bb)
        data = {key: {ds: cell_runs(root, ds, key, b, "test") for ds in DATASETS} for _, key, b in METHODS}
        avg, toks = {}, {}
        for key, cells in data.items():
            if all(cells[ds] for ds in DATASETS):
                n = min(len(cells[ds]) for ds in DATASETS)
                avg[key] = [{"hr": st.mean(cells[ds][i]["hr"] for ds in DATASETS),
                             "acc": st.mean(cells[ds][i]["acc"] for ds in DATASETS)
                             if all(cells[ds][i]["acc"] is not None for ds in DATASETS) else None} for i in range(n)]
                if all(r["tok"] for ds in DATASETS for r in cells[ds]):
                    toks[key] = st.mean(st.mean(r["tok"] for r in cells[ds]) for ds in DATASETS)
            else:
                avg[key] = None
        tmax = max([tmax] + list(toks.values()))
        blocks.append((bb, data, avg, toks))
    lines = []
    for bb, data, avg, toks in blocks:
        lines.append(f"\\grp{{8}}{{{BB_NAMES[bb]}}} \\\\")
        cols = {ds: {k: data[k][ds] for k in data} for ds in DATASETS}
        cols["avg"] = avg
        means = {c: {k: st.mean(r["hr"] for r in runs) for k, runs in col.items() if runs} for c, col in cols.items()}
        marks = {c: ranks_of(means[c]) if all(col.values()) else {} for c, col in cols.items()}
        for label, key, _ in METHODS:
            row = [("\\ours " if key == FINAL else "\\base ") + label]
            for c in DATASETS + ["avg"]:
                runs = cols[c][key]
                ref = means[c].get("direct")
                if runs and ref is not None:
                    m, sd = mstd([r["hr"] for r in runs])
                    row.append(tint(m, ref, marks[c].get(key)) + f"\\sd{{{sd:.1f}}}")
                else:
                    row.append("\\tbd")
            acc = [r["acc"] for r in (avg[key] or []) if r["acc"] is not None]
            row.append(f"{st.mean(acc):.1f}" if acc else "\\tbd")
            row.append(f"{toks[key] / 1000:.1f}K\\,\\costbar{{{2.6 * toks[key] / tmax:.2f}em}}" if key in toks else "\\tbd")
            lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n"


def signif_all(results):
    """Paired-bootstrap table for the three backbones, one group row per backbone."""
    lines = []
    for bb in BACKBONES:
        lines.append(f"\\grp{{6}}{{{BB_NAMES[bb]}}} \\\\")
        lines.append(signif_table(os.path.join(results, bb)).rstrip("\n"))
    return "\n".join(lines) + "\n"


# Benchmark groups of the grouped main and significance tables, in column order:
# knowledge-intensive (TruthfulQA, FreshQA) and reasoning-intensive (GSM8K, StrategyQA).
GROUP_COLS = [("knowledge", ["truthfulqa", "freshqa"]), ("reasoning", ["gsm8k", "strategyqa"])]


def grouped_cols():
    """Column keys of the grouped tables: each group's datasets, then the group average; the macro average last."""
    cols = []
    for g, dss in GROUP_COLS:
        cols += dss + [g]
    return cols + ["avg"]


def main_table_grouped(results):
    """main_table_all with the benchmarks grouped: per-dataset HR, each group's average, and the macro
    average over all four, each computed per run and then mean±std over runs; best bold / second
    underlined per column within the backbone; MCS is the four-benchmark mean (no token column)."""
    lines = []
    for bb in BACKBONES:
        root = os.path.join(results, bb)
        data = {key: {ds: cell_runs(root, ds, key, b, "test") for ds in DATASETS} for _, key, b in METHODS}
        cols = {ds: {k: data[k][ds] for k in data} for ds in DATASETS}
        for g, dss in GROUP_COLS + [("avg", DATASETS)]:
            cols[g] = {}
            for key, cells in data.items():
                if all(cells[ds] for ds in dss):
                    n = min(len(cells[ds]) for ds in dss)
                    cols[g][key] = [{"hr": st.mean(cells[ds][i]["hr"] for ds in dss)} for i in range(n)]
                else:
                    cols[g][key] = None
        means = {c: {k: st.mean(r["hr"] for r in runs) for k, runs in col.items() if runs} for c, col in cols.items()}
        marks = {c: ranks_of(means[c]) if all(col.values()) else {} for c, col in cols.items()}
        mcs = {}
        for key, cells in data.items():
            if all(cells[ds] for ds in DATASETS):
                n = min(len(cells[ds]) for ds in DATASETS)
                acc = [st.mean(cells[ds][i]["acc"] for ds in DATASETS) for i in range(n)
                       if all(cells[ds][i]["acc"] is not None for ds in DATASETS)]
                mcs[key] = st.mean(acc) if acc else None
        lines.append(f"\\grp{{9}}{{{BB_NAMES[bb]}}} \\\\")
        for label, key, _ in METHODS:
            row = [("\\ours " if key == FINAL else "\\base ") + label]
            for c in grouped_cols():
                runs = cols[c][key]
                ref = means[c].get("direct")
                if runs and ref is not None:
                    m, sd = mstd([r["hr"] for r in runs])
                    row.append(tint(m, ref, marks[c].get(key)) + f"\\sd{{{sd:.1f}}}")
                else:
                    row.append("\\tbd")
            if mcs.get(key) is not None and mcs.get("direct") is not None:
                row.append(tint(mcs[key], mcs["direct"], lower_better=False))
            else:
                row.append("\\tbd")
            lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n"


def ci_cell(point, boots, n_boot):
    """Difference with its 95% bootstrap interval; bold if the interval excludes zero (as in signif_table)."""
    boots = sorted(boots)
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    fmt = "[\\mnumbf]" if (hi < 0 or lo > 0) else "[\\mnum]"
    return "\\hcl" + fmt + f"{{{point:.1f}}}{{0}}" + f"\\ci{{{sgn(lo)}}}{{{sgn(hi)}}}"


def signif_grouped_table(root, n_boot=10000, seed=0, split="test"):
    """signif_table with the grouped columns. The dataset and macro-average columns repeat signif_table's
    draws exactly (same generator, same order), so their values are identical; each group average
    resamples within its two datasets with a second generator (seed + 1)."""
    import random
    rng, rng_g = random.Random(seed), random.Random(seed + 1)
    ours = {ds: per_question(root, ds, FINAL, False, split) for ds in DATASETS}
    lines = []
    for label, key, b in METHODS:
        if key == FINAL:
            continue
        base = {ds: per_question(root, ds, key, b, split) for ds in DATASETS}
        diffs = {}
        for ds in DATASETS:
            if ours[ds] and base[ds]:
                ids = sorted(set(ours[ds]) & set(base[ds]))
                diffs[ds] = [ours[ds][i][0] - base[ds][i][0] for i in ids]
        cells = {}
        for c in DATASETS + ["avg"]:                       # signif_table's order and draws
            if c == "avg":
                if len(diffs) < len(DATASETS):
                    continue
                point = st.mean(st.mean(d) for d in diffs.values())
                boots = [st.mean(st.mean(rng.choices(d, k=len(d))) for d in diffs.values()) for _ in range(n_boot)]
            elif c in diffs:
                d = diffs[c]
                point = st.mean(d)
                boots = [st.mean(rng.choices(d, k=len(d))) for _ in range(n_boot)]
            else:
                continue
            cells[c] = ci_cell(point, boots, n_boot)
        for g, dss in GROUP_COLS:
            if all(ds in diffs for ds in dss):
                point = st.mean(st.mean(diffs[ds]) for ds in dss)
                boots = [st.mean(st.mean(rng_g.choices(diffs[ds], k=len(diffs[ds]))) for ds in dss)
                         for _ in range(n_boot)]
                cells[g] = ci_cell(point, boots, n_boot)
        row = ["\\base " + label] + [cells.get(c, "\\tbd") for c in grouped_cols()]
        lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n"


def signif_grouped(results):
    """Grouped paired-bootstrap table for the three backbones, one group row per backbone."""
    lines = []
    for bb in BACKBONES:
        lines.append(f"\\grp{{8}}{{{BB_NAMES[bb]}}} \\\\")
        lines.append(signif_grouped_table(os.path.join(results, bb)).rstrip("\n"))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--only", default="", help="comma-separated table names (e.g. main_grouped_test.tex); default: all")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    only = set(filter(None, args.only.split(",")))

    def write(name, text):
        if name.startswith("signif"):
            text = tint_fragment(text, center="zero", set_marks=False)   # bold = interval excludes zero
        elif name.startswith("plans"):
            text = tint_fragment(text, scope="row")
        elif name.split("_")[0] in ("main", "ablation", "oracle"):
            text = tint_fragment(text)
        open(os.path.join(args.out, name), "w").write(text)
        print(f"{name}: {text.count(chr(92) + 'tbd')} cells pending")
    if only:   # rebuild only the named tables
        jobs = {"main_grouped_test.tex": lambda: main_table_grouped(args.results),
                "signif_grouped_test.tex": lambda: signif_grouped(args.results)}
        for bb in BACKBONES:
            root = os.path.join(args.results, bb)
            for name, fn in (("plans", plans_table), ("oracle", oracle_table), ("retrieval", retrieval_table),
                             ("consistency", consistency_table)):
                jobs[f"{name}_{bb}_test.tex"] = (lambda fn=fn, root=root: fn(root))
        jobs["ablation_qwen3-8b_test.tex"] = lambda: ablation_table(os.path.join(args.results, "qwen3-8b"))
        for name in sorted(only):
            write(name, jobs[name]())
        return
    for bb in BACKBONES:
        root = os.path.join(args.results, bb)
        text, data, avg = main_table(root, "test")
        write(f"main_{bb}_test.tex", text)
        write(f"cost_{bb}_test.tex", cost_table(data, avg))
        write(f"routes_{bb}_test.tex", routes_table(root))
        write(f"signif_{bb}_test.tex", signif_table(root))
        write(f"plans_{bb}_test.tex", plans_table(root))
        write(f"retrieval_{bb}_test.tex", retrieval_table(root))
        write(f"costds_{bb}_test.tex", cost_by_dataset(root))
        write(f"gatebins_{bb}_test.tex", gate_bins_table(root))
        write(f"oracle_{bb}_test.tex", oracle_table(root))
        write(f"consistency_{bb}_test.tex", consistency_table(root))
    write("main_all_test.tex", main_table_all(args.results))
    write("signif_all_test.tex", signif_all(args.results))
    write("main_grouped_test.tex", main_table_grouped(args.results))
    write("signif_grouped_test.tex", signif_grouped(args.results))
    write("sens_qwen3-8b_dev.tex", sens_table(os.path.join(args.results, "qwen3-8b")))
    export_figure_data(args.results, os.path.join(args.out, "data"))
    print("figure data written to", os.path.join(args.out, "data"))
    write("ablation_qwen3-8b_test.tex", ablation_table(os.path.join(args.results, "qwen3-8b")))


if __name__ == "__main__":
    main()
