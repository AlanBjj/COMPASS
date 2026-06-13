"""Generate human-eval annotation material for judge validity and complexity IAA/correlation.

Stratified sample of test items (per dataset, across complexity tertiles, balanced on the LLM
judge's hallucination label), emitted as 3 identical blind CSVs (one per annotator). The judge
label / gate complexity / routing path are withheld into a private ground-truth file and joined
back only after annotation, so annotators are blind. Run: python scripts/make_human_eval.py
"""
import json, csv, random, os

random.seed(0)
PER_DS = 40
OUT = "human_eval"
DS = ["truthfulqa", "gsm8k", "strategyqa", "freshqa"]
FILES = {
    "truthfulqa": "results/truthfulqa_truthfulqa_testfull5.json",
    "gsm8k": "results/gsm8k_gsm8k_testfull5.json",
    "strategyqa": "results/strategyqa_strategyqa_testfull5.json",
    "freshqa": "results/freshqa_freshqa_testfull5b.json",
}

def reference(ds, gold):
    """Human-readable correct answer for the annotator to judge against."""
    if ds == "gsm8k":
        return f"Correct final answer: {gold.get('final','')}"
    if ds == "truthfulqa":
        return f"Best answer: {gold.get('best','')}  |  Also correct: {gold.get('correct','')}"
    if ds == "strategyqa":
        return f"Correct (yes/no): {gold.get('answer','')}"
    if ds == "freshqa":
        a = gold.get("answer") or "; ".join(gold.get("answers", []) or [])
        return f"Correct answer: {a}"
    return str(gold)

def complexity(r):
    return r.get("trace", {}).get("gate", {}).get("complexity")

# --- stratified sample: per dataset, 3 complexity tertiles, balance on judge hallucination ---
items, truth = [], {}
for ds in DS:
    recs = [r for r in json.load(open(FILES[ds]))["records"] if complexity(r) is not None]
    recs.sort(key=complexity)
    n = len(recs)
    tertiles = [recs[: n // 3], recs[n // 3 : 2 * n // 3], recs[2 * n // 3 :]]
    picked, per_t = [], PER_DS // 3
    for t in tertiles:
        hal = [r for r in t if r["judgment"]["hallucination"] == 1]
        ok = [r for r in t if r["judgment"]["hallucination"] == 0]
        random.shuffle(hal); random.shuffle(ok)
        half = per_t // 2
        take = hal[:half] + ok[: per_t - len(hal[:half])]
        if len(take) < per_t:  # top up from whichever remains
            rest = [r for r in t if r not in take]
            random.shuffle(rest); take += rest[: per_t - len(take)]
        picked += take[:per_t]
    # top up dataset to exactly PER_DS from leftovers
    if len(picked) < PER_DS:
        rest = [r for r in recs if r not in picked]; random.shuffle(rest)
        picked += rest[: PER_DS - len(picked)]
    for r in picked[:PER_DS]:
        items.append({"_ds": ds, "_r": r})

random.shuffle(items)  # mix datasets so annotators can't infer blocks

os.makedirs(OUT, exist_ok=True)
rows = []
for i, it in enumerate(items, 1):
    ds, r = it["_ds"], it["_r"]
    seq = f"H{i:03d}"
    rows.append({
        "seq": seq,
        "dataset": ds,
        "question": r["question"],
        "model_answer": r["answer"],
        "reference_answer": reference(ds, r["gold"]),
        "is_hallucination(yes/no)": "",
        "difficulty(1-5)": "",
    })
    truth[seq] = {
        "dataset": ds,
        "id": r["id"],
        "judge_hallucination": r["judgment"]["hallucination"],
        "judge_accuracy": r["judgment"]["accuracy"],
        "gate_complexity": round(complexity(r), 4),
        "path": r.get("path"),
    }

cols = ["seq", "dataset", "question", "model_answer", "reference_answer",
        "is_hallucination(yes/no)", "difficulty(1-5)"]
for a in (1, 2, 3):
    with open(f"{OUT}/annotator_{a}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
with open(f"{OUT}/_ground_truth.json", "w") as f:
    json.dump(truth, f, indent=2)

# distribution report
from collections import Counter
print(f"Sampled {len(rows)} items → {OUT}/annotator_{{1,2,3}}.csv + _ground_truth.json (hidden)")
print("  per dataset:", dict(Counter(t["dataset"] for t in truth.values())))
print("  judge hallucination balance:", dict(Counter(t["judge_hallucination"] for t in truth.values())))
cx = sorted(t["gate_complexity"] for t in truth.values())
print(f"  complexity span: {cx[0]:.2f} – {cx[-1]:.2f} (median {cx[len(cx)//2]:.2f})")
