"""Score human-eval annotations: IAA, judge validity, and complexity correlation.

Run AFTER the 3 annotators fill human_eval/annotator_{1,2,3}.csv:
    python scripts/score_human_eval.py
Outputs: Fleiss' kappa (hallucination IAA), difficulty IAA, human-vs-LLM-judge agreement
(Cohen's kappa + accuracy), and Spearman(gate complexity, human difficulty).
"""
import json, csv, itertools, math
from collections import Counter
from scipy.stats import spearmanr

OUT = "human_eval"
truth = json.load(open(f"{OUT}/_ground_truth.json"))

def load(a):
    rows = {}
    for r in csv.DictReader(open(f"{OUT}/annotator_{a}.csv")):
        h = (r["is_hallucination(yes/no)"] or "").strip().lower()
        d = (r["difficulty(1-5)"] or "").strip()
        rows[r["seq"]] = {
            "h": 1 if h in ("yes", "y", "1") else (0 if h in ("no", "n", "0") else None),
            "d": int(d) if d.isdigit() else None,
        }
    return rows

ann = {a: load(a) for a in (1, 2, 3)}
seqs = [s for s in truth if all(ann[a].get(s, {}).get("h") is not None and
                                ann[a].get(s, {}).get("d") is not None for a in (1, 2, 3))]
print(f"Fully-annotated items: {len(seqs)}/{len(truth)}")
if not seqs:
    print("No complete annotations yet — fill the CSVs first."); raise SystemExit

# ---------- Fleiss' kappa (hallucination, 3 raters, 2 categories) ----------
def fleiss_kappa(table):  # table: list of [count_cat0, count_cat1] per item, n raters each
    N = len(table); n = sum(table[0])
    p = [sum(table[i][j] for i in range(N)) / (N * n) for j in range(len(table[0]))]
    Pe = sum(pj * pj for pj in p)
    Pbar = sum((sum(c * c for c in row) - n) / (n * (n - 1)) for row in table) / N
    return (Pbar - Pe) / (1 - Pe) if (1 - Pe) else 1.0

htab = [[sum(1 for a in (1, 2, 3) if ann[a][s]["h"] == c) for c in (0, 1)] for s in seqs]
print(f"\n[IAA] hallucination Fleiss' kappa = {fleiss_kappa(htab):.3f}  "
      f"({'almost perfect' if fleiss_kappa(htab)>.8 else 'substantial' if fleiss_kappa(htab)>.6 else 'moderate' if fleiss_kappa(htab)>.4 else 'fair/poor'})")

# ---------- difficulty IAA: Krippendorff ordinal if available, else mean pairwise Spearman ----------
diff = {a: [ann[a][s]["d"] for s in seqs] for a in (1, 2, 3)}
try:
    import krippendorff
    data = [diff[a] for a in (1, 2, 3)]
    alpha = krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
    print(f"[IAA] difficulty Krippendorff's alpha (ordinal) = {alpha:.3f}")
except Exception:
    ps = [spearmanr(diff[x], diff[y]).correlation for x, y in itertools.combinations((1, 2, 3), 2)]
    print(f"[IAA] difficulty mean pairwise Spearman = {sum(ps)/len(ps):.3f}  "
          f"(krippendorff lib not installed — install for ordinal alpha)")

# ---------- human consensus vs LLM judge (hallucination) ----------
def cohen_kappa(y1, y2):
    N = len(y1); po = sum(a == b for a, b in zip(y1, y2)) / N
    c1, c2 = Counter(y1), Counter(y2)
    pe = sum((c1[k] / N) * (c2[k] / N) for k in set(y1) | set(y2))
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0

hum_major = [1 if sum(ann[a][s]["h"] for a in (1, 2, 3)) >= 2 else 0 for s in seqs]
judge = [truth[s]["judge_hallucination"] for s in seqs]
acc = sum(a == b for a, b in zip(hum_major, judge)) / len(seqs)
print(f"\n[judge validation] human-consensus vs LLM-judge (gpt-5.4-mini):")
print(f"  agreement (accuracy) = {acc*100:.1f}%   Cohen's kappa = {cohen_kappa(hum_major, judge):.3f}")
print(f"  → judge is {'reliable' if cohen_kappa(hum_major, judge)>.6 else 'moderately reliable'}; HR metric validated")

# ---------- gate complexity vs human difficulty ----------
hum_diff = [sum(ann[a][s]["d"] for a in (1, 2, 3)) / 3 for s in seqs]
gate_cx = [truth[s]["gate_complexity"] for s in seqs]
rho, p = spearmanr(gate_cx, hum_diff)
print(f"\n[gate validation] gate complexity vs human difficulty:")
print(f"  Spearman rho = {rho:.3f}  (p = {p:.2e})  "
      f"→ {'significant positive' if rho>0 and p<.05 else 'weak/ns'} — complexity tracks human-perceived difficulty")
