"""STEP 3: quantify judge variance. Re-judge the SAME (question, answer, gold) triples
from a stability run file N times with a fresh judge client, and report per-run HR/Acc plus
how many of the per-example hallucination labels flip across re-judges. Temporary diagnostic."""

import asyncio
import json
import os
import sys

from compass.datasets.base import Example
from compass.envload import load_env
from compass.eval import judge_all
from compass.llm.client import LLMClient

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/_stab_gsm8k_A.json"
N_REJUDGE = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def hr_acc(judgments):
    valid = [j for j in judgments if j.valid]
    n = len(valid)
    hr = round(100.0 * sum(j.hallucination for j in valid) / n, 2) if n else None
    acc = round(sum(j.accuracy for j in valid) / n, 2) if n else None
    return hr, acc, n


async def main():
    load_env()
    data = json.load(open(PATH))
    ds_name = data["meta"]["dataset"]
    recs = data["records"]
    examples = [Example(id=r["id"], question=r["question"], gold=r["gold"], dataset=ds_name)
                for r in recs]
    answers = [r["answer"] for r in recs]

    jm = {"base_url": os.environ.get("OPENAI_BASE_URL"), "model": "gpt-5.4-mini"}
    judge_client = LLMClient(
        base_url=os.environ["OPENAI_BASE_URL"],
        model="gpt-5.4-mini",
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        default_temperature=0.0,
        default_max_tokens=128,
        concurrency=8,
    )

    runs = []
    for k in range(N_REJUDGE):
        judgments = await judge_all(examples, answers, judge_client)
        hr, acc, n = hr_acc(judgments)
        # per-example hallucination label (None where invalid)
        labels = [j.hallucination if j.valid else None for j in judgments]
        runs.append({"hr": hr, "acc": acc, "n_valid": n, "labels": labels})
        print(f"[rejudge {k+1}/{N_REJUDGE}] HR={hr} Acc={acc} n_valid={n}")

    # flip analysis: for each example, was the hallucination label identical across all runs?
    flips = 0
    flipped_ids = []
    any_invalid = 0
    for i in range(len(examples)):
        vals = [r["labels"][i] for r in runs]
        if any(v is None for v in vals):
            any_invalid += 1
            # treat invalid as part of disagreement only if mixed with valid differing values
        valset = set(v for v in vals if v is not None)
        if len(valset) > 1:  # label changed between at least two re-judges
            flips += 1
            flipped_ids.append(examples[i].id)

    print("\n=== JUDGE VARIANCE SUMMARY ===")
    print(f"file={PATH} dataset={ds_name} n_examples={len(examples)} n_rejudge={N_REJUDGE}")
    for k, r in enumerate(runs):
        print(f"  re-judge {k+1}: HR={r['hr']}  Acc={r['acc']}  n_valid={r['n_valid']}")
    print(f"hallucination labels that FLIPPED across the {N_REJUDGE} re-judges: "
          f"{flips}/{len(examples)}")
    print(f"examples with >=1 invalid judge output in some run: {any_invalid}")
    print(f"flipped ids: {flipped_ids}")


if __name__ == "__main__":
    asyncio.run(main())
