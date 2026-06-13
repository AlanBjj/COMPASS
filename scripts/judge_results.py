"""Decoupled post-hoc judge for the full-test sweep.

The sweep generates answers with --no-judge (GPUs flat-out, never blocked on the slow csun
relay). This script then judges every results/*_rep*.json: rate-limited concurrency + timeout
+ retry so the flaky relay can't stall everything, and it's RESUMABLE (already-judged records
are skipped) so it can be re-run or switched to a faster endpoint mid-way.

Usage:
  python scripts/judge_results.py                 # judge with the configured judge (.env OPENAI_*)
  python scripts/judge_results.py --openrouter    # judge with OpenRouter direct-to-OpenAI (fast, pricey)
  JUDGE_CONCURRENCY=8 python scripts/judge_results.py   # tune concurrency (default 8)
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from compass.datasets.base import Example
from compass.envload import load_env
from compass.eval import aggregate
from compass.eval.judge import Judgment, judge_one
from compass.llm.client import LLMClient


async def _judge_one_safe(ex, answer, client, sem, retries=4, timeout=75):
    """One judgment with bounded concurrency, per-call timeout, and retry on timeout/error."""
    async with sem:
        for k in range(retries):
            try:
                j = await asyncio.wait_for(judge_one(ex, answer, client), timeout=timeout)
                if j.valid:
                    return j
            except Exception:
                pass
            await asyncio.sleep(2 * (k + 1))  # back off before retry
        return Judgment(-1, -1, "judge failed after retries")  # excluded from metrics


async def judge_file(path, client, sem):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    recs = d["records"]
    ds = d["meta"]["dataset"]
    todo = [r for r in recs if r.get("judgment", {}).get("hallucination", -1) < 0]
    if not todo:
        print(f"[skip] {os.path.basename(path)} already judged ({len(recs)} recs)")
        return

    async def do(rec):
        ex = Example(id=rec["id"], question=rec["question"], gold=rec["gold"], dataset=ds)
        j = await _judge_one_safe(ex, rec["answer"], client, sem)
        rec["judgment"] = j.as_dict()

    print(f"[judge] {os.path.basename(path)}: {len(todo)}/{len(recs)} to judge ...")
    await asyncio.gather(*[do(r) for r in todo])

    judgments = [Judgment(r["judgment"]["hallucination"], r["judgment"]["accuracy"], "") for r in recs]
    m = aggregate(judgments, n_total=len(recs),
                  token_snapshot=client.ledger.snapshot(), latencies=[0.0] * len(recs))
    d["metrics"] = {**d.get("metrics", {}), **m}  # keep method-token fields, refresh HR/Acc
    d["meta"]["judge_model"] = client.model
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, default=str)
    n_fail = sum(1 for r in recs if r["judgment"]["hallucination"] < 0)
    print(f"[done] {ds}: HR={m['hallucination_rate']} Acc={m['accuracy']} "
          f"n={len(recs)} (judge-fail={n_fail})")


async def main():
    load_env()
    if "--openrouter" in sys.argv:
        base = os.environ["OPENROUTER_BASE_URL"]
        key = os.environ["OPENROUTER_API_KEY"]
        model = os.environ.get("OPENROUTER_JUDGE_MODEL", "gpt-5.4-mini")
        print(f"[judge endpoint] OpenRouter {model}")
    else:
        base = os.environ["OPENAI_BASE_URL"]
        key = os.environ["OPENAI_API_KEY"]
        model = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5.4-mini")  # zovelox; validated stable+accurate
        print(f"[judge endpoint] {base} {model}")
    conc = int(os.environ.get("JUDGE_CONCURRENCY", "8"))
    client = LLMClient(base_url=base, model=model, api_key=key,
                       default_temperature=0.0, default_max_tokens=128)
    sem = asyncio.Semaphore(conc)

    files = sorted(f for f in glob.glob("results/*_rep*.json") if "partial" not in f)
    if not files:
        print("no results/*_rep*.json yet (let the generation sweep finish first)")
        return
    for f in files:
        await judge_file(f, client, sem)
    print("=== all files judged ===")


if __name__ == "__main__":
    asyncio.run(main())
