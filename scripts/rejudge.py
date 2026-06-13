"""Re-judge already-generated ablation records (HR + Accuracy) without re-running the pipeline.

The ablation generations (results/*abl*.json) carry per-record question/answer/gold and the
token cost, but were written with the judge skipped (HR=None). This script reloads those
records, runs ONLY the judge (gpt-5.4-mini, same as the main table) over the stored answers,
and writes a sibling *_judged.json with hallucination_rate / accuracy filled in and the token
cost preserved. It never overwrites the original generation file.

Usage:
  python -m scripts.rejudge results/truthfulqa_truthfulqa_abl_nocpc.json --limit 5   # smoke
  python -m scripts.rejudge results/*abl2_d2.json results/*abl2_d3.json              # batch
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
from statistics import mean

from compass.envload import load_env
from compass.datasets.base import Example
from compass.eval import judge_all
from compass.llm.client import LLMClient


async def _judge_file(path: str, client: LLMClient, limit: int | None, batch: int) -> dict:
    d = json.load(open(path, "r", encoding="utf-8"))
    ds = d.get("meta", {}).get("dataset", "")
    recs = d.get("records", [])
    if limit:
        recs = recs[:limit]
    examples = [Example(id=r["id"], question=r["question"], gold=r.get("gold", {}) or {}, dataset=ds)
                for r in recs]
    answers = [r.get("answer", "") for r in recs]

    judgments = []
    for s in range(0, len(examples), batch):
        judgments.extend(await judge_all(examples[s:s + batch], answers[s:s + batch], client))

    hrs = [j.hallucination for j in judgments if j.hallucination in (0, 1)]
    accs = [j.accuracy for j in judgments if j.accuracy is not None and j.accuracy >= 0]
    n_bad = sum(1 for j in judgments if j.hallucination < 0)
    hr = round(100 * mean(hrs), 1) if hrs else None
    acc = round(mean(accs), 1) if accs else None

    for r, j in zip(recs, judgments):
        r["judgment"] = j.as_dict()
    metrics = dict(d.get("metrics", {}) or {})
    metrics["hallucination_rate"] = hr
    metrics["accuracy"] = acc
    metrics["n_judged"] = len(judgments)
    metrics["n_unparsable"] = n_bad
    out = {"meta": {**d.get("meta", {}), "judge_model": client.model, "rejudged": True},
           "metrics": metrics, "records": recs}
    out_path = path.replace(".json", "_judged.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    tok = (d.get("metrics", {}) or {}).get("avg_tokens_per_query")
    print(f"  {os.path.basename(path):42s} n={len(judgments):4d} HR={hr} Acc={acc} "
          f"tok={tok} bad={n_bad} -> {os.path.basename(out_path)}")
    return metrics


async def _main(paths, limit, batch):
    load_env()
    client = LLMClient(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model=os.environ.get("OPENAI_JUDGE_MODEL") or "gpt-5.4-mini",
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        default_temperature=0.0,
        default_max_tokens=512,
    )
    print(f"[rejudge] judge={client.model} base={os.environ.get('OPENAI_BASE_URL')} files={len(paths)}")
    for p in paths:
        await _judge_file(p, client, limit, batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="result json files (globs ok)")
    ap.add_argument("--limit", type=int, default=None, help="only judge first N records (smoke)")
    ap.add_argument("--batch", type=int, default=50, help="judge gather batch size")
    args = ap.parse_args()
    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) or ([p] if os.path.exists(p) else []))
    asyncio.run(_main(files, args.limit, args.batch))


if __name__ == "__main__":
    main()
