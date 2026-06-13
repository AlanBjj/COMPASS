"""Phase B of the two-phase split: judge already-GENERATED baseline answers with the LLM judge.

Generation (phase A, `python -m compass.baselines.run --gen-only`) runs on the GPU and writes
`<stem>.gen.json` artifacts (answers + traces + gold, no judgments). This module reads those and
runs ONLY the judge — so the judge relay never idles the GPUs, can run with high
concurrency, and can be resumed/re-run independently of generation. It reuses the IDENTICAL judge
(prompts/judge.txt) and metric aggregation as compass/run.py, and PRESERVES the method-token
snapshot recorded at generation time (the judge phase only adds judgments + judge_tokens).

Usage:
  python -m compass.baselines.judge_runs results/baselines               # all *.gen.json in a dir
  python -m compass.baselines.judge_runs results/baselines/react_gsm8k_test_r0.gen.json
  python -m compass.baselines.judge_runs <path> --concurrency 24 --judge-model gpt-5.4-mini
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
from typing import List, Optional

from ..datasets.base import Example
from ..envload import load_env
from ..eval import aggregate, judge_all
from ..eval.judge import Judgment
from ..llm.client import LLMClient


def _gen_files(path: str) -> List[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.gen.json")))
    return [path]


async def judge_file(
    gen_path: str,
    *,
    judge_model: Optional[str] = None,
    concurrency: int = 16,
    batch_size: int = 200,
    force: bool = False,
) -> Optional[dict]:
    with open(gen_path, "r", encoding="utf-8") as f:
        gen = json.load(f)
    meta, gen_metrics, recs = gen["meta"], gen.get("metrics", {}), gen["records"]

    stem = gen_path[: -len(".gen.json")] if gen_path.endswith(".gen.json") else os.path.splitext(gen_path)[0]
    out_path = f"{stem}.json"
    partial_path = f"{stem}.judge.partial.json"
    if os.path.exists(out_path) and not force:
        print(f"[judge] {os.path.basename(out_path)} exists; skip (use --force to re-judge)")
        return None

    dataset = meta["dataset"]
    examples = [Example(id=r["id"], question=r["question"], gold=r.get("gold", {}), dataset=dataset)
                for r in recs]
    answers = [r.get("answer", "") for r in recs]
    latencies = [r.get("latency", 0.0) for r in recs]

    # Resume: reload judged records from a prior interrupted judge pass.
    judged: dict = {}
    if os.path.exists(partial_path):
        with open(partial_path, "r", encoding="utf-8") as f:
            for r in json.load(f).get("records", []):
                judged[r["id"]] = r["judgment"]
        print(f"[judge] resuming {os.path.basename(stem)}: {len(judged)}/{len(recs)} already judged")

    # Judge = the SAME model both tracks use: zovelox gpt-5.4-mini (main thread default). judge_one
    # passes max_tokens=512 explicitly (gpt-5.x emit reasoning before the JSON), but set the client
    # default to 512 too for consistency.
    judge_client = LLMClient(
        base_url=os.environ.get("OPENAI_BASE_URL") or "EMPTY",
        model=judge_model or meta.get("judge_model", "gpt-5.4-mini"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        default_temperature=0.0, default_max_tokens=512,
        concurrency=concurrency,
        # The zovelox relay 429s with model_cooldown under bursts; be patient (6 tries, longer
        # backoff: ~4+8+12+16+20s) so a brief upstream cooldown is ridden out, not fatal.
        retries=6, retry_backoff=4.0,
    )

    # Empty / errored answers are NEVER sent to the judge — an empty string is not a valid answer
    # and the judge would sometimes reward it (audit BLOCKER-2, esp. TruthfulQA "I have no comment").
    # Force them to hallucination=1 / accuracy=0 (the honest rule). After the Serper fix these
    # should be rare, but the rule guarantees a blank can never count as correct.
    n_forced = 0
    for i, r in enumerate(recs):
        if r["id"] in judged:
            continue
        if not (answers[i] or "").strip():
            judged[r["id"]] = {"hallucination": 1, "accuracy": 0}
            n_forced += 1

    todo = [i for i, r in enumerate(recs) if r["id"] not in judged]
    print(f"[judge] {os.path.basename(gen_path)} | {dataset}/{meta.get('method')} | "
          f"{len(todo)} to judge, {n_forced} empty->H1/A0 (concurrency={concurrency})")
    for s in range(0, len(todo), batch_size):
        idx = todo[s:s + batch_size]
        js = await judge_all([examples[i] for i in idx], [answers[i] for i in idx], judge_client)
        for i, j in zip(idx, js):
            judged[recs[i]["id"]] = j.as_dict()
        # checkpoint
        part_recs = [{"id": r["id"], "judgment": judged[r["id"]]} for r in recs if r["id"] in judged]
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump({"records": part_recs}, f, ensure_ascii=False)
        done = sum(1 for r in recs if r["id"] in judged)
        print(f"[judge] {os.path.basename(stem)}: {done}/{len(recs)} judged")

    # Assemble the final judged file (same schema as a one-shot judged run).
    judgments = [Judgment(judged[r["id"]]["hallucination"], judged[r["id"]]["accuracy"], "") for r in recs]
    metrics = aggregate(judgments, n_total=len(recs),
                        token_snapshot=gen_metrics.get("tokens"), latencies=latencies)
    metrics["judge_tokens"] = judge_client.ledger.snapshot()

    out_meta = {k: v for k, v in meta.items() if k != "status"}
    out_meta["judge_model"] = judge_model or meta.get("judge_model", "gpt-5.4-mini")
    final_records = [{k: v for k, v in r.items() if k != "latency"} | {"judgment": judged[r["id"]]}
                     for r in recs]
    result = {"meta": out_meta, "metrics": metrics, "records": final_records}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"[judge] DONE {os.path.basename(out_path)} | HR={metrics['hallucination_rate']} "
          f"Acc={metrics['accuracy']} n={metrics['n_scored']}")
    return result


async def main_async(args) -> None:
    load_env()
    files = _gen_files(args.path)
    if not files:
        print(f"no .gen.json files at {args.path}")
        return
    # Judge files one at a time (each already saturates the judge with `concurrency` requests).
    # Per-file resilience: a transient relay failure (e.g. model_cooldown) on one file must NOT
    # abort the whole sweep. Failed files are skipped (no final .json), so simply re-running this
    # command retries only them (file-level resume). batch checkpoints make a file resumable too.
    failed = []
    for gp in files:
        try:
            await judge_file(gp, judge_model=args.judge_model, concurrency=args.concurrency,
                             batch_size=args.batch_size, force=args.force)
        except Exception as err:  # noqa: BLE001 - keep judging the other files
            print(f"[judge] FAILED {os.path.basename(gp)}: {err} (will be retried on re-run)")
            failed.append(os.path.basename(gp))
    if failed:
        print(f"[judge] {len(failed)} file(s) failed (re-run to retry): {failed}")
    else:
        print("[judge] all files judged.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge generated baseline answers (phase B)")
    ap.add_argument("path", help="a .gen.json file or a directory of them")
    ap.add_argument("--judge-model", type=str, default=None)
    ap.add_argument("--concurrency", type=int, default=6,
                    help="parallel judge requests (keep low: zovelox 429s/cooldowns under bursts)")
    ap.add_argument("--batch-size", type=int, default=50, help="judge checkpoint interval (finer = better resume)")
    ap.add_argument("--force", action="store_true", help="re-judge even if the final .json exists")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
