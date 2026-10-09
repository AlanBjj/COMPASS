"""Generic baseline run entry: `python -m compass.baselines.run --config <cfg> --method <name>`.

Deliberately mirrors compass/run.py so baselines share the IDENTICAL evaluation protocol:
same dataset loaders + dev/test split, same LLM judge, same metric aggregation, same
per-role token ledger, AND the same checkpoint/resume mechanism. The ONLY differences from
compass/run.py: it dispatches to a baseline method (instead of the COMPASS pipeline) and records a
`trace` instead of a COMPASS `path`.

One config holds the shared run settings + a `methods:` block with each method's hyperparameters;
--method picks which to run. --run-index only tags the output file (for the mean+/-std over 3
runs); the dev/test split is the SAME across runs (split seed fixed in config) so the only
variation is vLLM's generation nondeterminism.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import List, Optional

import yaml

from ..datasets import load_dataset
from ..envload import load_env
from ..eval import aggregate, judge_all
from ..eval.judge import Judgment
from ..llm.client import LLMClient, TokenLedger, date_note_text
from .registry import build_baseline
from .retrieval import build_retriever


async def run(
    config_path: str,
    method: str,
    *,
    no_judge: bool = False,
    sample_size: Optional[int] = None,
    judge_model: Optional[str] = None,
    split: Optional[str] = None,
    dev_size: Optional[int] = None,
    run_index: int = 0,
    concurrency: Optional[int] = None,
    gen_only: bool = False,
    base_url: Optional[str] = None,
) -> dict:
    # gen_only: generation only; the judge runs separately (scripts/judge_generated.py) over the
    # .gen.json artifacts. gen_only implies no judge calls here.
    skip_judge = no_judge or gen_only
    load_env()  # optional .env: OPENAI_BASE_URL / OPENAI_API_KEY for a remote judge
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    back = cfg["models"]["backbone"]
    # Qwen3 backbones run with thinking disabled (`enable_thinking: false`), exactly as for COMPASS;
    # run.current_date (FreshQA) gives every call the benchmark date, as for COMPASS.
    back_no_think = back.get("enable_thinking") is False
    client = LLMClient(
        base_url=base_url or back["base_url"],  # --base-url routes to a specific vLLM instance
        model=back["model"],
        default_temperature=back.get("temperature", 0.0),
        default_max_tokens=back.get("max_tokens", 2048),
        concurrency=concurrency or cfg.get("client", {}).get("concurrency", 8),
        retries=cfg.get("client", {}).get("retries", 3),
        ledger=TokenLedger(),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} if back_no_think else None,
        strip_thinking=back_no_think,
        system_note=date_note_text(cfg["run"].get("current_date")),
    )

    jm = cfg["models"]["judge"]
    # A local judge (`local: true`) is taken from the config only, never from OPENAI_* in .env,
    # with the same settings as compass/run.py (thinking off, request timeout, fresh connections).
    local_judge = bool(jm.get("local"))
    judge_no_think = jm.get("enable_thinking") is False
    judge_client = None if skip_judge else LLMClient(
        base_url=jm["base_url"] if local_judge else (os.environ.get("OPENAI_BASE_URL") or jm["base_url"]),
        model=judge_model or jm["model"],
        api_key="EMPTY" if local_judge else os.environ.get("OPENAI_API_KEY", "EMPTY"),
        default_temperature=jm.get("temperature", 0.0),
        default_max_tokens=jm.get("max_tokens", 128),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} if judge_no_think else None,
        strip_thinking=judge_no_think,
        request_timeout=float(jm.get("request_timeout", 300.0)),
        keepalive=not local_judge,
    )

    retriever = build_retriever(cfg.get("retrieval"))
    params = (cfg.get("methods", {}) or {}).get(method, {}) or {}
    baseline = build_baseline(
        method, client,
        temperature=back.get("temperature", 0.0),
        max_tokens=back.get("max_tokens", 2048),
        params=params,
        retriever=retriever,
        dataset=cfg["dataset"]["name"],
    )

    ds = cfg["dataset"]
    n = sample_size if sample_size is not None else cfg["run"].get("sample_size")
    split = split if split is not None else ds.get("split", "all")
    dev_size = dev_size if dev_size is not None else ds.get("dev_size", 100)
    examples = load_dataset(ds["name"], ds["path"], n, split=split,
                            dev_size=dev_size, seed=cfg["run"].get("seed", 0))
    print(f"[baseline:{method}] {ds['name']} | split={split} | r{run_index} | {len(examples)} examples")

    async def _one(ex):
        t0 = time.time()
        try:
            res = await baseline.answer(ex.question)
            answer, trace = res.answer, res.trace
        except Exception as err:  # noqa: BLE001 - one failed query must not abort a 1-2 day run
            answer, trace = "", {"error": str(err)}
        return answer, trace, time.time() - t0

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{method}_{ds['name']}_{split}_r{run_index}"
    # gen-only writes a distinct .gen.json artifact (phase A); the judge phase reads it and
    # writes the final <stem>.json. This keeps generation re-judgeable without re-running GPUs.
    suffix = ".gen" if gen_only else ""
    out_path = os.path.join(out_dir, f"{stem}{suffix}.json")
    partial_path = os.path.join(out_dir, f"{stem}{suffix}.partial.json")

    # --- RESUME from a prior crashed run (same mechanism as compass/run.py). The per-role
    # TokenLedger is seeded from the checkpoint's token snapshot, so the reported method tokens
    # cover the whole run; HR/Acc/records are fully resumed. ---
    records: List[dict] = []
    cum_judgments: List[Judgment] = []
    cum_latencies: List[float] = []
    done_ids = set()
    if os.path.exists(partial_path):
        with open(partial_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        for rec in prev.get("records", []):
            records.append(rec)
            done_ids.add(rec["id"])
            jd = rec.get("judgment", {})
            cum_judgments.append(Judgment(jd.get("hallucination", -1), jd.get("accuracy", -1), ""))
            cum_latencies.append(rec.get("latency", 0.0))
        pm = prev.get("metrics") or {}
        client.ledger.restore(pm.get("tokens"))
        if judge_client is not None:
            judge_client.ledger.restore(pm.get("judge_tokens"))
        print(f"[baseline:{method}] resuming from {partial_path}: {len(done_ids)} records done, "
              f"{client.ledger.total_tokens} method tokens carried over")

    remaining = [ex for ex in examples if ex.id not in done_ids]
    n_total = len(examples)
    batch_size = cfg["run"].get("checkpoint_every", 100)

    def _meta(n_records: int, status: Optional[str] = None) -> dict:
        meta = {
            "dataset": ds["name"], "method": method, "model": back["model"],
            "judge_model": judge_model or jm["model"], "config": config_path,
            "current_date": cfg["run"].get("current_date"), "effective_config": cfg,
            "seed": cfg["run"].get("seed"), "n": n_records, "split": split,
            "run_index": run_index,
        }
        if status is not None:
            meta["status"] = status
        return meta

    def _metrics() -> dict:
        m = aggregate(cum_judgments, n_total=n_total,
                      token_snapshot=client.ledger.snapshot(), latencies=cum_latencies)
        if judge_client is not None:
            m["judge_tokens"] = judge_client.ledger.snapshot()
        return m

    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        triples = await asyncio.gather(*[_one(ex) for ex in batch])
        answers = [t[0] for t in triples]

        if skip_judge:
            judgments = [Judgment(-1, -1, "judge deferred (gen-only)" if gen_only else "judge skipped") for _ in batch]
        else:
            # Empty/errored answers -> hallucination=1/accuracy=0 (never sent to the judge; an
            # empty answer must not be rewarded — audit BLOCKER-2). Judge only the non-empty ones.
            ne_idx = [k for k, a in enumerate(answers) if (a or "").strip()]
            ne_judg = await judge_all([batch[k] for k in ne_idx], [answers[k] for k in ne_idx], judge_client)
            judgments = [Judgment(1, 0, "empty answer -> forced wrong") for _ in batch]
            for k, j in zip(ne_idx, ne_judg):
                judgments[k] = j

        for ex, (a, tr, lat), j in zip(batch, triples, judgments):
            records.append({
                "id": ex.id, "question": ex.question, "answer": a,
                "judgment": j.as_dict(), "gold": ex.gold, "trace": tr, "latency": lat,
            })
            cum_judgments.append(j)
            cum_latencies.append(lat)

        partial = {"meta": _meta(len(records), status="partial"),
                   "metrics": _metrics(), "records": records}
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(partial, f, ensure_ascii=False, indent=2, default=str)
        m = partial["metrics"]
        print(f"[checkpoint:{method}/{ds['name']}] {len(records)}/{n_total} | "
              f"HR={m['hallucination_rate']} Acc={m['accuracy']}")

    metrics = _metrics()
    # gen-only keeps the per-example "latency" field (the judge phase reads it back); the judged
    # final file strips it to match the original schema.
    if gen_only:
        result = {"meta": _meta(len(records), status="generated"), "metrics": metrics, "records": records}
    else:
        final_records = [{k: v for k, v in r.items() if k != "latency"} for r in records]
        result = {"meta": _meta(len(records)), "metrics": metrics, "records": final_records}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    if os.path.exists(partial_path):
        os.remove(partial_path)

    if gen_only:
        print(f"[baseline:{method}] GENERATED {len(records)} answers (no judge) -> {out_path}")
    else:
        print(f"[baseline:{method}] HR={metrics['hallucination_rate']} Acc={metrics['accuracy']} "
              f"avg_tok={metrics.get('avg_tokens_per_query')} -> {out_path}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a reproduced baseline on a benchmark")
    ap.add_argument("--config", required=True, help="path to a baseline config YAML")
    ap.add_argument("--method", required=True, help="baseline name (see compass.baselines.registry)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--gen-only", action="store_true",
                    help="generate answers only, write <stem>.gen.json; "
                         "judge later with scripts/judge_generated.py")
    ap.add_argument("--sample-size", type=int, default=None)
    ap.add_argument("--judge-model", type=str, default=None)
    ap.add_argument("--split", choices=["dev", "test", "all"], default=None)
    ap.add_argument("--dev-size", type=int, default=None)
    ap.add_argument("--run-index", type=int, default=0, help="tag for the mean+/-std repeats")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--base-url", type=str, default=None,
                    help="override backbone base_url (route to a specific GPU vLLM, e.g. :8002)")
    args = ap.parse_args()
    asyncio.run(run(args.config, args.method, no_judge=args.no_judge,
                    sample_size=args.sample_size, judge_model=args.judge_model,
                    split=args.split, dev_size=args.dev_size, run_index=args.run_index,
                    concurrency=args.concurrency, gen_only=args.gen_only, base_url=args.base_url))


if __name__ == "__main__":
    main()
