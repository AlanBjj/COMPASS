"""End-to-end run entry: `python -m compass.run --config configs/<x>.yaml`.

Loads a config, runs COMPASS over the dataset, judges each answer (LLM judge), aggregates metrics
(HR / Acc / per-role tokens / latency), and writes one results JSON. Single entry point for all
benchmarks (vs the legacy dual-entry mess). Method-cost tokens (controller+backbone) and judge
tokens are tracked separately. The judge needs OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import List

import yaml

from .datasets import load_dataset
from .envload import load_env
from .eval import aggregate, judge_all
from .eval.judge import Judgment
from .llm.client import LLMClient
from .pipeline import build_compass


async def run(config_path: str, no_judge: bool = False, sample_size=None, judge_model=None,
              tau=None, split=None, dev_size=None, base_url=None, tag=None,
              shard=None, num_shards=None) -> dict:
    load_env()  # load .env: OPENAI_BASE_URL / OPENAI_API_KEY / SERPER_API_KEY
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if tau is not None:
        cfg["gate"]["tau"] = tau  # re-calibrated gate threshold for the open-source controller
    if base_url is not None:
        # Route this run to a specific vLLM instance (multi-GPU data parallelism: several
        # 7B instances on different GPUs, independent runs sharded across them).
        cfg["models"]["backbone"]["base_url"] = base_url
        cfg["models"]["controller"]["base_url"] = base_url

    compass = build_compass(cfg)

    jm = cfg["models"]["judge"]
    # judge base_url + model + key come from the environment first (.env); config is fallback.
    # NOTE: model MUST read env too — otherwise the config placeholder name is recorded even
    # though the actual judge is whatever the env endpoint serves (e.g. zovelox = gpt-5.4-mini).
    judge_client = LLMClient(
        base_url=os.environ.get("OPENAI_BASE_URL") or jm["base_url"],
        model=judge_model or os.environ.get("OPENAI_JUDGE_MODEL") or jm["model"],
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        default_temperature=jm.get("temperature", 0.0),
        default_max_tokens=jm.get("max_tokens", 128),
    )

    ds = cfg["dataset"]
    n = sample_size if sample_size is not None else cfg["run"].get("sample_size")
    # --split / --dev-size override the dataset config; defaults come from configs/.
    split = split if split is not None else ds.get("split", "all")
    dev_size = dev_size if dev_size is not None else ds.get("dev_size", 100)
    examples = load_dataset(ds["name"], ds["path"], n, split=split,
                            dev_size=dev_size, seed=cfg["run"].get("seed", 0))
    if shard is not None and num_shards:
        examples = examples[shard::num_shards]  # stride slicing: even, deterministic shards
    print(f"[run] {cfg['run']['name']} | {ds['name']} | split={split} | {len(examples)} examples"
          + (f" | shard {shard}/{num_shards}" if num_shards else ""))

    async def _one(ex):
        t0 = time.time()
        res = await compass.answer(ex.question)
        return res, time.time() - t0

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""  # e.g. _rep1 so the ×3 repeats don't overwrite each other
    out_path = os.path.join(out_dir, f"{cfg['run']['name']}_{ds['name']}{suffix}.json")
    partial_path = os.path.join(out_dir, f"{cfg['run']['name']}_{ds['name']}{suffix}.partial.json")

    # --- RESUME: load any already-finished records from a prior crashed run. ---
    # Records carry per-example id/judgment/path/trace and the latency; we keep cumulative
    # lists of judgments and latencies so aggregate() (which takes judgments + n_total +
    # token_snapshot + latencies) can be called over the running total after each batch.
    # NOTE: the per-role TokenLedger only accumulates within a single process, so on resume
    # the reported method-token counts cover ONLY the current process (the work redone since
    # the restart), not the tokens spent before the crash. HR/Acc/records are fully resumed.
    records: List[dict] = []
    cum_judgments: List[Judgment] = []
    cum_latencies: List[float] = []
    cum_paths: List[str] = []
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
            cum_paths.append(rec.get("path"))
        print(f"[run] resuming from {partial_path}: {len(done_ids)} records already done")

    remaining = [ex for ex in examples if ex.id not in done_ids]
    n_total = len(examples)
    batch_size = cfg["run"].get("checkpoint_every", 100)

    def _snapshot_meta(n_records: int, status: str = None) -> dict:
        # Final file reproduces the original meta exactly; the partial adds a "status" key.
        meta = {
            "dataset": ds["name"], "method": "compass",
            "model": cfg["models"]["backbone"]["model"],
            "judge_model": judge_client.model, "config": config_path,
            "seed": cfg["run"].get("seed"), "n": n_records, "split": split,
            "path_counts": {p: cum_paths.count(p) for p in set(cum_paths)},
        }
        if status is not None:
            meta["status"] = status
        return meta

    def _build_metrics() -> dict:
        m = aggregate(
            cum_judgments,
            n_total=n_total,
            token_snapshot=compass.client.ledger.snapshot(),  # method cost (controller+backbone)
            latencies=cum_latencies,
        )
        m["judge_tokens"] = judge_client.ledger.snapshot()  # eval cost, reported separately
        return m

    # Process the remaining examples in batches, checkpointing after each so a crashed
    # long run can resume (output schema is unchanged; checkpointing is transparent when
    # the run completes in one go).
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        # Concurrent across queries (bounded by the client semaphore). Per-query latency
        # therefore includes queueing — fine for smoke, NOT for the paper's latency table.
        pairs = await asyncio.gather(*[_one(ex) for ex in batch])
        answers = [p[0].answer for p in pairs]
        paths = [p[0].path for p in pairs]
        traces = [p[0].trace for p in pairs]
        latencies = [p[1] for p in pairs]

        if no_judge:
            judgments = [Judgment(-1, -1, "judge skipped (--no-judge)") for _ in batch]
        else:
            judgments = await judge_all(batch, answers, judge_client)

        for ex, a, p, j, tr, lat in zip(batch, answers, paths, judgments, traces, latencies):
            records.append({
                "id": ex.id, "question": ex.question, "answer": a, "path": p,
                "judgment": j.as_dict(), "gold": ex.gold, "trace": tr,
                "latency": lat,
            })
            cum_judgments.append(j)
            cum_latencies.append(lat)
            cum_paths.append(p)

        # Checkpoint: write the partial file and print a cumulative early-look line.
        partial = {
            "meta": _snapshot_meta(len(records), status="partial"),
            "metrics": _build_metrics(),
            "records": records,  # partial records carry an extra "latency" field for resume
        }
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(partial, f, ensure_ascii=False, indent=2, default=str)
        m = partial["metrics"]
        print(f"[checkpoint] {len(records)}/{n_total} | "
              f"HR={m['hallucination_rate']} Acc={m['accuracy']}")

    # All batches done: write the FINAL file (same schema as before) and drop the partial.
    # Strip the resume-only "latency" field so the final records match the original schema.
    metrics = _build_metrics()
    final_records = [{k: v for k, v in r.items() if k != "latency"} for r in records]
    result = {
        "meta": _snapshot_meta(len(records)),  # no "status" key -> original meta schema
        "metrics": metrics,
        "records": final_records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    if os.path.exists(partial_path):
        os.remove(partial_path)

    print(f"[run] HR={metrics['hallucination_rate']} Acc={metrics['accuracy']} "
          f"avg_tok={metrics.get('avg_tokens_per_query')} -> {out_path}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run COMPASS end-to-end on a benchmark")
    ap.add_argument("--config", required=True, help="path to a config YAML")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM judge (free COMPASS-only smoke; no HR/Acc)")
    ap.add_argument("--sample-size", type=int, default=None, help="override run.sample_size")
    ap.add_argument("--judge-model", type=str, default=None,
                    help="override judge model name (e.g. deepseek-chat)")
    ap.add_argument("--tau", type=float, default=None, help="override gate threshold tau")
    ap.add_argument("--split", choices=["dev", "test", "all"], default=None,
                    help="dataset split: tune on dev, run test once (default from config)")
    ap.add_argument("--dev-size", type=int, default=None,
                    help="number of dev examples (non-FreshQA; default from config / 100)")
    ap.add_argument("--base-url", type=str, default=None,
                    help="override backbone+controller vLLM base_url (multi-instance scheduling)")
    ap.add_argument("--tag", type=str, default=None,
                    help="output filename suffix (e.g. rep1) so repeated ×3 runs don't overwrite")
    ap.add_argument("--shard", type=int, default=None, help="shard index (0-based) for stride slicing")
    ap.add_argument("--num-shards", type=int, default=None, help="total number of shards")
    args = ap.parse_args()
    asyncio.run(run(args.config, no_judge=args.no_judge, sample_size=args.sample_size,
                    judge_model=args.judge_model, tau=args.tau, split=args.split,
                    dev_size=args.dev_size, base_url=args.base_url, tag=args.tag,
                    shard=args.shard, num_shards=args.num_shards))


if __name__ == "__main__":
    main()
