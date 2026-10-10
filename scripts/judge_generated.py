"""Judge generation-only result files: <name>_gen.json -> <name>.json.

Also baseline generation files (`python -m compass.baselines.run ... --gen-only`): <stem>.gen.json ->
<stem>.json, with the baselines' own in-run rule kept: an empty answer is scored hallucinated without
a judge call (compass/baselines/run.py), and the final records drop the resume-only "latency" field.

Generation (`python -m compass.run ... --no-judge --tag <tag>_gen`) and judging are split so the GPUs
can serve several backbone copies first and the 32B judge afterwards. This script runs ONLY the
judge over the stored answers. It reproduces the in-run judging exactly:

  * judge client built from the generation file's meta.effective_config.models.judge, the same way
    compass/run.py builds it (local vLLM model, thinking disabled, no OPENAI_* environment);
  * reference answers come from the dataset itself (reloaded by id), not from the stored `gold`, so
    the judge input is identical to the in-run judge (same as scripts/second_judge.py);
  * compass.eval.judge_all (same prompt and parser); metrics via compass.eval.metrics.aggregate.

Safety: the generation file is never modified. A complete final file is skipped; an incomplete one
stops the script (it is never overwritten). Progress is checkpointed to <final>.judging.partial.json
after every batch, so an interrupted run resumes where it stopped.

    python scripts/judge_generated.py results/qwen3-8b/*_gen.json results/qwen3-8b/baselines/*.gen.json
    python scripts/judge_generated.py results/qwen3-8b/compass_gsm8k_test_rep1_gen.json --dry-run --limit 5
    # split a large job over two instances of the same judge:
    python scripts/judge_generated.py <files A> &
    python scripts/judge_generated.py <files B> --judge-base-url http://localhost:8201/v1 &
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compass.datasets import load_dataset  # noqa: E402
from compass.eval import judge_all  # noqa: E402
from compass.eval.judge import Judgment  # noqa: E402
from compass.eval.metrics import aggregate  # noqa: E402
from compass.llm.client import LLMClient  # noqa: E402

MAX_RETRIES = 2  # extra attempts for judgments that could not be parsed


class StopError(RuntimeError):
    """A condition that needs a human decision (never resolved by overwriting)."""


def is_baseline_gen(gen_path: str) -> bool:
    return gen_path.endswith(".gen.json")


def final_path_for(gen_path: str) -> str:
    if is_baseline_gen(gen_path):
        return gen_path[: -len(".gen.json")] + ".json"
    if not gen_path.endswith("_gen.json"):
        raise StopError(f"not a generation file (expected *_gen.json or *.gen.json): {gen_path}")
    return gen_path[: -len("_gen.json")] + ".json"


def build_judge(jm: dict, concurrency: int) -> LLMClient:
    """Same construction as compass/run.py for a local judge; never reads OPENAI_* variables."""
    if not jm.get("local"):
        raise StopError(f"judge config is not marked local: {jm}")
    no_think = jm.get("enable_thinking") is False
    return LLMClient(
        base_url=jm["base_url"], model=jm["model"], api_key="EMPTY",
        default_temperature=jm.get("temperature", 0.0),
        default_max_tokens=jm.get("max_tokens", 512),
        concurrency=concurrency,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} if no_think else None,
        strip_thinking=no_think,
        request_timeout=float(jm.get("request_timeout", 300.0)),
        keepalive=False,
    )


def is_complete(d: dict) -> bool:
    recs = d.get("records") or []
    return bool(recs) and all((r.get("judgment") or {}).get("hallucination") in (0, 1) for r in recs)


def load_examples(cfg: dict, records: list) -> list:
    """Reload each record's Example from the dataset by id and check the question text matches."""
    ds = cfg["dataset"]
    exs = load_dataset(ds["name"], ds["path"], None, split="all",
                       dev_size=ds.get("dev_size", 100), seed=cfg["run"].get("seed", 0))
    by_id = {str(e.id): e for e in exs}
    out = []
    for r in records:
        ex = by_id.get(str(r["id"]))
        if ex is None:
            raise StopError(f"record id {r['id']!r} not found in dataset {ds['name']}")
        if " ".join(ex.question.split()) != " ".join(str(r["question"]).split()):
            raise StopError(f"question text mismatch for id {r['id']!r} in {ds['name']}")
        out.append(ex)
    return out


async def judge_with_retry(examples, answers, client) -> list:
    judgments = await judge_all(examples, answers, client)
    for _ in range(MAX_RETRIES):
        bad = [i for i, j in enumerate(judgments) if not j.valid]
        if not bad:
            break
        redo = await judge_all([examples[i] for i in bad], [answers[i] for i in bad], client)
        for i, j in zip(bad, redo):
            judgments[i] = j
    return judgments


async def judge_file(gen_path: str, args) -> str:
    final_path = final_path_for(gen_path)
    partial_path = final_path + ".judging.partial.json"
    gen = json.load(open(gen_path, encoding="utf-8"))
    records = gen["records"]
    meta = gen["meta"]
    cfg = meta.get("effective_config")
    if not cfg:
        raise StopError(f"{gen_path}: meta.effective_config missing (was it run with --variant?)")

    if not args.dry_run and os.path.exists(final_path):
        fin = json.load(open(final_path, encoding="utf-8"))
        if is_complete(fin) and len(fin["records"]) == len(records):
            return f"skip   {os.path.basename(final_path)} (already complete, n={len(records)})"
        raise StopError(f"{final_path} exists but is incomplete; not overwriting")

    if args.limit:
        records = records[: args.limit]
    examples = load_examples(cfg, records)
    jm = dict(cfg["models"]["judge"])
    if args.judge_base_url:
        # Another instance of the SAME judge (e.g. judge2 on port 8201) to split a large job; the
        # model name and every other setting stay as configured, and the endpoint used is recorded.
        jm["base_url"] = args.judge_base_url
    client = build_judge(jm, args.concurrency)

    # Resume: judgments already written to the partial file, keyed by record id.
    done = {}
    if not args.dry_run and os.path.exists(partial_path):
        prev = json.load(open(partial_path, encoding="utf-8"))
        if prev.get("judged_from") != os.path.basename(gen_path):
            raise StopError(f"{partial_path} belongs to another generation file")
        done = {str(k): v for k, v in prev["judgments"].items()}

    baseline = is_baseline_gen(gen_path)
    if baseline:
        # Same rule as the baselines' in-run judging: an empty answer is wrong, never sent to the judge.
        for r in records:
            if str(r["id"]) not in done and not (r.get("answer") or "").strip():
                done[str(r["id"])] = Judgment(1, 0, "empty answer -> forced wrong").as_dict()
    todo = [i for i, r in enumerate(records) if str(r["id"]) not in done]
    for s in range(0, len(todo), args.batch):
        idx = todo[s: s + args.batch]
        js = await judge_with_retry([examples[i] for i in idx], [records[i]["answer"] for i in idx], client)
        for i, j in zip(idx, js):
            done[str(records[i]["id"])] = j.as_dict()
        if not args.dry_run:
            tmp = partial_path + ".tmp"
            json.dump({"judged_from": os.path.basename(gen_path), "judgments": done}, open(tmp, "w"))
            os.replace(tmp, partial_path)

    judgments = [Judgment(done[str(r["id"])]["hallucination"], done[str(r["id"])]["accuracy"], "")
                 for r in records]
    gm = gen.get("metrics") or {}
    # With --limit the generation token total covers more records than were judged, so leave it out.
    metrics = aggregate(judgments, n_total=len(records), token_snapshot=None if args.limit else gm.get("tokens"))
    for k in ("avg_latency_s",):  # per-record latency is not kept in generation files
        if k in gm:
            metrics[k] = gm[k]
    metrics["judge_tokens"] = client.ledger.snapshot()

    out_records = []
    for r in records:
        rr = {k: v for k, v in r.items() if not (baseline and k == "latency")}
        rr["judgment"] = done[str(r["id"])]
        out_records.append(rr)
    out_meta = dict(meta)
    out_meta.pop("status", None)   # "generated" in baseline generation files
    out_meta.update({
        "judge_model": client.model,
        "judged_from": os.path.basename(gen_path),
        "judged_at": dt.datetime.now().isoformat(timespec="seconds"),
        "judge_config": jm,
        "n": len(out_records),
    })
    n_bad = sum(1 for j in judgments if not j.valid)
    line = (f"{'dryrun' if args.dry_run else 'judged'} {os.path.basename(final_path):40s} n={len(out_records):4d} "
            f"HR={metrics['hallucination_rate']} Acc={metrics['accuracy']} unparsable={n_bad} "
            f"tok/q={metrics.get('avg_tokens_per_query')}")
    if args.dry_run:
        return line
    tmp = final_path + ".tmp"
    json.dump({"meta": out_meta, "metrics": metrics, "records": out_records},
              open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, final_path)
    os.remove(partial_path)
    return line


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="*_gen.json (COMPASS) or *.gen.json (baseline) files (globs ok)")
    ap.add_argument("--dry-run", action="store_true", help="judge but write nothing")
    ap.add_argument("--limit", type=int, default=None, help="only the first N records (use with --dry-run)")
    ap.add_argument("--batch", type=int, default=100, help="records per checkpoint")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--judge-base-url", default=None,
                    help="use another instance of the same judge, e.g. http://localhost:8201/v1")
    args = ap.parse_args()
    if args.limit and not args.dry_run:
        ap.error("--limit is only allowed with --dry-run (a final file must contain every record)")
    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) or [p])
    files = [f for f in files if f.endswith("_gen.json") or f.endswith(".gen.json")]
    if not files:
        print("no *_gen.json / *.gen.json files matched")
        return 1
    status = 0
    for f in files:
        try:
            print(await judge_file(f, args), flush=True)
        except StopError as err:
            print(f"STOP   {err}", flush=True)
            status = 2
            break
    return status


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
