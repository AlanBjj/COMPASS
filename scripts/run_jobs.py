"""Run a grid of jobs (dataset x method x repeat) over one or more backbone vLLM instances.

A COMPASS job is `python -m compass.run --config configs/<backbone>/<ds>.yaml [--variant <ablation>]
--split <split> --tag <tag> --base-url URL`, written to results/<backbone>/compass_<ds>[_<ablation>]_<tag>.json
with tag = <split>[_<prefix>]_rep<K>. A method written "B.<name>" (e.g. B.targ) is a baseline job:
`python -m compass.baselines.run --config configs/<backbone>/baselines/<ds>.yaml --method <name> --split
<split> --run-index <K> --base-url URL`, written to results/<backbone>/baselines/<name>_<ds>_<split>_r<K>.json.

Jobs are spread over the given backbone URLs, at most --lanes jobs per URL at a time. A finished
result file is skipped; an interrupted job resumes from its .partial.json, so the script can simply be
started again. --gen-only generates without judging (COMPASS *_gen.json, baselines *.gen.json); judge
afterwards with scripts/judge_generated.py (or scripts/judge_daemon.py while generation runs).
--extra-args is appended to every COMPASS command (e.g. "--set gate.tau=0.40") and --tag-prefix names
those runs (e.g. "tau0.40"), so sensitivity runs never collide with the main runs.

    python scripts/run_jobs.py --backbone qwen3-8b --split test --methods full no-cpc B.direct B.targ \\
        --reps 1 2 3 --urls http://localhost:8100/v1 http://localhost:8110/v1 --gen-only
    python scripts/run_jobs.py --backbone qwen3-8b --split dev --methods full --reps 1 2 3 \\
        --extra-args "--set gate.tau=0.40" --tag-prefix tau0.40 --urls http://localhost:8100/v1
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import shlex
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def now() -> str:
    return dt.datetime.now().strftime("%F %T")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen3-8b")
    ap.add_argument("--methods", nargs="+", default=["full"],
                    help="'full' (COMPASS), an ablation name (compass.run.ABLATIONS), or B.<baseline>")
    ap.add_argument("--reps", nargs="+", type=int, default=[1])
    ap.add_argument("--datasets", nargs="+", default=["freshqa", "truthfulqa", "strategyqa", "gsm8k"])
    ap.add_argument("--jobs", nargs="+", default=None, metavar="DS:METHOD:REP",
                    help="explicit job list, run in this order (overrides --methods/--reps/--datasets)")
    ap.add_argument("--urls", nargs="+", required=True, help="backbone vLLM base URLs (.../v1)")
    ap.add_argument("--lanes", type=int, default=2, help="concurrent jobs per URL")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--gen-only", action="store_true", help="generate only (judge later with judge_generated.py)")
    ap.add_argument("--dry-run", action="store_true", help="print each queued job's command and output, run nothing")
    ap.add_argument("--extra-args", default="", help="extra arguments for every COMPASS command")
    ap.add_argument("--tag-prefix", default="", help="names COMPASS runs with --extra-args, e.g. tau0.40")
    args = ap.parse_args()
    os.chdir(ROOT)

    env = dict(os.environ, no_proxy="localhost,127.0.0.1", NO_PROXY="localhost,127.0.0.1")
    out_dir = f"results/{args.backbone}"
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    if args.jobs:
        jobs = [(j.split(":")[0], j.split(":")[1], int(j.split(":")[2])) for j in args.jobs]
    else:
        jobs = [(ds, m, rep) for rep in args.reps for m in args.methods for ds in args.datasets]
    queue: asyncio.Queue = asyncio.Queue()

    def tag(rep: int) -> str:
        return "_".join([args.split] + ([args.tag_prefix] if args.tag_prefix else []) + [f"rep{rep}"])

    def final_path(ds: str, method: str, rep: int) -> str:
        if method.startswith("B."):
            return os.path.join(out_dir, "baselines", f"{method[2:]}_{ds}_{args.split}_r{rep}.json")
        name = f"compass_{ds}" + ("" if method == "full" else f"_{method}")
        return os.path.join(out_dir, f"{name}_{tag(rep)}.json")

    def gen_path(ds: str, method: str, rep: int) -> str:
        final = final_path(ds, method, rep)
        return final[:-5] + (".gen.json" if method.startswith("B.") else "_gen.json")

    def command(ds: str, method: str, rep: int, url: str) -> list:
        if method.startswith("B."):
            return ([sys.executable, "-m", "compass.baselines.run",
                     "--config", f"configs/{args.backbone}/baselines/{ds}.yaml",
                     "--method", method[2:], "--split", args.split, "--run-index", str(rep), "--base-url", url]
                    + (["--gen-only"] if args.gen_only else []))
        return ([sys.executable, "-m", "compass.run",
                 "--config", f"configs/{args.backbone}/{ds}.yaml",
                 "--split", args.split, "--base-url", url]
                + ([] if method == "full" else ["--variant", method])
                + (["--no-judge", "--tag", f"{tag(rep)}_gen"] if args.gen_only else ["--tag", tag(rep)])
                + shlex.split(args.extra_args))

    for ds, method, rep in jobs:
        final = final_path(ds, method, rep)
        if os.path.exists(final) or (args.gen_only and os.path.exists(gen_path(ds, method, rep))):
            print(f"{now()} skip  {ds} {method} rep{rep} (done)", flush=True)
        else:
            queue.put_nowait((ds, method, rep))
    print(f"{now()} {queue.qsize()} jobs over {len(args.urls)} URLs x {args.lanes} lanes "
          f"(split={args.split}{', gen-only' if args.gen_only else ''})", flush=True)
    failed = []
    if args.dry_run:
        k = 0
        while not queue.empty():
            ds, method, rep = queue.get_nowait()
            url = args.urls[k % len(args.urls)]
            k += 1
            print(" ".join(command(ds, method, rep, url)), "  ->",
                  gen_path(ds, method, rep) if args.gen_only else final_path(ds, method, rep))
        return 0

    async def lane(url: str) -> None:
        while True:
            try:
                ds, method, rep = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            log = os.path.join(log_dir, f"{args.backbone}_{ds}_{method}_{tag(rep)}"
                               f"{'_gen' if args.gen_only else ''}.log")
            cmd = command(ds, method, rep, url)
            print(f"{now()} start {ds} {method} rep{rep} on {url}", flush=True)
            with open(log, "a") as fh:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
                rc = await proc.wait()
            print(f"{now()} {'done ' if rc == 0 else 'FAIL '} {ds} {method} rep{rep} rc={rc} (log {log})",
                  flush=True)
            if rc != 0:
                failed.append((ds, method, rep))

    await asyncio.gather(*[lane(u) for u in args.urls for _ in range(args.lanes)])
    print(f"{now()} all done, {len(failed)} failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
