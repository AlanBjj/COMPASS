"""Judge daemon: judge test-split generation files as soon as they appear (pipelined with generation).

Watches results/<backbone>/*_test_rep*_gen.json (COMPASS) and
results/<backbone>/baselines/*_test_r*.gen.json (baselines). Both are written only when their run
has finished, so a file that exists is complete. Every generation file without its final judged file is
judged by scripts/judge_generated.py against one of the judge URLs listed in --urls-file. That file is
re-read every loop, so a second judge instance can be added while the daemon runs. At most --per-url files
are judged at a time per URL. A file is claimed with an atomic lock directory (<final>.lockdir), so two
daemons never judge the same file. A file whose judging fails twice is left for a human (logged).
Exits once --done-flag exists (generation finished) and nothing is left to judge or running.

    setsid nohup python scripts/judge_daemon.py --urls-file logs/judge_urls.txt \\
        --done-flag logs/gen.done < /dev/null > logs/judge_daemon.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def now() -> str:
    return dt.datetime.now().strftime("%F %T")


def final_of(gen: str) -> str:
    return gen[: -len(".gen.json")] + ".json" if gen.endswith(".gen.json") else gen[: -len("_gen.json")] + ".json"


def pending(results: str) -> list:
    pats = [os.path.join(results, "*", "*_test_rep*_gen.json"),
            os.path.join(results, "*", "baselines", "*_test_r*.gen.json")]
    gens = sorted({f for p in pats for f in glob.glob(p)})
    return [g for g in gens if not os.path.exists(final_of(g))]


def read_urls(path: str) -> list:
    try:
        return [l.strip() for l in open(path) if l.strip() and not l.lstrip().startswith("#")]
    except FileNotFoundError:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--urls-file", required=True)
    ap.add_argument("--done-flag", required=True)
    ap.add_argument("--per-url", type=int, default=4, help="files judged at once per judge URL")
    ap.add_argument("--concurrency", type=int, default=32, help="judge requests in flight per file")
    ap.add_argument("--log-dir", default="logs/judge")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(args.log_dir, exist_ok=True)

    running = {}   # gen -> (proc, url, lockdir, logfile)
    failures = {}
    print(f"{now()} judge daemon started", flush=True)
    while True:
        for gen, (proc, url, lock, fh) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            try:
                os.rmdir(lock)
            except OSError:
                pass
            del running[gen]
            if rc == 0 and os.path.exists(final_of(gen)):
                print(f"{now()} judged {os.path.basename(final_of(gen))} on {url}", flush=True)
            else:
                failures[gen] = failures.get(gen, 0) + 1
                print(f"{now()} FAIL  {os.path.basename(gen)} rc={rc} (attempt {failures[gen]}; log in {args.log_dir})",
                      flush=True)

        urls = read_urls(args.urls_file)
        todo = [g for g in pending(args.results) if g not in running and failures.get(g, 0) < 2]
        for gen in todo:
            load = {u: sum(1 for _, (_, uu, _, _) in running.items() if uu == u) for u in urls}
            free = [u for u in urls if load[u] < args.per_url]
            if not free:
                break
            url = min(free, key=lambda u: load[u])
            lock = final_of(gen) + ".lockdir"
            try:
                os.mkdir(lock)
            except FileExistsError:
                continue
            fh = open(os.path.join(args.log_dir, os.path.basename(gen) + ".log"), "a")
            proc = subprocess.Popen(
                [sys.executable, "scripts/judge_generated.py", gen, "--judge-base-url", url,
                 "--concurrency", str(args.concurrency)],
                stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            running[gen] = (proc, url, lock, fh)
            print(f"{now()} start {os.path.basename(gen)} on {url}", flush=True)

        left = [g for g in pending(args.results) if failures.get(g, 0) < 2]
        if os.path.exists(args.done_flag) and not running and not left:
            gave_up = [os.path.basename(g) for g, n in failures.items() if n >= 2 and not os.path.exists(final_of(g))]
            print(f"{now()} all judged; gave up on {len(gave_up)}: {gave_up}", flush=True)
            return 1 if gave_up else 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
