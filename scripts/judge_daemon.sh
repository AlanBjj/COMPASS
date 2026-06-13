#!/usr/bin/env bash
# Judge daemon (phase B, overlapped). Repeatedly judges any *.gen.json that lacks a final *.json,
# so the judge runs CONCURRENTLY with GPU generation (they share no resources). Exits
# once the generation sweep has finished AND every .gen.json has been judged.
# Usage: bash scripts/judge_daemon.sh <gen_sweep_log> [concurrency]
set -uo pipefail
ENVPY=python3
GEN_LOG="${1:-results/baselines_logs/gen_sweep_dev_r0.log}"
CONC="${2:-20}"
DIR=results/baselines

while true; do
  "$ENVPY" -m compass.baselines.judge_runs "$DIR" --concurrency "$CONC" 2>&1 | grep -E "DONE|to judge|resuming" || true
  gen_done=$(grep -c 'GEN sweep done' "$GEN_LOG" 2>/dev/null || echo 0)
  unjudged=0
  for f in "$DIR"/*.gen.json; do
    [ -e "$f" ] || continue
    base="${f%.gen.json}"; [ -f "${base}.json" ] || unjudged=$((unjudged+1))
  done
  echo "[daemon] gen_done=$gen_done unjudged=$unjudged | $(date +%H:%M:%S)"
  if [ "$gen_done" -ge 1 ] && [ "$unjudged" -eq 0 ]; then
    echo "[daemon] all generated answers judged. exiting."
    break
  fi
  sleep 45
done
