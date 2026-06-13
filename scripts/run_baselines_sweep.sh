#!/usr/bin/env bash
# Sweep the reproduced baselines over the four datasets (parallel track, step ⑨).
# Usage: bash scripts/run_baselines_sweep.sh <split> <run_index> [sample_size]
#   split=dev|test|all   run_index=0|1|2   sample_size optional (omitted = whole split)
# Runs methods cheap->expensive, datasets in order, ONE process at a time (so the single :8001
# vLLM isn't overloaded). Continues past a failed run; logs each to results/baselines_logs/.
set -uo pipefail

ENVPY=python3
SPLIT="${1:-dev}"
RUN_INDEX="${2:-0}"
SAMPLE="${3:-}"
SS_ARG=(); [ -n "$SAMPLE" ] && SS_ARG=(--sample-size "$SAMPLE")

METHODS=(direct self_refine adaptive_rag react cok rowen halusearch)  # cheap -> expensive
DATASETS=(gsm8k truthfulqa strategyqa freshqa)
LOGDIR=results/baselines_logs
mkdir -p "$LOGDIR"

echo "=== baselines sweep | split=$SPLIT run_index=$RUN_INDEX sample=${SAMPLE:-full} | $(date) ==="
for ds in "${DATASETS[@]}"; do
  for m in "${METHODS[@]}"; do
    log="$LOGDIR/${m}_${ds}_${SPLIT}_r${RUN_INDEX}.log"
    echo ">>> [$ds/$m] -> $log"
    "$ENVPY" -m compass.baselines.run \
      --config "configs/baselines/${ds}.yaml" --method "$m" \
      --split "$SPLIT" --run-index "$RUN_INDEX" "${SS_ARG[@]}" \
      > "$log" 2>&1
    tail -n 1 "$log"
  done
done
echo "=== sweep done | $(date) ==="
