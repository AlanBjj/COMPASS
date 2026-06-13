#!/usr/bin/env bash
# PHASE A (generation, GPU-only) of the two-phase baseline sweep. Distributes the 4 datasets
# across the 4 GPU vLLMs (GPU 2/5/6/7 = ports 8001/8002/8003/8004) and runs them IN PARALLEL;
# within each dataset the 7 methods run sequentially (cheap->expensive). No LLM judge here —
# judge separately with: python -m compass.baselines.judge_runs results/baselines [--concurrency N]
#
# Usage: bash scripts/run_baselines_gen.sh <split> <run_index> [sample_size]
#   split=dev|test|all   run_index=0|1|2   sample_size optional (omit = whole split)
set -uo pipefail
ENVPY=python3
SPLIT="${1:-dev}"; RUN_INDEX="${2:-0}"; SAMPLE="${3:-}"
SS_ARG=(); [ -n "$SAMPLE" ] && SS_ARG=(--sample-size "$SAMPLE")
METHODS=(direct self_refine adaptive_rag react cok rowen halusearch)  # cheap -> expensive
LOGDIR=results/baselines_logs; mkdir -p "$LOGDIR"

# dataset -> GPU vLLM port (one dataset per GPU)
declare -A PORT=( [gsm8k]=8001 [truthfulqa]=8002 [strategyqa]=8003 [freshqa]=8004 )

run_dataset() {  # $1 = dataset
  local ds="$1" port="${PORT[$1]}"
  for m in "${METHODS[@]}"; do
    local log="$LOGDIR/gen_${m}_${ds}_${SPLIT}_r${RUN_INDEX}.log"
    echo ">>> [gen $ds/$m @:$port] -> $log"
    "$ENVPY" -m compass.baselines.run --config "configs/baselines/${ds}.yaml" --method "$m" \
      --split "$SPLIT" --run-index "$RUN_INDEX" --gen-only --base-url "http://localhost:${port}/v1" \
      "${SS_ARG[@]}" > "$log" 2>&1
    tail -n 1 "$log"
  done
  echo "=== [gen $ds] all methods done ==="
}

echo "=== GEN sweep | split=$SPLIT r$RUN_INDEX sample=${SAMPLE:-full} | $(date) ==="
for ds in gsm8k truthfulqa strategyqa freshqa; do run_dataset "$ds" & done
wait
echo "=== GEN sweep done | $(date) === then: $ENVPY -m compass.baselines.judge_runs results/baselines"
