#!/usr/bin/env bash
# COMPASS full-test ×3 sweep, sharded across 4 vLLM instances (data parallelism).
# Each worker owns one endpoint and runs its jobs serially; load-balanced so the heavy
# StrategyQA (2190) ×3 lands on 3 separate instances and GSM8K ×3 gets its own instance.
# Endpoints: 8000=GPU1, 8002=GPU0, 8003=GPU3, 8004=GPU4 (8001=GPU2 is the baseline track).
# Configs are locked to split:test, sample_size:null (full test). --tag repN keeps the 3
# repeats from overwriting; run.py checkpoints every 100 so a crash resumes.
set -u
cd "$(dirname "$0")/.."
PY=python3

worker(){ ep=$1; shift
  for spec in "$@"; do
    ds=${spec%%:*}; r=${spec##*:}
    echo "[$(date +%H:%M:%S) ep$ep] START $ds rep$r"
    # --no-judge: GENERATION ONLY, so GPUs run flat-out and never wait on the slow csun relay.
    # Judging is decoupled: run scripts/judge_results.py afterwards (rate-limited + retried).
    $PY -m compass.run --config configs/$ds.yaml --split test --no-judge \
      --base-url http://localhost:$ep/v1 --tag rep$r > /tmp/sweep_${ds}_rep${r}.log 2>&1
    echo "[$(date +%H:%M:%S) ep$ep] DONE  $ds rep$r -> results/${ds}_${ds}_rep${r}.json"
  done
}

# GSM8K is heaviest (~50min/rep, decompose path) — spread its 3 reps across 3 instances so it
# isn't a serial bottleneck; StrategyQA ×3 (lighter, mostly direct) gets the 4th. ~75min makespan.
worker 8000 gsm8k:1 truthfulqa:1 freshqa:1 &
worker 8002 gsm8k:2 truthfulqa:2 freshqa:2 &
worker 8003 gsm8k:3 truthfulqa:3 freshqa:3 &
worker 8004 strategyqa:1 strategyqa:2 strategyqa:3 &
wait
echo "=== ALL 12 RUNS DONE @ $(date +%H:%M:%S) ==="
