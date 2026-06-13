#!/usr/bin/env bash
set -euo pipefail

# Primary backbone/controller — matches the paper's Qwen2-7B backbone.
MODEL_7B=/home/share/models/Qwen2-7B-Instruct
# Secondary robustness check (TBD): a larger same-generation Qwen2 model. Qwen2 has no dense
# 14B, so the secondary is Qwen2-72B-Instruct (needs multi-GPU; not downloaded yet).
MODEL_LARGE=/home/share/models/Qwen2-72B-Instruct
PORT=8000

case "${1:-7b}" in
  7b)    MODEL=$MODEL_7B;    TP=1; GPUS=${GPU:-0}; NAME=qwen2-7b-instruct  ;;  # GPU=1 bash scripts/serve.sh 7b
  large) MODEL=$MODEL_LARGE; TP=4; GPUS=0,1,2,3; NAME=qwen2-72b-instruct ;;
  *) echo "usage: $0 [7b|large]"; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES=$GPUS vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --tensor-parallel-size "$TP" \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
