#!/usr/bin/env bash
set -euo pipefail

# Second vLLM for the PARALLEL baselines track: same backbone as
# COMPASS (Qwen2-7B-Instruct) on a SEPARATE free GPU + port, so it never contends with the main
# thread's :8000 server on GPU 1. Mirrors scripts/serve.sh; kept separate so the baselines track
# never edits the core serve script.
#
# Two robustness fixes vs a bare `vllm serve`:
#   1. prepend the compass env bin to PATH so flashinfer's JIT can find `ninja`;
#   2. VLLM_USE_FLASHINFER_SAMPLER=0 — greedy (temperature=0) eval needs no top-k/top-p sampling,
#      so we use the native torch sampler and avoid the flashinfer JIT/ninja dependency entirely.

ENVBIN=$HOME/.local/bin
MODEL=/home/share/models/Qwen2-7B-Instruct
NAME=qwen2-7b-instruct
GPU=${GPU:-2}        # free GPU; main thread uses GPU 1. Override: GPU=3 bash scripts/serve_baselines.sh
PORT=${PORT:-8001}

export PATH="$ENVBIN:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES="$GPU"

exec "$ENVBIN/vllm" serve "$MODEL" \
  --served-model-name "$NAME" \
  --tensor-parallel-size 1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
