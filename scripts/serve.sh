#!/usr/bin/env bash
# Start / stop the vLLM servers used in the paper.
#
#   bash scripts/serve.sh start qwen3-8b judge       # one backbone and the judge
#   bash scripts/serve.sh start qwen3-8b-g5          # an extra Qwen3-8B replica on GPU 5 (port 8115)
#   bash scripts/serve.sh start embed                # bge-m3 embeddings for the DTR baseline
#   bash scripts/serve.sh status
#   bash scripts/serve.sh stop [name ...]            # stops by recorded PID only
#
# name           GPU(s)   port     model (under $MODELS)
# qwen3-8b       0        8100     Qwen3-8B
# llama-3.1-8b   1        8101     Meta-Llama-3.1-8B-Instruct
# ministral-8b   2        8102     Ministral-8B-Instruct-2410
# judge          3,4      8200     Qwen3-32B (tensor parallel 2)
# <backbone>-gN  N        8110+N / 8120+N / 8130+N   extra replicas of the three backbones
# embed          4        8300     bge-m3 (embeddings for the DTR baseline, next to the judge)
#
# Environment: MODELS (model directory), VLLM_ENV (Python environment with vLLM), RUN_DIR (pid and
# log files), JUDGE_GPUS, EMBED_GPU, NCCL_P2P_OFF_ALL=1 (disable NCCL P2P where it hangs at init).
set -euo pipefail

MODELS=${MODELS:-$HOME/models}
EMBED_GPU=${EMBED_GPU:-4}
VLLM_ENV=${VLLM_ENV:-${CONDA_PREFIX:-}}
# vLLM's JIT compiler invokes ninja by name, including from non-interactive shells.
if [[ -n $VLLM_ENV ]]; then export PATH="$VLLM_ENV/bin:$PATH"; fi
RUN_DIR=${RUN_DIR:-$PWD/serve}
mkdir -p "$RUN_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

spec() {  # name -> "gpus port model served extra-args"
  case "$1" in
    qwen3-8b)     echo "0 8100 $MODELS/Qwen3-8B qwen3-8b --max-model-len 16384" ;;
    llama-3.1-8b) echo "1 8101 $MODELS/Meta-Llama-3.1-8B-Instruct llama-3.1-8b-instruct --max-model-len 16384" ;;
    ministral-8b) echo "2 8102 $MODELS/Ministral-8B-Instruct-2410 ministral-8b-instruct --max-model-len 16384 --tokenizer-mode mistral --config-format mistral --load-format mistral" ;;
    # Custom all-reduce can hang during Qwen3-32B warmup on some GPU pairs, so it is disabled.
    judge)        echo "${JUDGE_GPUS:-3,4} 8200 $MODELS/Qwen3-32B qwen3-32b --tensor-parallel-size 2 --max-model-len 8192 --disable-custom-all-reduce" ;;
    qwen3-8b-g[0-7]) g=${1#qwen3-8b-g}; echo "$g $((8110 + g)) $MODELS/Qwen3-8B qwen3-8b --max-model-len 16384" ;;
    llama-3.1-8b-g[0-7]) g=${1#llama-3.1-8b-g}; echo "$g $((8120 + g)) $MODELS/Meta-Llama-3.1-8B-Instruct llama-3.1-8b-instruct --max-model-len 16384" ;;
    ministral-8b-g[0-7]) g=${1#ministral-8b-g}; echo "$g $((8130 + g)) $MODELS/Ministral-8B-Instruct-2410 ministral-8b-instruct --max-model-len 16384 --tokenizer-mode mistral --config-format mistral --load-format mistral" ;;
    # bge-m3 embeddings for the DTR baseline, next to the judge (the later flag overrides 0.85).
    embed)        echo "$EMBED_GPU 8300 $MODELS/bge-m3 bge-m3 --runner pooling --convert embed --max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.09" ;;
    *) echo "unknown server: $1" >&2; exit 2 ;;
  esac
}

start() {
  local name=$1; read -r gpus port model served extra <<<"$(spec "$name")"
  if [[ -f $RUN_DIR/$name.pid ]] && kill -0 "$(cat "$RUN_DIR/$name.pid")" 2>/dev/null; then
    echo "$name already running (pid $(cat "$RUN_DIR/$name.pid"))"; return
  fi
  local p2p_off=0; [[ ${NCCL_P2P_OFF_ALL:-0} == 1 ]] && p2p_off=1
  NCCL_P2P_DISABLE=$p2p_off CUDA_VISIBLE_DEVICES=$gpus setsid nohup vllm serve "$model" \
    --served-model-name "$served" --port "$port" --host 127.0.0.1 \
    --gpu-memory-utilization 0.85 --seed 0 $extra \
    < /dev/null > "$RUN_DIR/$name.log" 2>&1 &
  echo $! > "$RUN_DIR/$name.pid"
  echo "started $name on GPU $gpus port $port (pid $!, log $RUN_DIR/$name.log)"
}

status() {
  for f in "$RUN_DIR"/*.pid; do
    [[ -e $f ]] || { echo "no servers"; return; }
    local name; name=$(basename "$f" .pid); read -r _ port _ <<<"$(spec "$name")"
    if kill -0 "$(cat "$f")" 2>/dev/null; then
      if curl -s -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null; then s=ready; else s=loading; fi
    else s=dead; fi
    echo "$name pid=$(cat "$f") port=$port $s"
  done
}

stop() {
  local names=("$@")
  [[ ${#names[@]} -gt 0 ]] || names=($(ls "$RUN_DIR" | sed -n 's/\.pid$//p'))
  for name in "${names[@]}"; do
    local f=$RUN_DIR/$name.pid
    [[ -f $f ]] || continue
    local pid; pid=$(cat "$f")
    # setsid made the server its own process group: signal that group only.
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    rm -f "$f"; echo "stopped $name (pid $pid)"
  done
}

cmd=${1:-status}; shift || true
case "$cmd" in
  start) for n in "$@"; do start "$n"; done ;;
  stop) stop "$@" ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status [name ...]" >&2; exit 2 ;;
esac
