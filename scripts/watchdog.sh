#!/usr/bin/env bash
# Generic experiment watchdog. Polls every --interval sec and EXITS (which notifies the
# launching agent via background-completion) on either a hard failure or normal completion.
# It catches the "hard" failures that silently stall long runs — it does NOT judge whether
# results are sane (that's for a human / an on-demand agent).
#
# Usage:
#   bash scripts/watchdog.sh \
#     --ports "8000 8002 8003 8004" \      # vLLM ports that must stay UP (optional)
#     --logs  "/tmp/sweep_*.log" \          # log glob to grep for hard errors (optional)
#     --proc  "compass.run --config" \      # process pattern: count drops to 0 => DONE (optional)
#     --progress "results/*_rep*.partial.json" \  # files whose mtime must keep advancing (optional)
#     --interval 120 --stall-mins 6
#
# Exit 1 = ALERT (a port down / a hard error in a log / progress stalled while procs run).
# Exit 0 = DONE  (the --proc pattern has dropped to 0 after having been running).
set -u
INTERVAL=120; STALL_MINS=6; PORTS=""; LOGS=""; PROC=""; PROGRESS=""
while [ $# -gt 0 ]; do case "$1" in
  --ports) PORTS="$2"; shift 2;;
  --logs) LOGS="$2"; shift 2;;
  --proc) PROC="$2"; shift 2;;
  --progress) PROGRESS="$2"; shift 2;;
  --interval) INTERVAL="$2"; shift 2;;
  --stall-mins) STALL_MINS="$2"; shift 2;;
  *) echo "watchdog: unknown arg $1"; exit 2;;
esac; done

started=0
while true; do
  alert=""
  # 1) vLLM ports must answer /v1/models
  if [ -n "$PORTS" ]; then
    down=""; for p in $PORTS; do curl -s --max-time 3 "http://localhost:$p/v1/models" 2>/dev/null | grep -q . || down+="$p "; done
    [ -n "$down" ] && alert+="PORT_DOWN[$down]; "
  fi
  # 2) hard errors in logs
  if [ -n "$LOGS" ]; then
    errs=$(grep -liE "Traceback|CUDA out of memory|OutOfMemory|Engine core init|Address already in use|FAILED" $LOGS 2>/dev/null | tr '\n' ' ')
    [ -n "$errs" ] && alert+="ERR_IN[$errs]; "
  fi
  # 3) process count -> completion + stall gating
  nproc=0; [ -n "$PROC" ] && nproc=$(pgrep -f "$PROC" 2>/dev/null | wc -l)
  [ "$nproc" -gt 0 ] && started=1
  if [ -n "$PROC" ] && [ "$started" -eq 1 ] && [ "$nproc" -eq 0 ]; then
    echo "[WATCHDOG done @ $(date +%H:%M:%S)] process '$PROC' finished (count=0)"; exit 0
  fi
  # 4) stall: newest progress file hasn't advanced in --stall-mins while procs still running
  if [ -n "$PROGRESS" ] && [ "$nproc" -gt 0 ]; then
    newest=$(ls -t $PROGRESS 2>/dev/null | head -1)
    if [ -n "$newest" ]; then
      age=$(( ( $(date +%s) - $(stat -c %Y "$newest" 2>/dev/null || date +%s) ) / 60 ))
      [ "$age" -ge "$STALL_MINS" ] && alert+="STALLED(${age}min no progress); "
    fi
  fi
  if [ -n "$alert" ]; then echo "[WATCHDOG ALERT @ $(date +%H:%M:%S)] $alert(nproc=$nproc)"; exit 1; fi
  sleep "$INTERVAL"
done
