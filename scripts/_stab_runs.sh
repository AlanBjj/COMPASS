#!/usr/bin/env bash
# STEP 2 driver: run a dataset's dev split twice (temp=0), copying each result to a distinct
# stability file so they don't overwrite. Usage: _stab_runs.sh <config> <dsname> <tag>
set -euo pipefail
PY=python3
CONFIG="$1"; DS="$2"; TAG="$3"
cd "$(dirname "$0")/.."
$PY -m compass.run --config "$CONFIG" --split dev --tau 0.55
cp "results/${DS}_${DS}.json" "results/_stab_${TAG}_A.json"
echo "=== ${TAG} A done ==="
$PY -m compass.run --config "$CONFIG" --split dev --tau 0.55
cp "results/${DS}_${DS}.json" "results/_stab_${TAG}_B.json"
echo "=== ${TAG} B done ==="
