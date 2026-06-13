#!/usr/bin/env bash
# Run COMPASS end-to-end on a benchmark.
#   bash scripts/run.sh --config configs/dev.yaml
# Requires: a vLLM server (bash scripts/serve.sh 7b) and OPENAI_API_KEY for the LLM judge.
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m compass.run "$@"
