# COMPASS

**Complexity-Aware Adaptive Reasoning for hallucination mitigation.**

COMPASS routes each query through a **Hybrid Decomposition Gate**, decomposes complex queries
into typed sub-queries, answers each with a **type-specific solver** (math / logic /
fact-RAG / general), and integrates the sub-answers with **Confidence-Aware Fusion**. Two
type-conditioned consistency mechanisms — **TRACE** (type-routed numeric voting for math) and
**CPC** (cross-perspective consistency for misconception-prone facts) — stabilize the answers.
Every component, including the controller, runs on a single open **Qwen2-7B-Instruct** model
served via vLLM (no proprietary model anywhere in the pipeline).

## Repository layout

```
compass/        core pipeline: gate/ decompose/ solvers/ fusion/ eval/
                + baselines/  (seven inference-time baselines)
configs/        one YAML per run; every hyperparameter lives here
prompts/        one .txt per prompt (gate, decomposition, solvers, TRACE/CPC, judge, ...)
scripts/        serve.sh, run.sh, and evaluation/utility scripts
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in your API keys
```

The evaluation judge reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the environment
(any OpenAI-compatible endpoint). The Fact solver's live retrieval reads `SERPER_API_KEY`.

## Run

```bash
# 1. serve the backbone+controller (Qwen2-7B-Instruct) via vLLM
bash scripts/serve.sh 7b

# 2. run COMPASS on a benchmark
bash scripts/run.sh --config configs/truthfulqa.yaml      # or gsm8k / strategyqa / freshqa

# 3. aggregate metrics (Hallucination Rate, Accuracy, per-role token cost, latency)
python -m compass.eval.report results/<run_id>
```

Baselines are served and run via `scripts/serve_baselines.sh` and
`scripts/run_baselines_gen.sh`. All settings are read from `configs/`; no hyperparameter is
hardcoded.

## Conventions

- One module per pipeline component; no monolithic scripts.
- Every hyperparameter (gate α/τ, fusion 0.6/0.4, RAG source weights, TRACE k, …) is read
  from `configs/`, never hardcoded.
- Every prompt is a file under `prompts/`, never inlined in code.
- Each run records its config, seed, model name, and output path under `results/`.
