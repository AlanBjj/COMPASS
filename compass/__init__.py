"""COMPASS — Cognitively-Oriented Multi-Pathway Adaptive Strategy Selection.

Open-source reimplementation: the Hybrid Decomposition Gate and Type-Aware Answering
run on an open-source instruct model (Qwen2-7B-Instruct via vLLM) instead of a larger proprietary model.
The package is organized one module per method component (gate / decompose / solvers /
fusion / eval), matching paper/main.tex. All hyperparameters come from configs/, all
prompts from prompts/.
"""

__version__ = "0.1.0"
