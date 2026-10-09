"""COMPASS — Cognitively-Oriented Multi-Pathway Adaptive Strategy Selection.

Every component runs on the same open instruct model that answers the question (served with vLLM).
The package is organized one module per method component (gate / decompose / solvers / fusion /
aggregation / eval). All hyperparameters come from configs/, all prompts from prompts/.
"""

__version__ = "0.1.0"
