"""Baseline methods reproduced on the SAME backbone (Qwen2-7B-Instruct) as COMPASS.

These exist so the paper's main table is a fair, controlled comparison: every baseline runs
through the identical evaluation protocol as COMPASS (temperature=0 backbone, LLM judge,
same dev/test split, mean +/- std over 3 runs) and its token cost is tallied by the same
per-role ledger. They ADD to the repo (never modify COMPASS core: gate/decompose/solvers/
fusion or their prompts/configs).

Each baseline is a `Baseline` subclass whose `answer(question)` returns a `BaselineResult`
(answer string + trace). The generic runner (compass.baselines.run) wires any baseline to the
shared dataset loaders, judge, and metric aggregation — exactly the path compass/run.py uses.
"""

from .base import Baseline, BaselineResult
from .registry import BASELINES, build_baseline

__all__ = ["Baseline", "BaselineResult", "BASELINES", "build_baseline"]
