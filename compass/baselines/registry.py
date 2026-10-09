"""Baseline registry — name -> Baseline subclass for the six baselines of the paper."""

from __future__ import annotations

from typing import Dict, Optional, Type

from ..llm.client import LLMClient
from ._retrieval_base import Retriever
from .adaptive_rag import AdaptiveRagBaseline
from .base import Baseline
from .direct import DirectBaseline
from .dtr import DtrBaseline
from .react import ReActBaseline
from .self_consistency import SelfConsistencyBaseline
from .targ import TargBaseline

BASELINES: Dict[str, Type[Baseline]] = {
    "direct": DirectBaseline,
    "adaptive_rag": AdaptiveRagBaseline,
    "react": ReActBaseline,
    "self_consistency": SelfConsistencyBaseline,
    # TARG (arXiv 2511.09803): the gated method, and its two branches for tau calibration on dev.
    "targ": TargBaseline,
    "targ_never": TargBaseline,
    "targ_always": TargBaseline,
    # DTR (arXiv 2601.03908): same pattern.
    "dtr": DtrBaseline,
    "dtr_never": DtrBaseline,
    "dtr_always": DtrBaseline,
}


def build_baseline(
    name: str,
    client: LLMClient,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    params: Optional[Dict[str, object]] = None,
    retriever: Optional[Retriever] = None,
    dataset: str = "",
) -> Baseline:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline: {name} (known: {sorted(BASELINES)})")
    return BASELINES[name](
        client, temperature=temperature, max_tokens=max_tokens, params=params,
        retriever=retriever, dataset=dataset,
    )
