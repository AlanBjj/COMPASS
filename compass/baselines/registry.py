"""Baseline registry — name -> Baseline subclass. The new methods (ReAct, Self-Refine) and
the paper's existing 5 (Direct, CoK, HaluSearch, Rowen, Adaptive-RAG). Toolformer is disclosed,
not reproduced (needs training) — intentionally absent."""

from __future__ import annotations

from typing import Dict, Optional, Type

from ..llm.client import LLMClient
from ._retrieval_base import Retriever
from .adaptive_rag import AdaptiveRagBaseline
from .base import Baseline
from .cok import CoKBaseline
from .direct import DirectBaseline
from .halusearch import HaluSearchBaseline
from .react import ReActBaseline
from .rowen import RowenBaseline
from .self_refine import SelfRefineBaseline

BASELINES: Dict[str, Type[Baseline]] = {
    "direct": DirectBaseline,
    "cok": CoKBaseline,
    "halusearch": HaluSearchBaseline,
    "rowen": RowenBaseline,
    "adaptive_rag": AdaptiveRagBaseline,
    "react": ReActBaseline,
    "self_refine": SelfRefineBaseline,
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
