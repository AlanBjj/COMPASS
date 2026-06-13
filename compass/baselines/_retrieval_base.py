"""Vendored generic retrieval/caching primitives for the baselines (isolation).

These are method-AGNOSTIC building blocks (an abstract Retriever, an Evidence container, an
on-disk snapshot cache, and a cache wrapper). They were originally borrowed from
compass.solvers.rag_solver, but baselines must NOT depend on any COMPASS *method* module
(gate/decompose/solvers/fusion/pipeline) — see compass/baselines/_check_isolation.py. So we
vendor a self-contained copy here. This is plain caching infrastructure, not the COMPASS RAG
method (no 3-source credibility weighting etc.); the baseline retriever in retrieval.py builds
on these to issue a single plain web query.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Evidence:
    source: str
    weight: float
    snippets: List[str]

    def as_dict(self) -> dict:
        return {"source": self.source, "weight": self.weight, "snippets": self.snippets}

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        return cls(source=d["source"], weight=float(d["weight"]), snippets=list(d["snippets"]))


class Retriever(ABC):
    @abstractmethod
    async def search(self, query: str) -> List[Evidence]:
        ...


class SnapshotCache:
    """On-disk snapshot of retrieval results, keyed by query hash (reproducibility, R3)."""

    def __init__(self, cache_dir: str) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.dir / f"{h}.json"

    def load(self, key: str) -> Optional[List[Evidence]]:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Evidence.from_dict(e) for e in data]

    def save(self, key: str, evidences: List[Evidence]) -> None:
        self._path(key).write_text(
            json.dumps([e.as_dict() for e in evidences], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class CachedRetriever(Retriever):
    """Wrap a retriever with a snapshot cache: replay if cached, else fetch and store."""

    def __init__(self, inner: Retriever, cache: SnapshotCache) -> None:
        self.inner = inner
        self.cache = cache

    async def search(self, query: str) -> List[Evidence]:
        cached = self.cache.load(query)
        if cached is not None:
            return cached
        evidences = await self.inner.search(query)
        self.cache.save(query, evidences)
        return evidences
