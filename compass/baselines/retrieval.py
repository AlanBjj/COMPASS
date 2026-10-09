"""Plain retrieval for the retrieval-augmented baselines (ReAct, Adaptive-RAG, TARG, DTR).

COMPASS's own Fact operator issues source-typed weighted queries (academic/news/official); that
weighting is part of the COMPASS method and is not grafted onto the baselines, whose original
papers use a single standard retriever. In the paper's setup (`backend: local_snapshot`) a baseline
issues ONE plain BM25 query over the whole frozen web-search snapshot (the union index "all" of
compass.snapshot_index) — the same corpus COMPASS retrieves from.

The live web-search backend is kept for completeness: a query that is empty or over-long is
sanitized first, calls are throttled and retried with backoff on 429/5xx/timeouts, a genuine
failure is not cached and degrades to no evidence, and the API key is read from SERPER_API_KEY.
All knobs come from the config `retrieval` block.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import List, Optional

import httpx

from ._retrieval_base import CachedRetriever, Evidence, Retriever, SnapshotCache

# Process-wide throttle on concurrent Serper calls (lazily bound to the running loop). The free
# tier rate-limits aggressively; with 4 GPU processes each running many queries this caps the burst.
_SEM: Optional[asyncio.Semaphore] = None
_SEM_LIMIT = 3


def _sem() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(_SEM_LIMIT)
    return _SEM


class RetrievalError(RuntimeError):
    """Raised when a Serper call fails after exhausting retries (so the cache skips it)."""


class PlainSerperRetriever(Retriever):
    """Single plain web query via Serper, returned as one `Evidence`."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        num_results: int = 5,
        timeout: float = 10.0,
        retries: int = 4,
        backoff: float = 2.0,
        query_max_chars: int = 400,
    ) -> None:
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        self.num_results = num_results
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.query_max_chars = query_max_chars

    @staticmethod
    def _sanitize(query: str, max_chars: int) -> str:
        # Serper 400s on empty or over-long queries; collapse whitespace and length-cap.
        q = " ".join((query or "").split())
        return q[:max_chars]

    async def search(self, query: str) -> List[Evidence]:
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY not set (export it; never hardcode)")
        q = self._sanitize(query, self.query_max_chars)
        if not q:
            # Nothing to search — a deterministic empty result (safe to cache, no API call).
            return [Evidence(source="web", weight=1.0, snippets=[])]

        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                async with _sem():
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(
                            "https://google.serper.dev/search",
                            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                            json={"q": q, "num": self.num_results, "gl": "us", "hl": "en"},
                        )
                # 429 (rate limit) and 5xx are transient -> back off and retry.
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                resp.raise_for_status()  # other 4xx (e.g. 400) -> non-retryable below
                organic = resp.json().get("organic", [])
                snippets = [o["snippet"] for o in organic if o.get("snippet")]
                return [Evidence(source="web", weight=1.0, snippets=snippets)]
            except httpx.HTTPStatusError as err:
                last_err = err
                code = err.response.status_code if err.response is not None else 0
                if code and code != 429 and code < 500:
                    break  # non-retryable client error (e.g. 400) — stop retrying
                await asyncio.sleep(self.backoff * (attempt + 1))
            except (httpx.TimeoutException, httpx.TransportError) as err:
                last_err = err
                await asyncio.sleep(self.backoff * (attempt + 1))
        raise RetrievalError(f"Serper failed after {self.retries} attempts: {last_err}")


class LocalPlainRetriever(Retriever):
    """Single plain BM25 query over the whole frozen snapshot, returned as one `Evidence`."""

    def __init__(self, corpus_path: str, index_dir: str, *, num_results: int = 5,
                 query_max_chars: int = 400) -> None:
        from ..snapshot_index import get_index  # method-agnostic infrastructure

        self.idx = get_index(corpus_path, index_dir)
        self.num_results = num_results
        self.query_max_chars = query_max_chars

    async def search(self, query: str, max_chars: Optional[int] = None) -> List[Evidence]:
        # max_chars overrides the query cap for one call (DTR's pseudo-context path).
        q = PlainSerperRetriever._sanitize(query, max_chars or self.query_max_chars)
        hits = self.idx.search(q, "all", self.num_results) if q else []
        return [Evidence(source="web", weight=1.0, snippets=[h["text"] for h in hits])]


def build_retriever(cfg: Optional[dict]) -> Optional[Retriever]:
    """Build a cached plain-web retriever from the config's `retrieval` block (or None)."""
    if not cfg:
        return None
    if cfg.get("backend") == "local_snapshot":
        # Deterministic local index: no per-query cache needed.
        return LocalPlainRetriever(cfg["corpus"], cfg["index_dir"],
                                   num_results=int(cfg.get("num_results", 5)),
                                   query_max_chars=int(cfg.get("query_max_chars", 400)))
    global _SEM_LIMIT
    _SEM_LIMIT = int(cfg.get("serper_concurrency", 3))
    inner = PlainSerperRetriever(
        num_results=cfg.get("num_results", 5),
        timeout=cfg.get("timeout", 10.0),
        retries=cfg.get("serper_retries", 4),
        backoff=cfg.get("serper_backoff", 2.0),
        query_max_chars=cfg.get("query_max_chars", 400),
    )
    cache_dir = cfg.get("cache_dir", "results/baselines/rag_cache")
    return CachedRetriever(inner, SnapshotCache(cache_dir))


async def web_search(retriever: Retriever, query: str, top_k: Optional[int] = None) -> List[str]:
    """Run a cached search and return a flat list of snippets. DEGRADES to [] on any retrieval
    failure — a Serper hiccup must never crash the calling method (it falls back to no-evidence,
    i.e. parametric-knowledge answering). Failures are not cached (search() raised), so a later
    run retries them."""
    try:
        evidences = await retriever.search(query)
    except Exception as err:  # noqa: BLE001 - degrade, don't crash the query
        print(f"[retrieval] degraded (no evidence) for query {query[:60]!r}: {err}", file=sys.stderr)
        return []
    snippets: List[str] = []
    for ev in evidences:
        snippets.extend(ev.snippets)
    return snippets[:top_k] if top_k else snippets


def format_snippets(snippets: List[str]) -> str:
    return "\n".join(f"- {s}" for s in snippets) if snippets else "(no results)"
