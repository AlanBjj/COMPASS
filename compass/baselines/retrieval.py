"""Plain web retrieval for retrieval-augmented baselines (ReAct / Adaptive-RAG / Rowen / CoK).

COMPASS's own RAG solver does three SOURCE-TYPED weighted queries (academic/news/official) —
that weighting is part of the COMPASS method and would be unfair to graft onto the baselines,
whose original papers use a single standard retriever. So here we issue ONE plain Serper query
per call. We reuse the vendored `SnapshotCache` + `CachedRetriever` + `Evidence` (a single
`Evidence(source="web", weight=1.0)`) so baseline retrieval is cached/replayable.

Robustness (after the dev-preview audit found 676 empty answers from Serper 429/400 — see
docs/baselines_audit.md): a query that is empty or over-long is sanitized BEFORE calling Serper;
calls are throttled by a process-wide semaphore and retried with backoff on 429/5xx/timeout; and
a genuine failure RAISES (so `CachedRetriever` does NOT cache a failure) while `web_search`
catches it and DEGRADES to no-evidence — a single retrieval hiccup must never crash a whole query
(the bug that produced the empty answers). All knobs come from the config `retrieval` block.

Serper key comes from SERPER_API_KEY in the environment, never hardcoded.
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


def build_retriever(cfg: Optional[dict]) -> Optional[Retriever]:
    """Build a cached plain-web retriever from the config's `retrieval` block (or None)."""
    if not cfg:
        return None
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
