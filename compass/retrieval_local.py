"""Local frozen-snapshot retrieval: BM25 over archived web-search snippets.

The corpus is the union of the web-search snippets collected for the benchmark questions in June 2026
(three source-typed endpoints + plain web search), deduplicated and frozen in
data/snapshot_corpus/corpus.jsonl (built by scripts/build_snapshot_corpus.py). Every method
retrieves from the same corpus, so retrieval is free, deterministic and identical across methods.

Source types are kept: "academic" (/scholar), "news" (/news) and "official" (/search; the
baselines' plain web results came from the same endpoint and are merged into it). COMPASS's
Multi-Source Weighted RAG queries each source index separately and keeps its credibility weights;
baselines query the union index ("all").

QueryRewriteRetriever wraps any retriever: the controller first rewrites the
question into a few short keyword queries (BM25 matches words, not meaning), each query is
searched, and the hits are merged per source with de-duplication.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .llm import render
from .llm.client import CONTROLLER, LLMClient
from .solvers.rag_solver import Evidence, Retriever
from .snapshot_index import GROUPS, _GROUP_OF, SnapshotIndex, get_index  # noqa: F401 (re-exported)


def _redistribute(evidences: List[Evidence]) -> List[Evidence]:
    """Same rule as SerperRetriever: an empty source's weight is spread over the non-empty ones."""
    non_empty = [e for e in evidences if e.snippets]
    if non_empty and len(non_empty) < len(evidences):
        total = sum(e.weight for e in non_empty)
        if total > 0:
            for e in evidences:
                e.weight = e.weight / total if e.snippets else 0.0
    return evidences


class LocalSourceRetriever(Retriever):
    """COMPASS Multi-Source Weighted RAG over the local snapshot: one BM25 search per source type."""

    def __init__(self, source_weights: Dict[str, float], corpus_path: str, index_dir: str,
                 per_source_k: int = 5) -> None:
        self.source_weights = source_weights
        self.per_source_k = per_source_k
        self.idx = get_index(corpus_path, index_dir)

    async def search(self, query: str) -> List[Evidence]:
        evidences = []
        for src, w in self.source_weights.items():
            hits = self.idx.search(query, _GROUP_OF.get(src, "official"), self.per_source_k)
            evidences.append(Evidence(src, w, [h["text"] for h in hits], [h.get("date", "") for h in hits]))
        return _redistribute(evidences)


_LINE_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


class QueryRewriteRetriever(Retriever):
    """Rewrite the question into up to n keyword queries (controller call), search each, merge.

    The original question is always searched too, so rewriting can only add evidence. Per source,
    hits are interleaved across queries (round-robin), de-duplicated, and capped at per_source_k.
    """

    def __init__(self, inner: Retriever, client: LLMClient, n_queries: int = 3, per_source_k: int = 6,
                 temperature: float = 0.0, max_tokens: int = 128) -> None:
        self.inner = inner
        self.client = client
        self.n_queries = n_queries
        self.per_source_k = per_source_k
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def _rewrite(self, question: str) -> List[str]:
        prompt = render("search_queries", question=question, n=str(self.n_queries))
        try:
            text = await self.client.chat_text(prompt, CONTROLLER, temperature=self.temperature,
                                               max_tokens=self.max_tokens)
        except Exception:  # noqa: BLE001 - rewriting is an optional boost; fall back to the question
            return []
        queries = []
        for line in text.splitlines():
            q = _LINE_PREFIX.sub("", line).strip().strip('"')
            if q and q.lower() not in {x.lower() for x in queries} and len(q) <= 200:
                queries.append(q)
        return queries[: self.n_queries]

    async def search(self, query: str) -> List[Evidence]:
        queries = [query] + [q for q in await self._rewrite(query) if q.lower() != query.lower()]
        per_query = [await self.inner.search(q) for q in queries]
        # Merge by source name and restore the configured weights: the inner retriever may have
        # zeroed a source that was empty for one query but is filled by another.
        weights = dict(getattr(self.inner, "source_weights", None) or {})
        order = [e.source for e in per_query[0]]
        merged: List[Evidence] = []
        for src in order:
            lists = [e for ev in per_query for e in ev if e.source == src]
            seen, snippets, dates = set(), [], []
            depth = max((len(e.snippets) for e in lists), default=0)
            for rank in range(depth):
                for e in lists:
                    if rank < len(e.snippets) and e.snippets[rank] not in seen:
                        seen.add(e.snippets[rank])
                        snippets.append(e.snippets[rank])
                        dates.append((e.dates or [""] * len(e.snippets))[rank])
            merged.append(Evidence(src, float(weights.get(src, lists[0].weight if lists else 0.0)),
                                   snippets[: self.per_source_k], dates[: self.per_source_k]))
        return _redistribute(merged)
