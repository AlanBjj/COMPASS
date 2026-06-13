"""Step 3 (Fact) — Multi-Source Weighted RAG (paper §III-C, Eq. 5).

Paper: parallel extraction from three source types (academic / news / official), credibility
weighting 0.4 / 0.3 / 0.3, then conflict resolution by weighted synthesis. Two fixes vs legacy:
  1. legacy `resolve_conflicts` just returned the single highest-weight snippet; here ALL
     weighted evidence is synthesized by the backbone (true weighted synthesis).
  2. live web search is wrapped in a snapshot cache so runs are reproducible:
     every retrieval's raw evidence is cached and can be replayed/published.
Serper API key comes from the SERPER_API_KEY env var, never hardcoded. Source weights come from
config. Only Fact-typed sub-queries reach this solver.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from ..llm import render
from ..llm.client import BACKBONE, LLMClient


@dataclass
class Evidence:
    source: str           # "academic" / "news" / "official"
    weight: float         # credibility weight from config
    snippets: List[str]
    # Per-snippet publication dates (parallel to `snippets`, "" when unknown), used to order
    # snippets most-recent-first for freshness. Optional so old cache snapshots still load.
    dates: Optional[List[str]] = None

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "weight": self.weight,
            "snippets": self.snippets,
            "dates": self.dates if self.dates is not None else ["" for _ in self.snippets],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        snippets = list(d["snippets"])
        dates = list(d["dates"]) if d.get("dates") is not None else None
        return cls(source=d["source"], weight=float(d["weight"]), snippets=snippets, dates=dates)


class Retriever(ABC):
    @abstractmethod
    async def search(self, query: str) -> List[Evidence]:
        ...


class SerperRetriever(Retriever):
    """Live web retrieval over three source types via the Serper API."""

    # Each source type maps to a real Serper endpoint (no narrow site: filtering).
    SOURCE_ENDPOINTS = {
        "academic": "scholar",
        "news": "news",
        "official": "search",
    }

    def __init__(
        self,
        source_weights: Dict[str, float],
        *,
        api_key: Optional[str] = None,
        num_results: int = 5,
        timeout: float = 10.0,
    ) -> None:
        self.source_weights = source_weights
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        self.num_results = num_results
        self.timeout = timeout

    async def search(self, query: str) -> List[Evidence]:
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY not set (export it; never hardcode)")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._one(client, src, query) for src in self.source_weights]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        evidences: List[Evidence] = []
        for src, res in zip(self.source_weights, results):
            pairs = [] if isinstance(res, Exception) else res
            snippets = [s for s, _ in pairs]
            dates = [d for _, d in pairs]
            evidences.append(Evidence(src, self.source_weights[src], snippets, dates))

        # Redistribute the weight of any EMPTY source proportionally across the non-empty
        # ones, so non-empty weights renormalize to ~1 and the official /search baseline
        # carries when scholar/news come back empty. If ALL are empty, leave weights as-is
        # (solve_rag will fall back to parametric knowledge).
        non_empty = [e for e in evidences if e.snippets]
        if non_empty and len(non_empty) < len(evidences):
            total_non_empty = sum(e.weight for e in non_empty)
            if total_non_empty > 0:
                for e in evidences:
                    e.weight = e.weight / total_non_empty if e.snippets else 0.0
        return evidences

    async def _one(self, client: httpx.AsyncClient, source: str, query: str) -> List[tuple]:
        endpoint = self.SOURCE_ENDPOINTS[source]
        resp = await client.post(
            f"https://google.serper.dev/{endpoint}",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json={"q": query, "num": self.num_results, "gl": "us", "hl": "en"},
        )
        resp.raise_for_status()
        data = resp.json()
        # /scholar and /search return an "organic" list; /news returns a "news" list.
        items = data.get("organic") if "organic" in data else data.get("news", [])
        items = items or []
        # Return (snippet, date) pairs. Serper /news carries a "date" per item; /search and
        # /scholar expose dates inconsistently (date / publishedDate / year), so capture
        # whatever is present and let _date_sort_key tolerate the missing/loose ones.
        pairs: List[tuple] = []
        for item in items:
            snippet = (item.get("snippet") or item.get("title") or "").strip()
            if not snippet:
                continue
            date = str(
                item.get("date") or item.get("publishedDate") or item.get("year") or ""
            ).strip()
            pairs.append((snippet, date))
        return pairs


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
        path = self._path(key)
        path.write_text(
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


def _strip_premise_scaffold(text: str) -> str:
    """The RAG solver prompt asks for a "Quote: ... / PremiseCheck: ... / Premise: true|false /
    Answer: ..." reply (the quote line forces grounding; the premise lines force an explicit
    false-premise check on a small model). Downstream consumers want only the answer text, so drop
    the scaffold and the "Answer:" label. We cut at the LAST "Answer:" so any "Answer:" mentioned
    inside the scaffold is ignored. Defensive: if the format is absent, return the text unchanged."""
    stripped = text.strip()
    lower = stripped.lower()
    marker = "\nanswer:"
    idx = lower.rfind(marker)
    # When the reply opens with the scaffold, return everything after the final "Answer:" label.
    if (
        lower.startswith("quote:")
        or lower.startswith("premisecheck:")
        or lower.startswith("premise:")
    ) and idx != -1:
        return stripped[idx + len(marker):].strip()
    # Tolerate a leading "Answer:" with no scaffold.
    if lower.startswith("answer:"):
        return stripped[len("answer:"):].strip()
    return stripped


def _recency_rank(date: str) -> float:
    """Map a Serper date string to a recency score in DAYS-SINCE-EPOCH (HIGHER = more recent),
    for ordering only. Everything lands on one scale so relative dates ("3 days ago") and
    absolute dates ("2024-01-01") compare correctly. Handles Serper's relative dates, 4-digit
    years, and ISO-ish dates. Unknown/empty dates rank lowest so dated snippets lead. This is a
    coarse ordering key, not a precise timestamp."""
    import re
    from datetime import datetime

    d = (date or "").strip().lower()
    if not d:
        return float("-inf")
    now_days = datetime.now().timestamp() / 86400.0
    # Relative: "N unit(s) ago" -> now minus the age, so it sits on the epoch-day scale.
    m = re.match(r"(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*ago", d)
    if m:
        n = int(m.group(1))
        unit_days = {
            "second": 1 / 86400, "minute": 1 / 1440, "hour": 1 / 24,
            "day": 1, "week": 7, "month": 30, "year": 365,
        }[m.group(2)]
        return now_days - n * unit_days
    if d in ("today", "just now"):
        return now_days
    if d == "yesterday":
        return now_days - 1.0
    # Absolute date or bare year -> days since epoch (so newer = larger).
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%Y/%m/%d", "%d %b %Y", "%Y"):
        try:
            dt = datetime.strptime(date.strip(), fmt)
            return dt.timestamp() / 86400.0
        except (ValueError, TypeError):
            continue
    m = re.search(r"(19|20)\d{2}", d)  # fall back to any 4-digit year in the string
    if m:
        try:
            return datetime.strptime(m.group(0), "%Y").timestamp() / 86400.0
        except ValueError:
            pass
    return float("-inf")


def format_evidence(evidences: List[Evidence]) -> str:
    """Render weighted evidence ordered for FRESHNESS: highest-weight source first (so
    official/news lead and low-weight academic comes last), then most-recent snippet first
    within each source. The model reads the most credible, most current evidence at the top."""
    lines: List[str] = []
    for ev in sorted(evidences, key=lambda e: e.weight, reverse=True):
        dates = ev.dates if ev.dates is not None else ["" for _ in ev.snippets]
        # Pair each snippet with its date and sort most-recent-first; stable for ties/unknowns.
        paired = list(zip(ev.snippets, dates + [""] * (len(ev.snippets) - len(dates))))
        paired.sort(key=lambda sd: _recency_rank(sd[1]), reverse=True)
        for snippet, date in paired:
            tag = f"[{ev.source} | {ev.weight:.2f}]"
            if date:
                tag = f"[{ev.source} | {ev.weight:.2f} | {date}]"
            lines.append(f"{tag} {snippet}")
    return "\n".join(lines)


async def solve_rag(
    question: str,
    main_q: str,
    client: LLMClient,
    retriever: Retriever,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Retrieve weighted evidence, then synthesize a backbone answer over all of it."""
    evidences = await retriever.search(question)
    formatted = format_evidence(evidences)
    if not formatted.strip():
        # No evidence: fall back to parametric knowledge (legacy behavior).
        return await client.chat_text(
            f"Answer this question based on your knowledge:\n{question}",
            BACKBONE,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    prompt = render("solver_rag", question=question, main_q=main_q, evidence=formatted)
    raw = await client.chat_text(
        prompt, BACKBONE, temperature=temperature, max_tokens=max_tokens
    )
    return _strip_premise_scaffold(raw)
