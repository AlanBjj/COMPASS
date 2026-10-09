"""Aggregate run metrics: Hallucination Rate (%, lower better), Accuracy (0-100, higher
better), plus cost/efficiency — avg tokens per query (BY ROLE, so the open-source controller's
tokens are counted separately) and avg latency. Invalid judgments are excluded, not defaulted."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .judge import Judgment


def aggregate(
    judgments: Sequence[Judgment],
    *,
    n_total: Optional[int] = None,
    token_snapshot: Optional[Dict] = None,
    latencies: Optional[List[float]] = None,
) -> dict:
    n_total = n_total if n_total is not None else len(judgments)
    valid = [j for j in judgments if j.valid]
    n = len(valid)

    out: dict = {
        "n_total": n_total,
        "n_scored": n,
        "n_invalid": len(judgments) - n,
        "hallucination_rate": round(100.0 * sum(j.hallucination for j in valid) / n, 2) if n else None,
        "accuracy": round(sum(j.accuracy for j in valid) / n, 2) if n else None,
    }
    if token_snapshot is not None:
        out["tokens"] = token_snapshot
        total = token_snapshot.get("total_tokens", 0)
        out["avg_tokens_per_query"] = round(total / n_total, 1) if n_total else None
        # per-role avg, so controller vs backbone cost is visible
        out["avg_tokens_per_query_by_role"] = {
            role: round(u["total_tokens"] / n_total, 1) if n_total else None
            for role, u in token_snapshot.get("by_role", {}).items()
        }
    if latencies:
        out["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)
    return out
