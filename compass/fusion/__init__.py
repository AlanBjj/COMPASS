"""Steps 4 & 5 — Sub-answer Quality Scoring and Confidence-Aware Fusion."""

from .fusion import fuse
from .scoring import (
    ScoredSubAnswer,
    compute_weights,
    is_low_quality,
    parse_scores,
    score_subanswers,
)

__all__ = [
    "ScoredSubAnswer",
    "score_subanswers",
    "compute_weights",
    "is_low_quality",
    "parse_scores",
    "fuse",
]
