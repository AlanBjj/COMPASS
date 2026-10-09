"""Evaluation: automated judge (HR + Accuracy), metric aggregation, and reporting."""

from .judge import Judgment, format_reference, judge_all, judge_one, parse_judgment
from .metrics import aggregate

__all__ = [
    "Judgment",
    "judge_one",
    "judge_all",
    "format_reference",
    "parse_judgment",
    "aggregate",
]
