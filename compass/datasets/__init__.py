"""Dataset loaders for the four benchmarks (TruthfulQA / FreshQA / GSM8K / StrategyQA)."""

from .base import Example
from .registry import LOADERS, load_dataset

__all__ = ["Example", "load_dataset", "LOADERS"]
