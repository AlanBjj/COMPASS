"""Dataset base types and IO helpers.

A loader turns a raw benchmark file into a list of `Example`. `gold` holds the raw,
dataset-specific reference fields the judge needs (TruthfulQA has correct/incorrect/best;
GSM8K has the numeric answer; StrategyQA has yes/no; FreshQA has acceptable answers) — the
eval/judge module adapts to `example.dataset`. This replaces the legacy per-dataset
BaseDataset.evaluate_answer (brittle substring matching) with a clean, judge-driven design.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Example:
    id: str
    question: str
    gold: Dict[str, object] = field(default_factory=dict)
    dataset: str = ""


def read_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def read_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample(items: list, n: Optional[int]) -> list:
    return items[:n] if n else items


def split_examples(examples: list, split: str = "all", dev_size: int = 100, seed: int = 0) -> list:
    """Deterministic dev/test split for tune-on-dev / test-once discipline.

    Shuffles a COPY of `examples` with random.Random(seed), then returns the first
    `dev_size` items for split=="dev", the rest for split=="test", or all items for
    split=="all" (default). Same seed -> same split, and dev/test are disjoint by
    construction. FreshQA has an official DEV/TEST split and is handled in the registry,
    not here.
    """
    if split == "all":
        return list(examples)
    if split not in ("dev", "test"):
        raise ValueError(f"unknown split: {split} (expected dev/test/all)")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:dev_size] if split == "dev" else shuffled[dev_size:]
