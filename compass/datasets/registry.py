"""Dataset registry — name -> loader."""

from __future__ import annotations

from typing import List, Optional

from .base import Example, split_examples
from .freshqa import load_freshqa
from .gsm8k import load_gsm8k
from .strategyqa import load_strategyqa
from .truthfulqa import load_truthfulqa

LOADERS = {
    "truthfulqa": load_truthfulqa,
    "gsm8k": load_gsm8k,
    "freshqa": load_freshqa,
    "strategyqa": load_strategyqa,
}


def _freshqa_split(examples: List[Example], split: str) -> List[Example]:
    """FreshQA has an official DEV/TEST split carried in gold["split"]."""
    if split == "all":
        return list(examples)
    want = split.upper()  # "DEV" / "TEST"
    return [ex for ex in examples if (ex.gold.get("split") or "").upper() == want]


def load_dataset(
    name: str,
    path: str,
    sample_size: Optional[int] = None,
    split: str = "all",
    dev_size: int = 100,
    seed: int = 0,
) -> List[Example]:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset: {name} (known: {sorted(LOADERS)})")
    # Load ALL examples, apply the dev/test split, THEN truncate to sample_size — so
    # --sample-size N takes the first N of the chosen split (tune on dev, test once).
    examples = LOADERS[name](path)
    if name == "freshqa":
        examples = _freshqa_split(examples, split)  # official split
    else:
        examples = split_examples(examples, split, dev_size=dev_size, seed=seed)
    if sample_size:
        examples = examples[:sample_size]
    return examples
