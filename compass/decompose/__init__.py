"""Step 2 — Sub-queries Decomposition: complex query -> 2-4 typed sub-queries."""

from .decompose import (
    FACT,
    GENERAL,
    LOGIC,
    MATH,
    SubQuery,
    decompose,
    normalize_type,
    parse_decomposition,
)

__all__ = [
    "SubQuery",
    "decompose",
    "parse_decomposition",
    "normalize_type",
    "MATH",
    "LOGIC",
    "FACT",
    "GENERAL",
]
