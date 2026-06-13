"""Loader for prompt files under prompts/.

Every prompt is a file (prompts/<name>.txt), never inlined in code — this is a project
convention and the files double as the paper's supplementary appendix.

Placeholders use ${name} (string.Template), NOT Python str.format's {name}. Several
prompts ask the model to emit JSON with literal { } (e.g. the gate decision and the
sub-answer scoring format), so {}-based formatting would force brittle {{ }} escaping.
With ${name} the literal braces stay untouched.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Dict

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

_cache: Dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Return the raw text of prompts/<name>.txt (cached)."""
    if name not in _cache:
        path = PROMPTS_DIR / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"prompt not found: {path}")
        _cache[name] = path.read_text(encoding="utf-8")
    return _cache[name]


def render(name: str, **kwargs: object) -> str:
    """Load prompts/<name>.txt and substitute ${placeholders}. Unknown ${...} are left
    as-is (safe_substitute) so stray '$' in prompt text never crashes a run."""
    return Template(load_prompt(name)).safe_substitute(**kwargs)
