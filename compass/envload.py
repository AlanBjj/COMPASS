"""Minimal .env loader (zero dependencies — no python-dotenv needed).

Reads KEY=VALUE lines from a .env file into os.environ WITHOUT overriding variables already
set in the real environment (real env > .env). Keys we use: OPENAI_BASE_URL and OPENAI_API_KEY
(LLM judge), SERPER_API_KEY (FreshQA Fact retrieval). Never commit .env (see .gitignore).
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)  # real env wins over .env
