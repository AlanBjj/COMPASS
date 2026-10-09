"""LLM access layer: vLLM OpenAI-compatible client with per-role token accounting,
and the prompt-file loader. Everything that touches a model goes through here."""

from .client import LLMClient, TokenLedger, RoleUsage, build_messages
from .prompts import load_prompt, render

__all__ = [
    "LLMClient",
    "TokenLedger",
    "RoleUsage",
    "build_messages",
    "load_prompt",
    "render",
]
