"""Async client for the vLLM OpenAI-compatible endpoint, with per-role token accounting.

COMPASS serves BOTH the controller (gate, decomposition, sub-answer scoring) and the
backbone (answering, fusion) from the SAME open-source model via vLLM. Cost reporting counts the controller's tokens separately from the
backbone's. Every call is therefore tagged with a `role`, and token usage is
tallied per role from the API's own `usage` field (exact, no client-side tokenizer needed).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openai import AsyncOpenAI

# Roles we bill separately. "controller" = gate/decompose/scoring (the components that used
# the controller); "backbone" = answering/fusion; "judge" = automated HR/Acc evaluator.
Role = str
CONTROLLER: Role = "controller"
BACKBONE: Role = "backbone"
JUDGE: Role = "judge"


@dataclass
class RoleUsage:
    """Accumulated usage for one role."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> Dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class TokenLedger:
    """Per-role token tally for one run. Serialized into each run's results so cost can be
    reported with the controller's tokens included (not just the backbone's)."""

    by_role: Dict[Role, RoleUsage] = field(default_factory=dict)

    def record(self, role: Role, prompt_tokens: int, completion_tokens: int) -> None:
        usage = self.by_role.setdefault(role, RoleUsage())
        usage.calls += 1
        usage.prompt_tokens += int(prompt_tokens or 0)
        usage.completion_tokens += int(completion_tokens or 0)

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.by_role.values())

    def snapshot(self) -> Dict[str, object]:
        """JSON-serializable view for results logging."""
        roles = {role: usage.as_dict() for role, usage in sorted(self.by_role.items())}
        return {
            "by_role": roles,
            "total_calls": sum(u.calls for u in self.by_role.values()),
            "total_tokens": self.total_tokens,
        }


def build_messages(user: str, system: str = "You are a helpful assistant.") -> List[Dict[str, str]]:
    """Standard chat-format messages. Kept identical in spirit to the legacy
    construct_input_message so prompts port over unchanged."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class LLMError(RuntimeError):
    """Raised when a chat call fails after exhausting retries."""


class LLMClient:
    """Thin async wrapper over a vLLM OpenAI-compatible endpoint.

    The client owns the connection, a concurrency limiter, and the token ledger. It does
    NOT know the semantics of a role beyond billing — callers pass `role` and the
    sampling params (temperature / max_tokens), which they read from configs/.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",  # vLLM ignores it but the OpenAI SDK requires a value
        default_temperature: float = 0.2,
        default_max_tokens: int = 2048,
        concurrency: int = 8,
        retries: int = 3,
        retry_backoff: float = 2.0,
        ledger: Optional[TokenLedger] = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.retries = retries
        self.retry_backoff = retry_backoff
        # Some relays (e.g. zovelox) return 403 for any request whose User-Agent contains
        # "OpenAI" (the SDK's default UA). Override it so the judge endpoint accepts calls;
        # harmless for vLLM, which ignores the UA.
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key,
                                   default_headers={"User-Agent": "compass-client/1.0"})
        self._sem = asyncio.Semaphore(concurrency)
        self.ledger = ledger if ledger is not None else TokenLedger()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        role: Role,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Single chat completion. Records usage under `role`. Raises LLMError on failure."""
        temperature = self.default_temperature if temperature is None else temperature
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens

        async with self._sem:
            last_err: Optional[Exception] = None
            for attempt in range(self.retries):
                try:
                    resp = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    usage = getattr(resp, "usage", None)
                    if usage is not None:
                        self.ledger.record(
                            role,
                            getattr(usage, "prompt_tokens", 0),
                            getattr(usage, "completion_tokens", 0),
                        )
                    return (resp.choices[0].message.content or "").strip()
                except Exception as err:  # noqa: BLE001 - surface as LLMError after retries
                    last_err = err
                    if attempt < self.retries - 1:
                        await asyncio.sleep(self.retry_backoff * (attempt + 1))
            raise LLMError(f"chat failed after {self.retries} attempts: {last_err}")

    async def chat_text(
        self,
        user: str,
        role: Role,
        *,
        system: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Convenience: wrap a user string into messages and call chat()."""
        return await self.chat(
            build_messages(user, system=system),
            role,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def chat_many(
        self,
        prompts: List[str],
        role: Role,
        *,
        system: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Run many user prompts concurrently (bounded by the client's semaphore).
        Failures propagate as LLMError for the whole batch — callers needing per-item
        fault tolerance should gather chat() calls themselves with return_exceptions."""
        tasks = [
            self.chat_text(
                p, role, system=system, temperature=temperature, max_tokens=max_tokens
            )
            for p in prompts
        ]
        return await asyncio.gather(*tasks)
