"""Async client for the vLLM OpenAI-compatible endpoint, with per-role token accounting.

COMPASS serves BOTH the controller (gate, decomposition, sub-answer scoring) and the
backbone (answering, fusion) from the SAME open-source model via vLLM. Cost reporting counts the controller's tokens separately from the
backbone's. Every call is therefore tagged with a `role`, and token usage is
tallied per role from the API's own `usage` field (exact, no client-side tokenizer needed).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

# Qwen3 "thinking" models emit a leading <think>...</think> chain-of-thought block before the
# real answer. For the controller (gate scoring / decomposition / type routing) this corrupts
# the structured parsers, so the controller client strips it (in addition to disabling thinking
# server-side via chat_template_kwargs). Matches a single leading think block, DOTALL.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove a leading <think>...</think> block (Qwen3 reasoning trace) from a completion.
    No-op when absent, so it is safe to apply unconditionally to thinking-model output."""
    return _THINK_BLOCK.sub("", text or "")

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

    def restore(self, snapshot: Optional[Dict[str, object]]) -> None:
        """Add a checkpoint's snapshot() to the tally, so a resumed run also counts the tokens
        spent before the restart."""
        for role, u in ((snapshot or {}).get("by_role") or {}).items():
            usage = self.by_role.setdefault(role, RoleUsage())
            usage.calls += int(u.get("calls", 0))
            usage.prompt_tokens += int(u.get("prompt_tokens", 0))
            usage.completion_tokens += int(u.get("completion_tokens", 0))

    def snapshot(self) -> Dict[str, object]:
        """JSON-serializable view for results logging."""
        roles = {role: usage.as_dict() for role, usage in sorted(self.by_role.items())}
        return {
            "by_role": roles,
            "total_calls": sum(u.calls for u in self.by_role.values()),
            "total_tokens": self.total_tokens,
        }


def build_messages(user: str, system: str = "You are a helpful assistant.") -> List[Dict[str, str]]:
    """Standard chat-format messages: one system message and one user message."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def date_note_text(current_date: Optional[str]) -> Optional[str]:
    """The benchmark-date line every method receives on a time-sensitive benchmark (run.current_date
    in the config), or None. Shared by COMPASS and the baselines so the wording is identical."""
    return f"Today's date is {current_date}." if current_date else None


def _with_system_note(messages: List[Dict[str, str]], note: str) -> List[Dict[str, str]]:
    """Copy of `messages` with `note` prepended to the system message (one is added if absent)."""
    if messages and messages[0].get("role") == "system":
        first = dict(messages[0])
        first["content"] = f"{note}\n{first.get('content', '')}".strip()
        return [first] + list(messages[1:])
    return [{"role": "system", "content": note}] + list(messages)


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
        extra_body: Optional[dict] = None,
        strip_thinking: bool = False,
        request_timeout: float = 300.0,
        keepalive: bool = True,
        system_note: Optional[str] = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.retries = retries
        self.retry_backoff = retry_backoff
        # extra_body is forwarded verbatim on every completion call. For Qwen3 thinking models
        # we pass {"chat_template_kwargs": {"enable_thinking": False}} to suppress <think> output
        # server-side; strip_thinking additionally removes any residual <think> block client-side.
        self.extra_body = extra_body
        self.strip_thinking = strip_thinking
        # Evaluation setting shared by every method on a time-sensitive benchmark (e.g. FreshQA's
        # "Today's date is April 2026."): prepended to the system message of EVERY call made through
        # this client, so no component (or baseline) can answer "this year" questions without it.
        self.system_note = system_note
        # Some OpenAI-compatible relays reject the SDK's default User-Agent, so it is overridden;
        # vLLM ignores the UA.
        # Bound each request so a momentarily hung vLLM/judge call fails fast and hits our own
        # retry loop instead of blocking a whole run. max_retries=0 leaves retry policy to us.
        # keepalive=False opens a fresh connection per request: on the local judge, requests sent on
        # a pooled connection were occasionally lost (no response, no server log) and each loss cost
        # a full request_timeout before the retry.
        http_client = None
        if not keepalive:
            http_client = httpx.AsyncClient(
                timeout=request_timeout,
                limits=httpx.Limits(max_keepalive_connections=0),
            )
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key,
                                   default_headers={"User-Agent": "compass-client/1.0"},
                                   timeout=request_timeout, max_retries=0,
                                   http_client=http_client)
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
        resp = await self._complete(messages, role, temperature, max_tokens)
        content = resp.choices[0].message.content or ""
        if self.strip_thinking:
            content = strip_think(content)
        return content.strip()

    async def _complete(
        self,
        messages: List[Dict[str, str]],
        role: Role,
        temperature: Optional[float],
        max_tokens: Optional[int],
        extra: Optional[dict] = None,
    ):
        """One completion with the client's retry policy and token accounting; returns the raw
        response. `extra` adds request fields (e.g. logprobs)."""
        temperature = self.default_temperature if temperature is None else temperature
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        if self.system_note:
            messages = _with_system_note(messages, self.system_note)

        async with self._sem:
            last_err: Optional[Exception] = None
            for attempt in range(self.retries):
                try:
                    kwargs = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if extra:
                        kwargs.update(extra)
                    if self.extra_body is not None:
                        kwargs["extra_body"] = self.extra_body
                    resp = await self._client.chat.completions.create(**kwargs)
                    usage = getattr(resp, "usage", None)
                    if usage is not None:
                        self.ledger.record(
                            role,
                            getattr(usage, "prompt_tokens", 0),
                            getattr(usage, "completion_tokens", 0),
                        )
                    resp.choices[0].message  # a malformed response is retried, as before _complete()
                    return resp
                except Exception as err:  # noqa: BLE001 - surface as LLMError after retries
                    last_err = err
                    if attempt < self.retries - 1:
                        await asyncio.sleep(self.retry_backoff * (attempt + 1))
            raise LLMError(f"chat failed after {self.retries} attempts: {last_err}")

    async def chat_text_logprobs(
        self,
        user: str,
        role: Role,
        *,
        system: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_logprobs: int = 0,
    ) -> Tuple[str, List[Tuple[str, float, List[float]]]]:
        """Like chat_text(), plus per generated token (token, logprob, [top-k logprobs, best first]).
        Used by baselines that gate retrieval on generation uncertainty (TARG, DTR, ETC). vLLM
        returns at most 20 alternatives per position, so full-vocabulary quantities (entropy) are
        approximated from the top-k."""
        extra = {"logprobs": True}
        if top_logprobs:
            extra["top_logprobs"] = int(top_logprobs)
        resp = await self._complete(build_messages(user, system=system), role, temperature, max_tokens, extra)
        choice = resp.choices[0]
        tokens: List[Tuple[str, float, List[float]]] = []
        for t in (getattr(choice, "logprobs", None) and choice.logprobs.content) or []:
            alts = sorted((a.logprob for a in (t.top_logprobs or [])), reverse=True)
            tokens.append((t.token, t.logprob, alts))
        content = choice.message.content or ""
        if self.strip_thinking:
            content = strip_think(content)
        return content.strip(), tokens

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


class DualLLMClient:
    """Routes calls to one of two underlying LLMClients by role, when the controller (gate scoring /
    decomposition / sub-answer scoring / CPC legs — role=CONTROLLER) is served by a different
    model/endpoint than the backbone (answering / RAG synthesis / fusion / verify — role=BACKBONE).

    It exposes the exact LLMClient call surface (chat / chat_text / chat_many) so every component
    keeps calling `client.<method>(..., ROLE, ...)` unchanged — only the dispatch differs. Both
    sub-clients share ONE TokenLedger, so per-role token accounting (controller vs backbone) is
    preserved and `ledger.snapshot()` keeps working in run.py. When controller and backbone are the
    same model+endpoint, build_compass uses a single LLMClient instead and behaviour is identical to
    before this change.
    """

    def __init__(self, controller: LLMClient, backbone: LLMClient, ledger: TokenLedger) -> None:
        self.controller = controller
        self.backbone = backbone
        self.ledger = ledger  # shared with both sub-clients; what run.py snapshots

    def _for(self, role: Role) -> LLMClient:
        # CONTROLLER -> controller client; everything else (BACKBONE, and JUDGE if ever routed
        # here, which it is not — the judge has its own client) -> backbone client.
        return self.controller if role == CONTROLLER else self.backbone

    async def chat(self, messages, role: Role, **kw) -> str:
        return await self._for(role).chat(messages, role, **kw)

    async def chat_text(self, user: str, role: Role, **kw) -> str:
        return await self._for(role).chat_text(user, role, **kw)

    async def chat_many(self, prompts: List[str], role: Role, **kw) -> List[str]:
        return await self._for(role).chat_many(prompts, role, **kw)

    async def chat_text_logprobs(self, user: str, role: Role, **kw):
        return await self._for(role).chat_text_logprobs(user, role, **kw)
