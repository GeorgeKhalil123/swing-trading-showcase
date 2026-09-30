"""Responsibility: the backend contract every LLM call in this project goes through.

One call = one agent, one ticker, one point in time, one schema. The `Backend` protocol is
deliberately tiny (text in, text and usage out): everything that makes an agent an agent - its
prompt, its schema, its retry, its cache entry - lives in `swingcore.llm.runtime`, so a new backend
cannot accidentally acquire tools, memory or a second turn.

The private system ships three backends (a local CLI, the vendor SDK and `fake`). Only `fake`, the
one that replays recorded replies from disk, is part of this showcase.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


class LLMError(RuntimeError):
    """A backend could not produce a reply (not installed, timed out, refused, no recording)."""


class LLMTimeout(LLMError):  # noqa: N818 - named for the condition, like LLMError above
    """The per-call timeout elapsed."""


@dataclass(frozen=True)
class AgentCall:
    """One agent's single turn. `ticker` is "MARKET" for the market-wide agents.

    `effort` is how much reasoning the turn is allowed to spend; empty means "whatever the backend
    defaults to".
    """

    agent: str
    ticker: str
    as_of: str
    system: str
    prompt: str
    model: str
    effort: str = ""
    timeout: float = 90.0
    should_stop: Callable[[], bool] = field(default=lambda: False, compare=False)


@dataclass(frozen=True)
class LLMReply:
    """What a backend returns. Token counts and cost are 0 when the backend cannot report them."""

    text: str
    model: str
    backend: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Backend(Protocol):
    """Text in, text out. No tools, no state, no second turn."""

    name: str

    def complete(self, call: AgentCall) -> LLMReply: ...
