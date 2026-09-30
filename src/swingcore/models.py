"""Responsibility: the typed records a run is made of - failures, gaps, usage lines and stages.

TypedDicts rather than classes because every one of them is written straight into the run manifest
as JSON and validated there against `schemas/run_manifest.json`; the types exist for mypy, the
schema exists for everything that reads the file later.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

MARKET_KEY = "MARKET"
StageStatus = Literal["ran", "not_run", "failed"]


class AgentFailure(TypedDict):
    agent: str
    ticker: str
    stage: str
    error: str


class DataGap(TypedDict):
    """A named absence. Every blank the report would render has one of these explaining it."""

    ticker: str
    field: str
    detail: str


class AgentUsage(TypedDict):
    """Per-agent cost line for the run manifest. Filled by the LLM runtime, zeros on a dry run.

    Two money columns, because a replay is not free research: `cost_usd` is what the answers in
    this manifest cost when they were produced (a cached call restores the tokens and cost of the
    call that filled the cache), while `spent_usd` is what *this* run paid - zero for every call
    served from the cache.
    """

    agent: str
    model: str
    backend: str
    calls: int
    cached_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    spent_usd: float


class RunStage(TypedDict):
    """One pipeline stage and whether this run actually executed it.

    Without this list a reader could not tell an empty result ("the agent passed on everything")
    from one that was never asked ("the agent did not run").
    """

    stage: str
    status: StageStatus
    note: str


class TickerMeta(TypedDict):
    sector: str | None
    last: NotRequired[float]
