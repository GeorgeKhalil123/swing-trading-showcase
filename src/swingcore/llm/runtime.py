"""Responsibility: turn a backend into an *agent* - cached, schema-validated, retried exactly once.

This is the only place an LLM reply becomes data the rest of the system may use. It enforces the
three rules that make an agent auditable:

* the reply must parse as JSON and satisfy the agent's JSON Schema (plus any caller-supplied check,
  e.g. "every number you quoted must be one we gave you"),
* one retry, with the validation error fed back verbatim, then the agent is marked failed,
* the result of a *model* call is cached by (backend, agent, ticker, as_of, sha256 of the input) in
  the SQLite payloads table, so re-running the same day costs nothing. The `fake` backend is never
  cached: a fixture read is already free, and a cached fixture would outlive the file it came from
  and replay an answer the repository no longer contains.

What identity the cache key has, and what noise it forgives, is `swingcore.llm.cache_key`. A miss
that is *not* a first call - the same agent, ticker and day under a digest that has since moved - is
counted as a stale miss, and the canonical input is stored next to the reply so two rows can be
diffed rather than guessed at.

Token counts and cost are accumulated per agent for the run manifest's usage table. A cached reply
carries the tokens and cost of the call that *produced* it; what this run actually paid is the
separate `spent_usd` column, which a cached call never increases. A cached row with no reply
metadata at all is treated as a *miss* and re-run: replaying it would put a model call in the cost
table at zero tokens and zero dollars.

Trimmed from the private runtime: the per-agent config loaders and the non-fake backend factory.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from swingcore.bitemporal_cache import Cache
from swingcore.llm.base import AgentCall, Backend, LLMError, LLMReply
from swingcore.llm.cache_key import canonical_payload, input_hash, record_stale_miss
from swingcore.models import AgentFailure, AgentUsage
from swingcore.schemas import validation_error

log = logging.getLogger(__name__)
CACHE_KIND = "agent"
RETRY_TEMPLATE = (
    "Your previous reply was rejected: {error}\n\n"
    "Reply again with the corrected JSON object only. Do not explain the correction, do not add "
    "any key the schema does not list, and do not change any number to one that was not supplied "
    "to you."
)
Check = Callable[[dict[str, Any]], str | None]


class AgentFailed(RuntimeError):  # noqa: N818 - the agent is 'marked as failed'
    """Raised after the single allowed retry, or when the backend itself failed."""

    def __init__(self, agent: str, ticker: str, stage: str, error: str) -> None:
        super().__init__(f"{agent} on {ticker} failed at {stage}: {error}")
        self.failure = AgentFailure(agent=agent, ticker=ticker, stage=stage, error=error)


@dataclass
class AgentResult:
    """One validated agent output plus what it cost to get it."""

    agent: str
    ticker: str
    data: dict[str, Any]
    reply: LLMReply
    cached: bool
    attempts: int


def extract_json(text: str) -> dict[str, Any]:
    """Parse the JSON object out of a reply, tolerating a ```json fence or a sentence around it."""
    body = text.strip()
    if body.startswith("```"):
        lines = [ln for ln in body.split("\n") if not ln.strip().startswith("```")]
        body = "\n".join(lines).strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"the reply contains no JSON object: {text.strip()[:300]!r}") from None
        try:
            parsed = json.loads(body[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"the reply is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"the reply is a {type(parsed).__name__}, not a JSON object")
    return parsed


@dataclass
class Runtime:
    """Runs agents against one backend, with one cache and one model table."""

    backend: Backend
    cache: Cache | None = None
    models: Mapping[str, str] = field(default_factory=dict)
    efforts: Mapping[str, str] = field(default_factory=dict)
    timeouts: Mapping[str, float] = field(default_factory=dict)
    default_model: str = ""
    default_effort: str = ""
    timeout: float = 90.0
    stale_misses: Counter[str] = field(default_factory=Counter, init=False)
    _usage: dict[str, AgentUsage] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def model_for(self, agent: str) -> str:
        model = self.models.get(agent) or self.default_model
        if not model:
            raise AgentFailed(agent, "", "config", f"no model configured for {agent}")
        return model

    def effort_for(self, agent: str) -> str:
        return self.efforts.get(agent) or self.default_effort

    def timeout_for(self, agent: str) -> float:
        """Wall clock for one turn: the agent's own timeout, else the default."""
        return float(self.timeouts.get(agent, self.timeout))

    # ---- usage ----------------------------------------------------------------
    def usage(self) -> list[AgentUsage]:
        """Per-agent usage for the run manifest, sorted by agent name.

        Sorted rather than in first-call order because the agents run in a thread pool: whichever
        one answers first is a race, and a run manifest that shuffles its own cost table between
        two identical runs is not a calibration record.
        """
        with self._lock:
            return [AgentUsage(**row) for row in sorted(self._usage.values(), key=lambda r: r["agent"])]

    def _record(self, agent: str, reply: LLMReply, cached: bool) -> None:
        with self._lock:
            row = self._usage.setdefault(
                agent,
                AgentUsage(
                    agent=agent,
                    model=reply.model,
                    backend=reply.backend,
                    calls=0,
                    cached_calls=0,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    spent_usd=0.0,
                ),
            )
            row["calls"] += 1
            # A cached hit still contributes the tokens and cost of the call that produced it: the
            # research was paid for once and the table has to keep saying so. Only `spent_usd`
            # distinguishes "what this run cost" from "what this research cost".
            row["input_tokens"] += reply.input_tokens
            row["output_tokens"] += reply.output_tokens
            row["cost_usd"] = round(row["cost_usd"] + reply.cost_usd, 6)
            if cached:
                row["cached_calls"] += 1
                return
            row["spent_usd"] = round(row["spent_usd"] + reply.cost_usd, 6)

    # ---- cache ----------------------------------------------------------------
    def _cache_key(self, agent: str, ticker: str, digest: str) -> str:
        # The backend is part of the identity: a recorded fixture reply from a `--dry-run` must
        # never be served back to a real run (or the reverse) just because the input matched.
        return f"{self.backend.name}|{agent}|{ticker}|{digest}"

    def _replayable(self) -> bool:
        """Only a backend that costs something is worth caching."""
        return self.cache is not None and self.backend.name != "fake"

    def _cached(
        self, agent: str, ticker: str, as_of: str, digest: str
    ) -> tuple[dict[str, Any], LLMReply] | None:
        if self.cache is None or not self._replayable():
            return None
        hit = self.cache.get_payload(CACHE_KIND, self._cache_key(agent, ticker, digest), as_of=as_of)
        if hit is None:
            return None
        stored, stored_as_of = hit.data, hit.as_of
        if stored_as_of[:10] != as_of[:10]:  # same input, different day: re-run rather than replay
            return None
        meta = stored.get("reply", {})
        if not meta:
            log.info("%s on %s: cached reply has no usage metadata; re-running it", agent, ticker)
            return None
        reply = LLMReply(
            text=json.dumps(stored["data"]),
            model=str(meta.get("model", "")),
            backend=str(meta.get("backend", self.backend.name)),
            input_tokens=int(meta.get("input_tokens", 0) or 0),
            output_tokens=int(meta.get("output_tokens", 0) or 0),
            cost_usd=float(meta.get("cost_usd", 0.0) or 0.0),
        )
        return stored["data"], reply

    def _store(self, as_of: str, digest: str, result: AgentResult, canonical: Any) -> None:
        """Write the answer, and the exact question it answered, so a same-day miss can be diffed."""
        if self.cache is None or not self._replayable():
            return
        self.cache.put_payload(
            CACHE_KIND,
            self._cache_key(result.agent, result.ticker, digest),
            {
                "data": result.data,
                "input": canonical,
                "input_digest": digest,
                "reply": {
                    "model": result.reply.model,
                    "backend": result.reply.backend,
                    "input_tokens": result.reply.input_tokens,
                    "output_tokens": result.reply.output_tokens,
                    "cost_usd": result.reply.cost_usd,
                },
            },
            source=result.reply.backend,
            as_of=as_of,
        )

    def _note_stale_miss(self, agent: str, ticker: str, as_of: str, digest: str) -> None:
        """Count a miss that is not a first call: the same question, asked earlier today."""
        if self.cache is None or not self._replayable():
            return
        key = self._cache_key(agent, ticker, digest)
        earlier = [
            other
            for other in self.cache.payload_keys_like(
                CACHE_KIND, self._cache_key(agent, ticker, ""), as_of[:10]
            )
            if other != key
        ]
        if not earlier:
            return
        with self._lock:
            self.stale_misses[agent] += 1
        record_stale_miss(agent, as_of[:10])
        log.info(
            "%s on %s: same-day input changed; previous digest %s",
            agent,
            ticker,
            earlier[-1].rsplit("|", 1)[-1],
        )

    # ---- the one public entry point -------------------------------------------
    def run(
        self,
        agent: str,
        ticker: str,
        as_of: str,
        system: str,
        payload: Any,
        schema: str,
        extra_check: Check | None = None,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> AgentResult:
        """Call `agent` once (twice at most), validate, cache, and return the parsed output."""
        model = self.model_for(agent)
        effort = self.effort_for(agent)
        canonical = canonical_payload(payload)
        digest = input_hash(system, canonical, model, schema, effort)
        hit = self._cached(agent, ticker, as_of, digest)
        if hit is not None:
            data, reply = hit
            self._record(agent, reply, cached=True)
            return AgentResult(agent, ticker, data, reply, cached=True, attempts=0)
        self._note_stale_miss(agent, ticker, as_of, digest)

        prompt = json.dumps(payload, indent=1, default=str) if not isinstance(payload, str) else payload
        error = ""
        for attempt in (1, 2):
            call = AgentCall(
                agent=agent,
                ticker=ticker,
                as_of=as_of,
                system=system,
                prompt=prompt if attempt == 1 else f"{prompt}\n\n{RETRY_TEMPLATE.format(error=error)}",
                model=model,
                effort=effort,
                timeout=self.timeout_for(agent),
                should_stop=should_stop,
            )
            try:
                reply = self.backend.complete(call)
            except LLMError as exc:
                raise AgentFailed(agent, ticker, "backend", str(exc)) from exc
            try:
                data = extract_json(reply.text)
                error = validation_error(schema, data) or ""
                if not error and extra_check is not None:
                    error = extra_check(data) or ""
            except ValueError as exc:
                error = str(exc)
            if not error:
                self._record(agent, reply, cached=False)
                result = AgentResult(agent, ticker, data, reply, cached=False, attempts=attempt)
                self._store(as_of, digest, result, canonical)
                return result
            self._record(agent, reply, cached=False)  # a rejected reply still costs tokens
        raise AgentFailed(agent, ticker, "schema", f"rejected twice: {error}")
