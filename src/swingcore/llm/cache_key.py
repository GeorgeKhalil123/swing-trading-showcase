"""Responsibility: what makes two agent calls the same call - the cache key, and what it forgives.

The agent cache is keyed on a hash of everything an agent was told. Taken over the payload exactly
as written, two kinds of noise look like new research and a run pays for the same answer twice:

* the data layer re-pulls the last few sessions on every run and the provider restates that tail's
  volume and adjusted close, so every float derived from it moves in its far decimals;
* an intraday cycle stamps its payloads with the wall-clock minute they were assembled
  (`fetched_at`), and repeats that minute in the session `note`. A key containing the current minute
  can never be hit.

`canonical_payload` is the answer to both: round floats to four places, drop `fetched_at` at any
depth, drop the session note. Four places is a deliberate floor. A price, a ratio or a percentage
that differs by less than 1e-4 is not a different question; 100.0 and 101.0, or 0.5231 and 0.5232,
still are.

What this module does *not* do is relax the rules that keep a replay honest. The backend, the model,
the schema and the effort stay in the key, `as_of` stays in the payload, and the runtime still
refuses to serve an answer across days: yesterday's judgement against today's bars is a correctness
bug, not a saving.

A miss that is not a first call is a *stale miss*: the same agent, the same ticker, the same day,
under a digest that has since moved. It is counted here, per agent and per day, for the line a cycle
prints at the end of a run.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

FLOAT_PLACES = 4  # below this a bar-tail restatement is noise, not a different question
VOLATILE_KEYS = ("fetched_at",)  # the wall-clock minute a payload was assembled, at any depth
SESSION_KEY = "session"
SESSION_NOTE_KEY = "note"  # the intraday session note quotes `fetched_at` in prose


def canonical_payload(payload: Any) -> Any:
    """The payload as the cache key should see it: the same question, none of the run-to-run noise.

    Every float is rounded to `FLOAT_PLACES`, every key in `VOLATILE_KEYS` is dropped and the session
    `note` is dropped, at every depth. The function is idempotent - canonicalising a canonical
    payload returns it unchanged - so the digest is the same whether `input_hash` is handed the raw
    payload or the canonical form stored beside the reply.
    """
    if isinstance(payload, Mapping):
        canonical: dict[str, Any] = {}
        for raw_key, value in payload.items():
            key = str(raw_key)
            if key in VOLATILE_KEYS:
                continue
            value = canonical_payload(value)
            if key == SESSION_KEY and isinstance(value, dict):
                value.pop(SESSION_NOTE_KEY, None)
            canonical[key] = value
        return canonical
    if isinstance(payload, list | tuple):
        return [canonical_payload(item) for item in payload]
    if isinstance(payload, bool) or not isinstance(payload, float):
        return payload
    # `+ 0.0` only to turn -0.0 into 0.0: they are the same number but not the same JSON token, and
    # a value that crossed zero by a millionth would otherwise cost a full re-run.
    return round(payload, FLOAT_PLACES) + 0.0


def input_hash(system: str, payload: Any, model: str, schema: str, effort: str = "") -> str:
    """Cache identity: same agent, same instructions, same data, same model, schema and effort.

    Effort belongs in the key because a low-effort answer is not the same answer as a high-effort
    one; replaying one for the other would quietly change what the run manifest recorded.
    """
    blob = json.dumps(
        {
            "system": system,
            "payload": canonical_payload(payload),
            "model": model,
            "schema": schema,
            "effort": effort,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class StaleMissReport:
    """Calls this process paid for again although it had asked the same question earlier today."""

    by_agent: Mapping[str, int]
    days: tuple[str, ...]

    @property
    def total(self) -> int:
        return sum(self.by_agent.values())


_stale_misses: Counter[str] = Counter()
_stale_days: set[str] = set()
_stale_lock = threading.Lock()


def record_stale_miss(agent: str, day: str) -> None:
    """Count one call that was paid for again although the same question was asked earlier that day."""
    with _stale_lock:
        _stale_misses[agent] += 1
        _stale_days.add(day)


def stale_miss_report() -> StaleMissReport:
    """Every stale miss seen anywhere in this process, as a snapshot."""
    with _stale_lock:
        return StaleMissReport(by_agent=dict(_stale_misses), days=tuple(sorted(_stale_days)))


def reset_stale_misses() -> None:
    """Forget the process-wide tally. For tests, and for a second cycle in one process."""
    with _stale_lock:
        _stale_misses.clear()
        _stale_days.clear()
