"""The LLM runtime: JSON extraction, schema validation, the single retry, caching and usage.

A tiny in-process recorder stands in for a real backend; `test_fake_backend.py` covers the fixture
replay backend itself.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest

from swingcore.bitemporal_cache import Cache
from swingcore.demo import FIXTURES
from swingcore.llm import AgentCall, AgentFailed, AgentResult, LLMError, LLMReply, Runtime
from swingcore.llm.cache_key import (
    canonical_payload,
    input_hash,
    reset_stale_misses,
    stale_miss_report,
)
from swingcore.llm.runtime import extract_json

AGENT = "echo_trend_agent"
VALID = json.loads((FIXTURES / AGENT / "AAA.json").read_text())


class Recorder:
    # Any name but "fake": a fixture read is free, so the runtime deliberately never caches it.
    name = "recorder"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[AgentCall] = []

    def complete(self, call: AgentCall) -> LLMReply:
        self.calls.append(call)
        return LLMReply(
            text=self.replies.pop(0) if self.replies else "{}",
            model=call.model,
            backend=self.name,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.002,
        )


def runtime(*replies: str, cache: Cache | None = None) -> tuple[Runtime, Recorder]:
    backend = Recorder(*replies)
    return Runtime(backend=backend, cache=cache, models={AGENT: "m"}), backend


def run(rt: Runtime, as_of: str = "2026-09-14T07:30:00-04:00", payload: Any = None) -> AgentResult:
    return rt.run(
        agent=AGENT,
        ticker="AAA",
        as_of=as_of,
        system="be precise",
        payload=payload if payload is not None else {"close": 100.0},
        schema="toy_trend",
    )


# ---- parsing ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Here is the object:\n{"a": 1}\nThat is all.',
    ],
)
def test_extract_json_tolerates_fences_and_prose(text: str) -> None:
    assert extract_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["no object here", '{"a": ', "[1, 2, 3]"])
def test_extract_json_fails_loud(text: str) -> None:
    with pytest.raises(ValueError):
        extract_json(text)


def test_input_hash_tracks_every_component() -> None:
    base = input_hash("sys", {"a": 1}, "model", "toy_trend")
    assert base == input_hash("sys", {"a": 1}, "model", "toy_trend")
    assert base != input_hash("sys", {"a": 2}, "model", "toy_trend")
    assert base != input_hash("sys", {"a": 1}, "other", "toy_trend")
    assert base != input_hash("other", {"a": 1}, "model", "toy_trend")
    assert base != input_hash("sys", {"a": 1}, "model", "toy_tone")
    assert base != input_hash("sys", {"a": 1}, "model", "toy_trend", "low")


# ---- the canonical form the key is taken over ------------------------------------
def test_canonical_payload_drops_the_minute_and_the_float_noise_but_not_the_numbers() -> None:
    """What an agent was told, without when it was told or which run assembled the floats."""
    payload = {
        "ticker": "AAA",
        "indicators": {"atr_14": 1.2345649, "sma_20": 54.0},
        "bars": [{"close": 100.000049, "volume": 1_000_000}],
        "session": {
            "date": "2026-09-14",
            "partial_bar": True,
            "fetched_at": "2026-09-14T12:31:00-04:00",
            "note": "The prints in it are what had reached the cache at 2026-09-14T12:31:00-04:00.",
        },
        "gaps": [],
    }
    assert canonical_payload(payload) == {
        "ticker": "AAA",
        "indicators": {"atr_14": 1.2346, "sma_20": 54.0},
        "bars": [{"close": 100.0, "volume": 1_000_000}],
        "session": {"date": "2026-09-14", "partial_bar": True},
        "gaps": [],
    }
    assert canonical_payload(canonical_payload(payload)) == canonical_payload(payload), "idempotent"


def test_canonical_payload_keeps_a_difference_that_is_a_real_difference() -> None:
    """The rounding is a floor, not a blur: a tenth of a basis point is noise, a cent is not."""
    assert canonical_payload({"close": 100.0}) != canonical_payload({"close": 101.0})
    assert canonical_payload({"close": 0.5231}) != canonical_payload({"close": 0.5232})
    assert canonical_payload({"note": "a"}) != canonical_payload({"note": "b"})
    assert canonical_payload({"news": [{"fetched_at": "12:31", "headline": "x"}]}) == {
        "news": [{"headline": "x"}]
    }
    # -0.0 and 0.0 are one number but two JSON tokens; a value crossing zero must not cost a re-run.
    assert json.dumps(canonical_payload({"drift": -0.00001})) == json.dumps({"drift": 0.0})
    assert canonical_payload({"flag": True}) == {"flag": True}, "a bool is not a float to round"


# ---- validation and retry --------------------------------------------------------
def test_a_valid_reply_is_returned_and_counted() -> None:
    rt, backend = runtime(json.dumps(VALID))
    result = run(rt)
    assert result.data == VALID and result.attempts == 1 and not result.cached
    assert len(backend.calls) == 1 and backend.calls[0].model == "m"
    usage = rt.usage()[0]
    assert usage["calls"] == 1 and usage["input_tokens"] == 100 and usage["cost_usd"] == 0.002


def test_an_invalid_reply_is_retried_once_with_the_error() -> None:
    broken = json.dumps(VALID | {"strength": "enormous"})
    rt, backend = runtime(broken, json.dumps(VALID))
    result = run(rt)
    assert result.attempts == 2 and result.data == VALID
    assert "enormous" in backend.calls[1].prompt and "rejected" in backend.calls[1].prompt
    assert rt.usage()[0]["calls"] == 2  # a rejected reply still cost tokens


def test_two_invalid_replies_mark_the_agent_failed() -> None:
    broken = json.dumps(VALID | {"trend": "maybe"})
    rt, backend = runtime(broken, broken)
    with pytest.raises(AgentFailed) as exc:
        run(rt)
    assert len(backend.calls) == 2
    failure = exc.value.failure
    assert failure["agent"] == AGENT and failure["ticker"] == "AAA" and failure["stage"] == "schema"
    assert "trend" in failure["error"]


def test_an_extra_check_can_reject_a_schema_valid_reply() -> None:
    rt, backend = runtime(json.dumps(VALID), json.dumps(VALID))
    with pytest.raises(AgentFailed, match="not a number we gave you"):
        rt.run(
            agent=AGENT,
            ticker="AAA",
            as_of="2026-09-14",
            system="s",
            payload={},
            schema="toy_trend",
            extra_check=lambda data: "sma_20 is not a number we gave you",
        )
    assert len(backend.calls) == 2


def test_a_backend_failure_is_an_agent_failure_not_a_crash() -> None:
    class Broken:
        name = "recorder"

        def complete(self, call: AgentCall) -> LLMReply:
            raise LLMError("backend not installed")

    with pytest.raises(AgentFailed, match="backend not installed") as exc:
        run(Runtime(backend=Broken(), models={AGENT: "m"}))
    assert exc.value.failure["stage"] == "backend"


def test_no_model_configured_is_an_agent_failure() -> None:
    with pytest.raises(AgentFailed, match="no model configured"):
        run(Runtime(backend=Recorder(), models={}))


def test_effort_and_timeout_fall_back_to_the_defaults_and_reach_the_call() -> None:
    backend = Recorder(json.dumps(VALID))
    rt = Runtime(
        backend=backend,
        default_model="m",
        efforts={"slow_agent": "high"},
        default_effort="low",
        timeouts={"slow_agent": 420.0},
        timeout=90.0,
    )
    assert rt.effort_for("slow_agent") == "high" and rt.effort_for(AGENT) == "low"
    assert rt.timeout_for("slow_agent") == 420.0 and rt.timeout_for(AGENT) == 90.0
    run(rt)
    assert backend.calls[0].effort == "low" and backend.calls[0].timeout == 90.0


# ---- caching ---------------------------------------------------------------------
def test_one_backends_reply_is_never_served_back_to_another(cache: Cache) -> None:
    """Two backends share a cache; the backend is part of the key so they cannot mix."""
    first, _ = runtime(json.dumps(VALID), cache=cache)
    run(first)

    class Real(Recorder):
        name = "real"

    real_backend = Real(json.dumps(VALID))
    real = Runtime(backend=real_backend, cache=cache, models={AGENT: "m"})
    result = run(real)
    assert not result.cached, "the real run must ask the model, not replay another backend"
    assert len(real_backend.calls) == 1


def test_a_fixture_reply_is_never_cached_at_all(cache: Cache) -> None:
    """A fixture read is already free, and a cached one would outlive the file it came from."""

    class Fake(Recorder):
        name = "fake"

    backend = Fake(json.dumps(VALID), json.dumps(VALID))
    rt = Runtime(backend=backend, cache=cache, models={AGENT: "m"})
    first, second = run(rt), run(rt)
    assert len(backend.calls) == 2, "the fixture is read again, not replayed from SQLite"
    assert not first.cached and not second.cached
    stored = cache.conn.execute("SELECT COUNT(*) FROM payloads WHERE kind='agent'").fetchone()[0]
    assert stored == 0, "nothing the fake backend said was written to the payloads table"


def test_the_same_day_and_same_input_is_served_from_cache(cache: Cache) -> None:
    rt, backend = runtime(json.dumps(VALID), cache=cache)
    first = run(rt)
    second = run(rt)
    assert len(backend.calls) == 1, "a same-day re-run must not call the backend again"
    assert second.cached and second.data == first.data
    usage = rt.usage()[0]
    assert usage["calls"] == 2 and usage["cached_calls"] == 1
    # The cached call restores what the call that filled the cache cost; `spent_usd` is what *this*
    # run paid, and the replay paid nothing.
    assert usage["input_tokens"] == 200 and usage["output_tokens"] == 40
    assert usage["cost_usd"] == 0.004 and usage["spent_usd"] == 0.002


def test_a_replay_in_a_new_runtime_still_shows_what_the_call_cost(cache: Cache) -> None:
    first, _ = runtime(json.dumps(VALID), cache=cache)
    run(first)
    replay, backend = runtime(cache=cache)
    result = run(replay)
    assert result.cached and not backend.calls
    usage = replay.usage()[0]
    assert usage["calls"] == 1 and usage["cached_calls"] == 1
    assert usage["input_tokens"] == 100 and usage["output_tokens"] == 20
    assert usage["cost_usd"] == 0.002, "the cost table on a replay shows the original call's cost"
    assert usage["spent_usd"] == 0.0, "and this run spent nothing to get it"


def test_a_cache_row_written_before_costs_were_stored_restores_zero_not_a_guess(cache: Cache) -> None:
    rt, _ = runtime(json.dumps(VALID), cache=cache)
    run(rt)
    cache.conn.execute(
        "UPDATE payloads SET payload=? WHERE kind='agent'",
        (json.dumps({"data": VALID, "reply": {"model": "m", "backend": "recorder"}}),),
    )
    cache.conn.commit()
    replay, _ = runtime(cache=cache)
    assert run(replay).cached
    usage = replay.usage()[0]
    assert usage["input_tokens"] == 0 and usage["cost_usd"] == 0.0


def test_a_cache_row_with_no_usage_receipts_is_a_miss_not_a_free_model_call(cache: Cache) -> None:
    """Replaying a row with no reply metadata would put a model call in the cost table at $0.00."""
    rt, backend = runtime(json.dumps(VALID), cache=cache)
    run(rt)
    cache.conn.execute("UPDATE payloads SET payload = json_remove(payload, '$.reply') WHERE kind = 'agent'")
    cache.conn.commit()
    replay, replay_backend = runtime(json.dumps(VALID), cache=cache)
    result = run(replay)
    assert not result.cached and len(replay_backend.calls) == 1
    assert replay.usage()[0]["spent_usd"] == 0.002, "the re-run paid for the answer it now has"
    assert len(backend.calls) == 1


def test_changed_input_or_a_new_day_re_runs_the_agent(cache: Cache) -> None:
    rt, backend = runtime(*[json.dumps(VALID)] * 3, cache=cache)
    run(rt)
    run(rt, payload={"close": 101.0})
    run(rt, as_of="2026-09-15T07:30:00-04:00")
    assert len(backend.calls) == 3


def test_a_reply_from_the_new_york_evening_is_not_replayed_into_the_next_session(cache: Cache) -> None:
    """01:00 UTC on the 15th is 21:00 on the 14th in New York. As text both stamps say "2026-09-15",
    and comparing the first ten characters replayed the 14th's reply into the 15th's session."""
    rt, backend = runtime(*[json.dumps(VALID)] * 2, cache=cache)
    run(rt, as_of="2026-09-15T01:00:00+00:00")
    second = run(rt, as_of="2026-09-15T09:30:00-04:00")
    assert not second.cached and len(backend.calls) == 2


def test_one_new_york_session_written_in_two_offsets_is_one_cache_entry(cache: Cache) -> None:
    rt, backend = runtime(json.dumps(VALID), cache=cache)
    run(rt, as_of="2026-09-14T20:00:00-04:00")
    second = run(rt, as_of="2026-09-15T00:30:00+00:00")  # 20:30 on the 14th in New York
    assert second.cached and len(backend.calls) == 1


def midday_payload(atr: float, minute: str) -> dict[str, Any]:
    """An intraday payload: an indicator off a re-pulled bar tail, and the minute it was pulled."""
    return {
        "ticker": "AAA",
        "indicators": {"atr_14": atr},
        "session": {
            "date": "2026-09-14",
            "as_of": "2026-09-14T12:30:00-04:00",
            "partial_bar": True,
            "fetched_at": minute,
            "note": f"The prints in it are what had reached the cache at {minute}.",
        },
    }


def test_float_noise_and_fetched_at_do_not_change_the_cache_key(cache: Cache) -> None:
    """The two causes of a same-day miss, in one payload: neither is a new question."""
    rt, backend = runtime(json.dumps(VALID), cache=cache)
    run(rt, payload=midday_payload(1.00001, "2026-09-14T12:31:00-04:00"))
    second = run(rt, payload=midday_payload(1.00002, "2026-09-14T13:07:00-04:00"))
    assert len(backend.calls) == 1, "same data an hour later is the same question"
    assert second.cached and second.data == VALID


def test_a_stored_row_carries_its_canonical_input(cache: Cache) -> None:
    """The answer is only diffable if the question was kept, in the form the key was taken over."""
    rt, _ = runtime(json.dumps(VALID), cache=cache)
    run(rt, payload=midday_payload(1.00001, "2026-09-14T12:31:00-04:00"))
    key, blob = cache.conn.execute("SELECT key, payload FROM payloads WHERE kind='agent'").fetchone()
    stored = json.loads(blob)
    assert stored["data"] == VALID
    assert stored["input"] == {
        "ticker": "AAA",
        "indicators": {"atr_14": 1.0},
        "session": {"date": "2026-09-14", "as_of": "2026-09-14T12:30:00-04:00", "partial_bar": True},
    }
    assert stored["input_digest"] == key.rsplit("|", 1)[-1], "the digest names the row it is in"


def test_a_same_day_input_change_is_counted_as_a_stale_miss(
    cache: Cache, caplog: pytest.LogCaptureFixture
) -> None:
    """The first call of the day is research; the second, under a different input, is the same
    research paid for twice, and the run has to say so."""
    reset_stale_misses()
    rt, backend = runtime(*[json.dumps(VALID)] * 3, cache=cache)
    with caplog.at_level("INFO", logger="swingcore.llm.runtime"):
        run(rt, payload={"close": 100.0})
        assert not rt.stale_misses, "a first call is a miss, but not a stale one"
        run(rt, payload={"close": 101.0})
    assert rt.stale_misses == Counter({AGENT: 1})
    assert "same-day input changed" in caplog.text and "AAA" in caplog.text
    report = stale_miss_report()
    assert report.total == 1 and report.days == ("2026-09-14",)
    assert report.by_agent == {AGENT: 1}
    run(rt, payload={"close": 101.0}, as_of="2026-09-15T07:30:00-04:00")
    assert rt.stale_misses == Counter({AGENT: 1}), "a new day starts from nothing"
    assert len(backend.calls) == 3
    reset_stale_misses()


def test_a_fixture_backed_run_counts_no_stale_miss(cache: Cache) -> None:
    class Fake(Recorder):
        name = "fake"

    reset_stale_misses()
    rt = Runtime(backend=Fake(*[json.dumps(VALID)] * 2), cache=cache, models={AGENT: "m"})
    run(rt, payload={"close": 100.0})
    run(rt, payload={"close": 101.0})
    assert not rt.stale_misses and stale_miss_report().total == 0


def test_a_stale_miss_is_counted_on_the_new_york_session_not_the_utc_date(cache: Cache) -> None:
    reset_stale_misses()
    rt, _ = runtime(*[json.dumps(VALID)] * 4, cache=cache)
    run(rt, payload={"close": 100.0}, as_of="2026-09-14T20:00:00-04:00")
    run(rt, payload={"close": 101.0}, as_of="2026-09-15T00:30:00+00:00")  # same New York evening
    assert rt.stale_misses == Counter({AGENT: 1})
    assert stale_miss_report().days == ("2026-09-14",)
    run(rt, payload={"close": 102.0}, as_of="2026-09-15T09:30:00-04:00")
    assert rt.stale_misses == Counter({AGENT: 1}), "the next morning is a new session, not a stale miss"
    reset_stale_misses()


def test_usage_is_sorted_by_agent_not_by_who_answered_first() -> None:
    backend = Recorder(json.dumps(VALID), json.dumps(VALID))
    rt = Runtime(backend=backend, default_model="m")
    for agent in ("zeta_agent", "alpha_agent"):
        rt.run(agent, "AAA", "2026-09-14", "s", {}, "toy_trend")
    assert [row["agent"] for row in rt.usage()] == ["alpha_agent", "zeta_agent"]
