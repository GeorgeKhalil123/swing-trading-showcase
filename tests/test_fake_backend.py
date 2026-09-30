"""The `fake` backend: recorded replies from disk, re-stamped for the run that replays them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swingcore.demo import FIXTURES
from swingcore.llm import AgentCall, AgentFailed, FakeBackend, LLMError, Runtime

VALID = json.loads((FIXTURES / "echo_trend_agent" / "AAA.json").read_text())


def test_fake_backend_replays_recorded_replies() -> None:
    backend = FakeBackend(FIXTURES)
    call = AgentCall("echo_trend_agent", "AAA", "2026-09-11", "s", "p", "m")
    assert json.loads(backend.complete(call).text) == VALID
    missing = AgentCall("echo_trend_agent", "ZZZ", "2026-09-14", "s", "p", "m")
    with pytest.raises(LLMError, match="no recorded reply"):
        backend.complete(missing)


def test_a_replay_is_re_stamped_with_the_run_that_is_replaying_it() -> None:
    """Every agent must echo the payload's as_of, so a recording is unusable on any later date.

    The stamp is the one thing a recording cannot know. Nothing else is touched.
    """
    backend = FakeBackend(FIXTURES)
    call = AgentCall("echo_trend_agent", "AAA", "2027-03-01T16:30:00-05:00", "s", "p", "m")
    replayed = json.loads(backend.complete(call).text)
    assert replayed["as_of"] == "2027-03-01T16:30:00-05:00"
    assert {k: v for k, v in replayed.items() if k != "as_of"} == {
        k: v for k, v in VALID.items() if k != "as_of"
    }, "only the stamp may change: no trend, number or note is rewritten"


def test_a_reply_that_is_not_json_is_replayed_untouched() -> None:
    """A deliberately malformed fixture still has to exercise the runtime's retry path."""
    backend = FakeBackend(FIXTURES)
    call = AgentCall("echo_trend_agent", "AAA", "2026-09-14", "s", "p", "m")
    assert backend.replay("not json at all", call) == "not json at all"
    assert backend.replay("[1, 2]", call) == "[1, 2]"


def test_the_lookup_falls_back_from_ticker_to_default_to_agent_file(tmp_path: Path) -> None:
    (tmp_path / "a_agent").mkdir()
    (tmp_path / "a_agent" / "default.json").write_text('{"from": "default"}')
    (tmp_path / "b_agent.json").write_text('{"from": "agent file"}')
    backend = FakeBackend(tmp_path)
    assert json.loads(backend.complete(AgentCall("a_agent", "BBB", "d", "s", "p", "m")).text) == {
        "from": "default"
    }
    assert json.loads(backend.complete(AgentCall("b_agent", "BBB", "d", "s", "p", "m")).text) == {
        "from": "agent file"
    }


def test_a_missing_fixture_is_a_named_backend_failure_through_the_runtime() -> None:
    """A dry run never invents a reply: no recording means a failure the manifest can name."""
    rt = Runtime(backend=FakeBackend(FIXTURES), default_model="m")
    with pytest.raises(AgentFailed) as exc:
        rt.run("toy_headline_agent", "BBB", "2026-09-14", "s", {}, "toy_tone")
    assert exc.value.failure["stage"] == "backend" and "no recorded reply" in exc.value.failure["error"]
    assert rt.usage() == [], "a call that never reached a backend costs nothing and is not counted"
