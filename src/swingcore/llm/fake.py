"""Responsibility: replay recorded agent replies from disk - the backend `--dry-run` and tests use.

It reads `<fixtures>/<agent>/<TICKER>.json`, falling back to `<agent>/default.json` and
`<agent>.json`. A missing fixture is an error, never an empty or invented reply: a dry run that
silently skipped an agent would be exactly the silent fabrication the system forbids.

The one thing a recording cannot know is fixed up on the way out: **the stamp.** Every agent must
echo the payload's `as_of`, so a reply recorded on one session would be rejected by every later one.
The replay re-stamps it with the call's own `as_of`. Nothing else in the text is touched - the
judgement and its quoted numbers stay exactly as recorded - and a fixture-backed run manifest says
`backend: fake`, so the restamped date can never be read as research about that date.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from swingcore.llm.base import AgentCall, LLMError, LLMReply

log = logging.getLogger(__name__)


class FakeBackend:
    """Deterministic, offline, free. Costs and token counts are zero and reported as zero."""

    name = "fake"

    def __init__(self, fixtures: Path) -> None:
        self.fixtures = Path(fixtures)

    def candidates(self, call: AgentCall) -> list[Path]:
        """Where a recorded reply for this call may live, most specific first."""
        return [
            self.fixtures / call.agent / f"{call.ticker}.json",
            self.fixtures / call.agent / "default.json",
            self.fixtures / f"{call.agent}.json",
        ]

    def _restamp(self, data: dict[str, Any], call: AgentCall) -> None:
        recorded = str(data.get("as_of", ""))
        if recorded and recorded[:10] != call.as_of[:10]:
            log.info(
                "%s on %s: replaying a reply recorded as_of %s under this run's %s",
                call.agent,
                call.ticker,
                recorded[:10],
                call.as_of[:10],
            )
            data["as_of"] = call.as_of

    def replay(self, text: str, call: AgentCall) -> str:
        """The recorded text, re-stamped for this run. Unparseable text is passed through untouched
        so a deliberately malformed fixture still exercises the retry path."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text
        if not isinstance(data, dict):
            return text
        self._restamp(data, call)
        return json.dumps(data)

    def complete(self, call: AgentCall) -> LLMReply:
        for path in self.candidates(call):
            if path.exists():
                return LLMReply(text=self.replay(path.read_text(), call), model=call.model, backend=self.name)
        looked = ", ".join(str(p) for p in self.candidates(call))
        raise LLMError(f"no recorded reply for {call.agent} on {call.ticker}; looked in {looked}")
