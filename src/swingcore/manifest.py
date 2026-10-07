"""Responsibility: record one cycle as a `RunManifest` - what ran, what did not, what is missing.

Everything a run produced arrives here as parts (stage outcomes, agent outputs, failures, gaps,
usage) and leaves as one schema-validated dict. Two rules govern the assembly and both exist so an
absence can never read as a decision:

* `stages` records what ran, what did not and what failed, so an empty result is always
  distinguishable from an agent that was never asked;
* every blank a reader would see - a missing sector, a ticker with no summary - is matched by an
  entry in `data_gaps` explaining it. `cross_check` refuses a manifest where one is not.

Nothing here calls an agent or decides anything. Slimmed from the private system's manifest, whose
decision blocks (portfolio, sizing, risk) are not part of this showcase.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from swingcore.calendar import session_day
from swingcore.models import AgentFailure, AgentUsage, DataGap, RunStage, StageStatus, TickerMeta
from swingcore.schemas import SchemaValidationError, validate

DRY_RUN_SUFFIX = "-dryrun"
FIXTURE_BACKEND = "fake"
RunType = Literal["morning", "midday", "postmortem"]
NO_SECTOR = "insufficient_data: the data layer returned no sector"
NO_SUMMARY = "no summary: {why}"
META_FIELDS = ("sector",)


class ManifestError(ValueError):
    """A manifest that would let a reader mistake an absence for a result."""


def run_id(session: str, run_type: RunType, backend: str) -> str:
    """The run record's identity. A fixture-backed run never claims a real run's id."""
    return f"{session}-{run_type}" + (DRY_RUN_SUFFIX if backend == FIXTURE_BACKEND else "")


@dataclass
class RunRecorder:
    """Collects a cycle as it happens. Each stage reports its own status exactly once."""

    session: str
    as_of: str
    run_type: RunType = "morning"
    stages: list[RunStage] = field(default_factory=list)
    gaps: list[DataGap] = field(default_factory=list)
    failures: list[AgentFailure] = field(default_factory=list)
    research: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, TickerMeta] = field(default_factory=dict)

    # ---- stages ---------------------------------------------------------------
    def stage(self, name: str, status: StageStatus, note: str) -> None:
        if any(s["stage"] == name for s in self.stages):
            raise ManifestError(f"stage {name!r} reported twice; a stage has one status per run")
        if not note.strip():
            raise ManifestError(f"stage {name!r} has no note; say what ran or why it did not")
        self.stages.append(RunStage(stage=name, status=status, note=note))

    def ran(self, name: str, note: str) -> None:
        self.stage(name, "ran", note)

    def not_run(self, name: str, note: str) -> None:
        self.stage(name, "not_run", note)

    def failed(self, name: str, note: str) -> None:
        self.stage(name, "failed", note)

    # ---- what came back, and what did not ---------------------------------------
    def gap(self, ticker: str, field_name: str, detail: str) -> None:
        self.gaps.append(DataGap(ticker=ticker, field=field_name, detail=detail))

    def failure(self, failure: AgentFailure) -> None:
        self.failures.append(failure)

    def output(self, ticker: str, data: dict[str, Any]) -> None:
        self.research.setdefault(ticker, []).append(data)

    def has_gap(self, ticker: str, field_name: str) -> bool:
        return any(g["ticker"] == ticker and g["field"] == field_name for g in self.gaps)

    # ---- assembly -------------------------------------------------------------
    def build(self, usage: list[AgentUsage], backend: str) -> dict[str, Any]:
        """Assemble and validate the manifest. Everything absent is absent because it is named.

        A blank sector the caller did not already explain gets the generic gap here, and a ticker
        with metadata but no summary gets one too: the manifest is the last place a silent blank
        can be caught before a reader sees it.
        """
        for ticker, meta in self.meta.items():
            for field_name in META_FIELDS:
                if not meta.get(field_name) and not self.has_gap(ticker, field_name):
                    self.gap(ticker, field_name, NO_SECTOR)
            if ticker not in self.summaries and not self.has_gap(ticker, "summary"):
                self.gap(ticker, "summary", NO_SUMMARY.format(why="no stage produced one"))
        manifest: dict[str, Any] = {
            "run_id": run_id(self.session, self.run_type, backend),
            "run_type": self.run_type,
            "backend": backend,
            "date": self.session,
            "as_of": self.as_of,
            "research": {t: list(rows) for t, rows in sorted(self.research.items())},
            "summaries": dict(sorted(self.summaries.items())),
            "ticker_meta": {t: dict(m) for t, m in sorted(self.meta.items())},
            "agent_failures": list(self.failures),
            "data_gaps": list(self.gaps),
            "usage": list(usage),
            "stages": list(self.stages),
        }
        return validate_manifest(manifest)


# ---- the checks no JSON Schema can express ----------------------------------------
def _check_coverage(raw: Mapping[str, Any], problems: list[str]) -> None:
    gapped = {(g["ticker"], g["field"]) for g in raw["data_gaps"]}
    for ticker, meta in raw["ticker_meta"].items():
        for field_name in META_FIELDS:
            if meta.get(field_name) in (None, "") and (ticker, field_name) not in gapped:
                problems.append(
                    f"ticker_meta[{ticker}].{field_name} is empty but no data gap explains it; "
                    "the report must show the gap rather than write a blank"
                )
        if ticker not in raw["summaries"] and (ticker, "summary") not in gapped:
            problems.append(f"{ticker} has no summary and no data gap saying why")


def _check_failures(raw: Mapping[str, Any], problems: list[str]) -> None:
    """A failed agent cannot hide inside a stage table that says everything ran."""
    if raw["agent_failures"] and not any(s["status"] == "failed" for s in raw["stages"]):
        names = sorted({f"{f['agent']}[{f['ticker']}]" for f in raw["agent_failures"]})
        problems.append(f"agent failure(s) {', '.join(names)} but no stage is marked failed")


def _check_research(raw: Mapping[str, Any], problems: list[str]) -> None:
    """Every stored output answers for the ticker it is filed under, on this run's session.

    The session is the New York date of each stamp (`session_day`), so a reply stamped
    `2026-09-15T01:00:00+00:00` belongs to September 14 and is refused by a September 15 run.
    """
    run_day = _session_or_none(str(raw["as_of"]))
    for ticker, rows in raw["research"].items():
        for row in rows:
            agent = row.get("agent", "?")
            if row.get("ticker") != ticker:
                problems.append(f"{agent} output filed under {ticker} names {row.get('ticker')!r}")
            day = _session_or_none(str(row.get("as_of", "")))
            if day is None or day != run_day:
                problems.append(f"{agent}[{ticker}] is stamped {row.get('as_of')!r}, not session {run_day}")


def _session_or_none(as_of: str) -> str | None:
    """The session of a stamp, or None for one that does not parse (which matches no session)."""
    try:
        return session_day(as_of)
    except ValueError:
        return None


def _check_usage(raw: Mapping[str, Any], problems: list[str]) -> None:
    for row in raw["usage"]:
        if row["cached_calls"] > row["calls"]:
            problems.append(f"usage[{row['agent']}]: more cached calls than calls")
        if row["spent_usd"] > row["cost_usd"] + 1e-9:
            problems.append(f"usage[{row['agent']}]: spent_usd exceeds cost_usd")


def cross_check(raw: Mapping[str, Any]) -> None:
    """Pure-Python checks no JSON Schema can express: the parts must agree with each other."""
    problems: list[str] = []
    _check_coverage(raw, problems)
    _check_failures(raw, problems)
    _check_research(raw, problems)
    _check_usage(raw, problems)
    if problems:
        raise ManifestError("run manifest is inconsistent:\n  - " + "\n  - ".join(problems))


def validate_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    """Fail loud on anything a reader would otherwise take as a half-truth."""
    try:
        validate("run_manifest", raw, "run manifest")
    except SchemaValidationError as exc:
        raise ManifestError(str(exc)) from exc
    cross_check(raw)
    return raw
