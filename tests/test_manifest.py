"""The run manifest: stages, named gaps and failures, and the cross-check that refuses silent blanks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from swingcore import demo
from swingcore.manifest import ManifestError, RunRecorder, cross_check, run_id, validate_manifest
from swingcore.models import TickerMeta


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return demo.cycle(free_slots=1)


def test_a_fixture_run_never_claims_a_real_runs_id() -> None:
    assert run_id("2026-09-14", "morning", "fake") == "2026-09-14-morning-dryrun"
    assert run_id("2026-09-14", "morning", "real") == "2026-09-14-morning"


def test_the_demo_names_its_failed_agent_and_its_missing_source(manifest: dict[str, Any]) -> None:
    assert manifest["backend"] == "fake" and manifest["run_id"].endswith("-dryrun")
    failures = {(f["agent"], f["ticker"], f["stage"]) for f in manifest["agent_failures"]}
    assert failures == {("echo_trend_agent", "BBB", "schema")}
    gaps = {(g["ticker"], g["field"]) for g in manifest["data_gaps"]}
    assert ("BBB", "headlines") in gaps, "the missing source is named"
    assert ("BBB", "summary") in gaps and ("BBB", "sector") in gaps
    assert not any(g["ticker"] == "AAA" for g in manifest["data_gaps"]), "AAA had everything"


def test_every_status_is_reported_once_and_the_failure_is_visible_in_the_stage_table(
    manifest: dict[str, Any],
) -> None:
    status = {s["stage"]: s["status"] for s in manifest["stages"]}
    assert status["data"] == "ran" and status["summary"] == "ran"
    assert status["research"] == "failed", "one agent failed, so research did not simply 'run'"
    assert status["sizing"] == "not_run"
    assert len(status) == len(manifest["stages"])


def test_replayed_outputs_are_stamped_with_this_run_not_the_recording(manifest: dict[str, Any]) -> None:
    for rows in manifest["research"].values():
        assert all(row["as_of"] == demo.AS_OF for row in rows)


def test_a_blank_field_without_a_named_gap_is_refused(manifest: dict[str, Any]) -> None:
    broken = copy.deepcopy(manifest)
    broken["data_gaps"] = [g for g in broken["data_gaps"] if g["field"] != "sector"]
    with pytest.raises(ManifestError, match=r"ticker_meta\[BBB\].sector is empty but no data gap"):
        cross_check(broken)


def test_a_ticker_without_a_summary_needs_a_reason(manifest: dict[str, Any]) -> None:
    broken = copy.deepcopy(manifest)
    broken["data_gaps"] = [g for g in broken["data_gaps"] if g["field"] != "summary"]
    with pytest.raises(ManifestError, match="BBB has no summary"):
        cross_check(broken)


def test_a_failure_cannot_hide_under_a_stage_table_that_says_everything_ran(manifest: dict[str, Any]) -> None:
    broken = copy.deepcopy(manifest)
    for stage in broken["stages"]:
        if stage["status"] == "failed":
            stage["status"] = "ran"
    with pytest.raises(ManifestError, match=r"echo_trend_agent\[BBB\] but no stage is marked failed"):
        cross_check(broken)


def test_an_output_filed_under_the_wrong_ticker_or_day_is_refused(manifest: dict[str, Any]) -> None:
    broken = copy.deepcopy(manifest)
    broken["research"]["AAA"][0]["as_of"] = "2026-09-11T16:30:00-04:00"
    broken["research"]["AAA"][1]["ticker"] = "ZZZ"
    with pytest.raises(ManifestError) as exc:
        cross_check(broken)
    assert "stamped '2026-09-11" in str(exc.value) and "names 'ZZZ'" in str(exc.value)


def test_a_run_cannot_spend_more_than_its_answers_cost(manifest: dict[str, Any]) -> None:
    broken = copy.deepcopy(manifest)
    broken["usage"][0]["spent_usd"] = 1.0
    with pytest.raises(ManifestError, match="spent_usd exceeds cost_usd"):
        cross_check(broken)


def test_the_schema_refuses_an_unknown_stage_status(manifest: dict[str, Any]) -> None:
    broken = copy.deepcopy(manifest)
    broken["stages"][0]["status"] = "skipped"
    with pytest.raises(ManifestError, match="skipped"):
        validate_manifest(broken)


def test_a_stage_reports_once_and_always_says_why() -> None:
    run = RunRecorder(session="2026-09-14", as_of="2026-09-14T16:30:00-04:00")
    run.ran("data", "ok")
    with pytest.raises(ManifestError, match="reported twice"):
        run.failed("data", "late failure")
    with pytest.raises(ManifestError, match="has no note"):
        run.not_run("research", "  ")


def test_build_names_a_blank_the_caller_forgot() -> None:
    run = RunRecorder(session="2026-09-14", as_of="2026-09-14T16:30:00-04:00")
    run.meta["CCC"] = TickerMeta(sector=None)
    run.ran("data", "ok")
    built = run.build([], backend="fake")
    assert {(g["ticker"], g["field"]) for g in built["data_gaps"]} == {("CCC", "sector"), ("CCC", "summary")}


def test_a_closed_gate_buys_no_research_and_says_so() -> None:
    closed = demo.cycle(free_slots=0)
    research = next(s for s in closed["stages"] if s["stage"] == "research")
    assert research["status"] == "not_run" and "gate closed" in research["note"]
    assert closed["usage"] == [] and closed["research"] == {}
    assert demo.research_budget(2) == 2 * demo.CANDIDATES_PER_FREE_SLOT


def test_the_demo_cli_prints_and_writes_the_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "manifest.json"
    assert demo.main(["--dry-run", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "research   failed" in printed and "GAP     BBB.headlines" in printed
    assert json.loads(out.read_text())["run_id"] == "2026-09-14-morning-dryrun"
    assert demo.main([]) == 2, "there is no live mode to fall back to"
