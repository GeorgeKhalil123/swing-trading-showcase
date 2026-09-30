"""The scheduler's one piece of logic: the guard that stops a cycle on a day the NYSE is shut.

`cron` and `systemd` both fire on wall-clock days. Rather than teach either of them a holiday
calendar, the cycle runs `python -m swingcore.trading_day_guard` first and stops when it exits
non-zero. That exit code is the whole contract, so it is asserted here by running the module the way
cron runs it.

The last tests read the shipped crontab and units as text. They cannot prove cron fires - nothing in
a test suite can - but they can prove the time, the timezone, the weekday restriction and the guard
are all in the files a user is told to install.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from swingcore.calendar import (
    is_trading_day,
    last_trading_day,
    next_trading_day,
    trading_days_between,
)
from swingcore.trading_day_guard import CANNOT_TELL, IS_SESSION, NOT_A_SESSION

ROOT = Path(__file__).resolve().parents[1]
CRONTAB = ROOT / "deploy" / "crontab.example"
UNITS = ROOT / "deploy" / "systemd"


def guard(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "swingcore.trading_day_guard", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("day", "why"),
    [
        ("2026-01-01", "New Year's Day"),
        ("2026-07-03", "Independence Day, observed on the Friday"),
        ("2026-11-26", "Thanksgiving"),
        ("2026-12-25", "Christmas Day"),
    ],
)
def test_a_holiday_stops_the_cycle(day: str, why: str) -> None:
    result = guard("--date", day)
    assert result.returncode == NOT_A_SESSION, f"{day} ({why}) is not a session"
    assert "NYSE holiday" in result.stderr
    assert "Next session:" in result.stderr, "a skipped day has to say when the next one is"


def test_a_weekend_stops_the_cycle() -> None:
    saturday = guard("--date", "2026-09-12")
    assert saturday.returncode == NOT_A_SESSION
    assert "weekend" in saturday.stderr
    assert guard("--date", "2026-09-13").returncode == NOT_A_SESSION


def test_a_trading_day_lets_the_cycle_run() -> None:
    result = guard("--date", "2026-09-15")
    assert result.returncode == IS_SESSION
    assert "is an NYSE session" in result.stdout
    assert guard("--date", "2026-09-15", "--quiet").stdout == "", "cron mail only on the exception"


def test_a_date_it_cannot_parse_is_not_treated_as_a_pass() -> None:
    """Exit 2 is not exit 0: a guard that could not check is not a guard that passed."""
    result = guard("--date", "not-a-date")
    assert result.returncode == CANNOT_TELL and "error:" in result.stderr


def test_a_year_the_toy_calendar_does_not_cover_is_not_treated_as_a_pass() -> None:
    """Outside the listed years every weekday would look open; the guard refuses to guess."""
    result = guard("--date", "2031-01-01")
    assert result.returncode == CANNOT_TELL and "does not cover 2031" in result.stderr


def test_session_arithmetic_steps_over_weekends_and_holidays() -> None:
    thanksgiving = date(2026, 11, 26)
    assert not is_trading_day(thanksgiving)
    assert last_trading_day(thanksgiving) == date(2026, 11, 25)
    assert next_trading_day(date(2026, 7, 2)) == date(2026, 7, 6), "Friday holiday, then the weekend"
    assert last_trading_day(date(2026, 9, 15)) == date(2026, 9, 15), "a session is its own last session"
    week = trading_days_between(date(2026, 9, 7), date(2026, 9, 13))
    assert week == [date(2026, 9, d) for d in (8, 9, 10, 11)], "Labor Day week has four sessions"


# ---- what the user is told to install ---------------------------------------------------
def test_the_crontab_runs_on_weekdays_in_new_york_behind_the_guard() -> None:
    text = CRONTAB.read_text()
    assert "CRON_TZ=America/New_York" in text
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("30") and "swingcore.demo" in ln)
    assert line.split()[:5] == ["30", "7", "*", "*", "1-5"], "the cycle is weekdays at 07:30"
    guard_at, cycle_at = line.index("trading_day_guard"), line.index("swingcore.demo")
    assert guard_at < cycle_at and "&&" in line[guard_at:cycle_at], "the guard gates the cycle"


def test_the_systemd_unit_is_persistent_in_new_york_and_conditioned_on_the_guard() -> None:
    timer = (UNITS / "swingcore-morning.timer").read_text()
    service = (UNITS / "swingcore-morning.service").read_text()
    assert "Persistent=true" in timer, "a missed cycle has to be caught up"
    assert "America/New_York" in timer and "Mon..Fri" in timer
    assert "ExecCondition=" in service and "trading_day_guard" in service
