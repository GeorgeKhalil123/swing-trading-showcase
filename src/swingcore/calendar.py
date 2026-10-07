"""Responsibility: NYSE trading-day arithmetic (last session on/before a date, the newest completed one).

The private system asks `pandas_market_calendars`; this showcase reads a hand-typed toy holiday list
(`holidays.yaml`, full-day closures only) behind the same function names, so the lookahead rule in
`swingcore.bitemporal_cache` and the cron guard can be exercised without the dependency.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

HOLIDAYS_FILE = Path(__file__).with_name("holidays.yaml")
NEW_YORK = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)
SEARCH_DAYS = 10  # a longer run of closed days than this means the holiday file is wrong


def as_of_instant(stamp: str) -> datetime:
    """The instant an `as_of` names: an offset is honoured, anything without one is New York time.

    A bare date is therefore the start of that New York day. Raises ValueError on an unparseable
    stamp.
    """
    parsed = datetime.fromisoformat(stamp)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=NEW_YORK)


def session_day(as_of: str) -> str:
    """The New York date an `as_of` falls on, which is the session it belongs to.

    Every "same day?" question about an `as_of` goes through this, never through `as_of[:10]`:
    `2026-09-15T01:00:00+00:00` is 21:00 on September 14 in New York.
    """
    return as_of_instant(as_of).astimezone(NEW_YORK).date().isoformat()


@cache
def _calendar() -> tuple[frozenset[date], frozenset[int]]:
    raw = yaml.safe_load(HOLIDAYS_FILE.read_text()) or {}
    days = frozenset(
        d if isinstance(d, date) else date.fromisoformat(str(d)) for d in raw.get("holidays", [])
    )
    years = frozenset(int(y) for y in raw.get("years", []))
    return days, years


def holidays() -> frozenset[date]:
    return _calendar()[0]


def covers(d: date) -> bool:
    """True when the holiday list claims to be complete for `d`'s year.

    Outside those years a weekday would look like a session simply because nobody typed the
    holiday in, which is why the guard treats an uncovered year as "cannot tell", not as a pass.
    """
    return d.year in _calendar()[1]


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in holidays()


def last_trading_day(d: date) -> date:
    """`d` itself if it is a session, else the most recent session before it."""
    for back in range(SEARCH_DAYS + 1):
        day = d - timedelta(days=back)
        if is_trading_day(day):
            return day
    raise ValueError(f"no NYSE session within {SEARCH_DAYS} days before {d}")


def next_trading_day(d: date) -> date:
    """The first session strictly after `d`."""
    for ahead in range(1, SEARCH_DAYS + 1):
        day = d + timedelta(days=ahead)
        if is_trading_day(day):
            return day
    raise ValueError(f"no NYSE session within {SEARCH_DAYS} days after {d}")


def trading_days_between(start: date, end: date) -> list[date]:
    days = (start + timedelta(days=i) for i in range((end - start).days + 1))
    return [d for d in days if is_trading_day(d)]


def last_completed_session(now: datetime | None = None) -> date:
    """Most recent NYSE session whose close is already in the past (America/New_York).
    While the market is open, that is the previous session, so partial bars never count."""
    now = (now or datetime.now(tz=NEW_YORK)).astimezone(NEW_YORK)
    today = now.date()
    if is_trading_day(today) and now.time() >= MARKET_CLOSE:
        return today
    return last_trading_day(today - timedelta(days=1))
