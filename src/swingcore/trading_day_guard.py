"""Responsibility: exit non-zero when a date is not an NYSE trading session, so cron can skip it.

`cron` and `systemd` both fire on wall-clock days, not market days. Rather than teach each of them a
holiday calendar, the cycle wrapper runs this first and stops if it exits non-zero. A weekend, a
market holiday and a date NYSE never opened all fail the same way, with one line on stderr saying
which and naming the next session.

Exit codes: 0 the date is a session, 1 it is not, 2 the calendar itself could not answer (an
unparseable date, or a year the toy holiday list does not cover) - a guard that cannot check is not
a guard that passed.

    python -m swingcore.trading_day_guard --date 2026-11-26
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from swingcore.calendar import covers, is_trading_day, next_trading_day

IS_SESSION, NOT_A_SESSION, CANNOT_TELL = 0, 1, 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exit 0 if the date is an NYSE session, 1 if it is a weekend or a holiday."
    )
    p.add_argument("--date", default=date.today().isoformat(), help="date to check (YYYY-MM-DD)")
    p.add_argument("-q", "--quiet", action="store_true", help="print nothing on success")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        day = date.fromisoformat(args.date)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return CANNOT_TELL
    if not covers(day):
        print(f"error: the holiday calendar does not cover {day.year}; refusing to guess", file=sys.stderr)
        return CANNOT_TELL
    try:
        trading = is_trading_day(day)
        following = next_trading_day(day)
    except (ValueError, OSError) as exc:  # pragma: no cover - a broken calendar file
        print(f"error: the NYSE calendar could not be read: {exc}", file=sys.stderr)
        return CANNOT_TELL
    if not trading:
        weekend = day.weekday() >= 5
        why = "a weekend" if weekend else "an NYSE holiday"
        print(f"{day} is {why}; skipping this cycle. Next session: {following}.", file=sys.stderr)
        return NOT_A_SESSION
    if not args.quiet:
        print(f"{day} is an NYSE session")
    return IS_SESSION


if __name__ == "__main__":
    sys.exit(main())
