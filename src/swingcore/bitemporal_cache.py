"""Responsibility: SQLite point-in-time cache for daily bars and generic JSON payloads.

Choice: SQLite (stdlib, single file, atomic writes, SQL date filtering) over Parquet.
Every row - bar *and* payload - carries `fetched_at`, the wall-clock moment the value was
actually pulled, next to `as_of`, the point in time it is labelled with. The two differ
whenever a run is dated in the past: a fundamentals payload fetched today but stamped
`as_of` last month would be lookahead, and only `fetched_at` can say so. That pair is the
"bitemporal" part: `as_of` is valid time, `fetched_at` is the time the system learned it.

Reads therefore take two cut-offs. `as_of` refuses anything that describes a later moment;
`known_at` additionally refuses anything the system had not yet downloaded at that moment, so a
replay of a past morning sees what that morning could have seen and nothing restated since.
Payloads are keyed (kind, key, as_of); a re-fetch overwrites the row and moves `fetched_at`
forward, which is exactly what makes a `known_at` replay report "not yet known" rather than
quietly serving the restated value.

A payload's `as_of` is returned exactly as the caller wrote it, because its date part is the
session it belongs to (the runtime and the manifest both compare `as_of[:10]`). The cut-off and the
ordering use a second column, `as_of_utc`, holding the same instant on the UTC clock: two stamps
written with different offsets cannot be compared as text without serving a later value to an
earlier read.

`as_of` is exchange time. A stamp with an offset is the instant it names; a bare date or a stamp
without an offset is read on the New York clock, a bare date as the start of that New York day (the
moment the day's label first applies, which is also how a bare date compared before stamps carried
offsets). Its day - for a bar cut-off or the lookahead rule - is the New York date of that instant,
so `2026-09-15T01:00:00+00:00` belongs to the evening of September 14. `fetched_at` and `known_at`
are machine time and keep reading an offset-less stamp as UTC.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from swingcore.calendar import NEW_YORK

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ticker TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    source TEXT NOT NULL, fetched_at TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS bars_date ON bars(date);
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, key TEXT NOT NULL,
    source TEXT NOT NULL, fetched_at TEXT NOT NULL, n_rows INTEGER, detail TEXT
);
CREATE TABLE IF NOT EXISTS payloads (
    kind TEXT NOT NULL, key TEXT NOT NULL, as_of TEXT NOT NULL,
    source TEXT NOT NULL, payload TEXT NOT NULL, fetched_at TEXT NOT NULL DEFAULT '',
    as_of_utc TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (kind, key, as_of)
);
"""


BUSY_TIMEOUT = 30.0  # seconds a second process waits for the writer instead of failing outright


def utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _utc(stamp: str) -> str:
    """An ISO stamp in UTC, so timestamp comparisons are string comparisons on one clock.

    A bare date means the start of that UTC day. Offsets are converted rather than trusted, because
    `2026-09-14T20:00:00-04:00` sorts *before* `2026-09-14T21:00:00+00:00` as text but is the later
    instant.
    """
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat()


def _as_of_instant(stamp: str) -> datetime:
    """The instant an `as_of` names: offsets are honoured, everything else is New York time."""
    parsed = datetime.fromisoformat(stamp)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=NEW_YORK)


def as_of_utc(stamp: str) -> str:
    """An `as_of` on the UTC clock, fixed width down to the microsecond.

    Every value has the same `YYYY-MM-DDTHH:MM:SS.ffffff+00:00` shape, so comparing two of them as
    text is comparing the instants; dropping the microseconds would let a row at 20:00:00.9 through
    to a read at 20:00:00.1.
    """
    return _as_of_instant(stamp).astimezone(UTC).isoformat(timespec="microseconds")


AS_OF_UTC_SHAPE = "____-__-__T__:__:__.______+00:00"  # LIKE pattern of a value `as_of_utc` wrote


def session_day(as_of: str) -> str:
    """The New York date an `as_of` falls on, which is the session it can know about."""
    return _as_of_instant(as_of).astimezone(NEW_YORK).date().isoformat()


def newest_session(now: datetime | None = None) -> date:
    """The most recent NYSE session whose close had already passed at `now` (default: right now).

    Lookahead is defined against the *market*, not the calendar. A morning run is stamped at the
    previous session's close on purpose, so comparing that stamp to `date.today()` would refuse every
    scheduled pre-open run its own data. What actually makes a fetch lookahead is that a newer
    session has since closed: the provider would answer with prices the stamped moment could not
    have seen.

    `now` is injectable so the rule can be tested at a named instant instead of against the wall
    clock, which is the only way a test can pin "before the close" and "after the close" apart.
    """
    from swingcore.calendar import last_completed_session  # local: keeps the data layer import-light

    return last_completed_session(now)


def is_lookahead(as_of: str, now: datetime | None = None) -> bool:
    """True when fetching now would stamp a newer session's values with an older session's `as_of`.

    False for the newest completed session (the pre-open run's own stamp) and for anything later;
    true only once at least one more session has closed since. `fetched_at` on every row records the
    real retrieval moment, so a replay can still see the gap between the two.
    """
    return session_day(as_of) < newest_session(now).isoformat()


def lookahead_gap(as_of: str, what: str, now: datetime | None = None) -> str:
    """The gap text a caller records instead of fetching."""
    return (
        f"insufficient_data: as_of {session_day(as_of)} is an older session than the newest completed one "
        f"({newest_session(now).isoformat()}), so fetching {what} now would label a later session's "
        "values with a past date. Nothing was fetched; only what the cache already holds for that "
        "date may be used."
    )


class CachedPayload(NamedTuple):
    """One stored payload with both of its timestamps.

    `as_of` is the point in time the payload claims to describe; `fetched_at` is when it was really
    retrieved. A reader that cares about lookahead compares the two.
    """

    data: Any
    as_of: str
    fetched_at: str


class Cache:
    """Thin wrapper over one SQLite file. Safe to open repeatedly; schema is idempotent.

    Agents run in a thread pool and each of them reads and writes the `payloads` table, so the
    connection is opened for multi-thread use and every statement is serialised behind one lock.
    SQLite is fast enough that the lock is never the bottleneck next to a model call, and one
    connection keeps the point-in-time rules in a single place.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=BUSY_TIMEOUT)
        self._lock = threading.RLock()
        with self._lock:
            self._share()
            self.conn.executescript(_SCHEMA)
            self._migrate()

    def _share(self) -> None:
        """Let a second process read this cache while a run is writing to it.

        WAL lets readers through while one writer works, and the busy timeout makes two writers
        queue instead of failing. A file on a filesystem that cannot do WAL (a network share) keeps
        its journal mode and only gets the timeout.
        """
        try:
            self.conn.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT * 1000)}")
            if str(self.path) != ":memory:":
                self.conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:  # pragma: no cover - only on an exotic filesystem
            pass

    def _migrate(self) -> None:
        """Add columns a cache file written by an earlier version has no room for."""
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(payloads)")}
        with self.conn:
            if "fetched_at" not in columns:
                self.conn.execute("ALTER TABLE payloads ADD COLUMN fetched_at TEXT NOT NULL DEFAULT ''")
            if "as_of_utc" not in columns:
                self.conn.execute("ALTER TABLE payloads ADD COLUMN as_of_utc TEXT NOT NULL DEFAULT ''")
            self._backfill_as_of_utc()

    def _backfill_as_of_utc(self) -> None:
        """Give every row a current-shape `as_of_utc`: rows from before the column, and rows an
        earlier version stamped without microseconds or with a bare date read as UTC midnight.

        A row whose `as_of` does not parse keeps a blank, and `get_payload` never serves a blank:
        a row that cannot say what moment it describes cannot pass a point-in-time cut-off.
        """
        stale = self.conn.execute(
            "SELECT DISTINCT as_of FROM payloads WHERE as_of_utc NOT LIKE ?", (AS_OF_UTC_SHAPE,)
        ).fetchall()
        updates = []
        for (raw,) in stale:
            try:
                updates.append((as_of_utc(raw), raw))
            except ValueError:
                updates.append(("", raw))
        self.conn.executemany("UPDATE payloads SET as_of_utc=? WHERE as_of=?", updates)

    # ---- bars -------------------------------------------------------------
    def upsert_bars(self, ticker: str, bars: pd.DataFrame, source: str, fetched_at: str | None = None) -> int:
        """bars: DatetimeIndex, columns open/high/low/close/volume. Returns rows written."""
        if bars.empty:
            return 0
        now = _utc(fetched_at) if fetched_at else utcnow_iso()
        dates = pd.DatetimeIndex(bars.index).strftime("%Y-%m-%d")
        rows = [
            (
                ticker,
                dates[i],
                *[None if pd.isna(r[c]) else float(r[c]) for c in BAR_COLUMNS],
                source,
                now,
            )
            for i, (_, r) in enumerate(bars[BAR_COLUMNS].iterrows())
        ]
        with self._lock, self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?,?)", rows)
            self.conn.execute(
                "INSERT INTO fetch_log(kind,key,source,fetched_at,n_rows) VALUES (?,?,?,?,?)",
                ("bars", ticker, source, now, len(rows)),
            )
        return len(rows)

    def get_bars(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        as_of: str | None = None,
        known_at: str | None = None,
    ) -> pd.DataFrame:
        """Return bars with start <= date <= end. `as_of` (ISO) additionally refuses any bar
        whose date is after as_of's New York date (`session_day`), enforcing point-in-time
        discipline; `known_at` refuses any bar the cache had not yet fetched at that moment."""
        q = "SELECT date,open,high,low,close,volume FROM bars WHERE ticker=?"
        params: list[Any] = [ticker]
        if start:
            q += " AND date>=?"
            params.append(start)
        hard_end = end
        if as_of:
            as_of_date = session_day(as_of)
            hard_end = min(end, as_of_date) if end else as_of_date
        if hard_end:
            q += " AND date<=?"
            params.append(hard_end)
        if known_at:
            q += " AND fetched_at<=?"
            params.append(_utc(known_at))
        q += " ORDER BY date"
        with self._lock:
            df = pd.read_sql_query(q, self.conn, params=params, parse_dates=["date"])
        return df.set_index("date")

    def bars_coverage(self, ticker: str) -> tuple[str, str] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT MIN(date), MAX(date) FROM bars WHERE ticker=?", (ticker,)
            ).fetchone()
        return (row[0], row[1]) if row and row[0] else None

    # ---- generic payloads (fundamentals, news, agent replies) ---------------
    def put_payload(
        self,
        kind: str,
        key: str,
        payload: Any,
        source: str,
        as_of: str | None = None,
        fetched_at: str | None = None,
    ) -> str:
        """Store one payload under (kind, key, as_of), stamped with the moment it was written.

        `as_of` is kept as written and also stored on the UTC clock for the `get_payload` cut-off.
        `fetched_at` defaults to now. It is a parameter only so a fixture can record a download
        that really happened at another moment; nothing in a live run passes it.
        """
        as_of = as_of or utcnow_iso()
        utc_as_of = as_of_utc(as_of)
        stamp = _utc(fetched_at) if fetched_at else utcnow_iso()
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO payloads (kind, key, as_of, source, payload, fetched_at, as_of_utc)"
                " VALUES (?,?,?,?,?,?,?)",
                (kind, key, as_of, source, json.dumps(payload, default=str), stamp, utc_as_of),
            )
        return as_of

    def get_payload(
        self, kind: str, key: str, as_of: str | None = None, known_at: str | None = None
    ) -> CachedPayload | None:
        """Latest payload for (kind,key) with as_of <= given as_of, with both timestamps.

        "Latest" and "<=" are decided on the UTC clock, so a row stamped 18:00 in New York is
        later than a 21:00 UTC read and is refused, whatever offsets the two stamps were written in.
        A row whose `as_of` could not be parsed (only possible in a migrated file) is never served.

        With `known_at`, only a row fetched on or before that moment qualifies. A row whose
        `fetched_at` is blank (written before the column existed) cannot prove when it was learned,
        so a `known_at` read refuses it rather than assuming it was always there.
        """
        q = "SELECT payload, as_of, fetched_at FROM payloads WHERE kind=? AND key=? AND as_of_utc<>''"
        params: list[Any] = [kind, key]
        if as_of:
            q += " AND as_of_utc<=?"
            params.append(as_of_utc(as_of))
        if known_at:
            q += " AND fetched_at<>'' AND fetched_at<=?"
            params.append(_utc(known_at))
        with self._lock:
            row = self.conn.execute(
                q + " ORDER BY as_of_utc DESC, fetched_at DESC LIMIT 1", params
            ).fetchone()
        return CachedPayload(json.loads(row[0]), row[1], row[2] or "") if row else None

    def payload_keys_like(self, kind: str, prefix: str, as_of_day: str) -> list[str]:
        """Every key under `kind` that starts with `prefix` and is stamped on `as_of_day`.

        The agent cache keys its rows `backend|agent|ticker|digest`, so a prefix query over the
        first three fields answers the question the run itself cannot: "have I already asked this
        agent about this ticker today, under some other input?". Only the day part of `as_of` is
        compared, because a cycle stamps every row with the same session timestamp and two runs of
        the same morning differ in `fetched_at`, not in `as_of`.

        The day is the session day as the stamp was written (`as_of[:10]`), not its UTC day: that is
        the day the runtime compares when it decides whether a cached reply belongs to this run, and
        an evening stamp in New York would otherwise fall on the next UTC day.

        Rows are ordered by `fetched_at`, so the last key returned is the most recently written one.
        `%` and `_` in `prefix` are escaped: agent names contain underscores, and an unescaped one
        is a single-character wildcard that would silently match a different agent.
        """
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            rows = self.conn.execute(
                "SELECT key FROM payloads WHERE kind=? AND key LIKE ? ESCAPE '\\' AND as_of LIKE ?"
                " ORDER BY fetched_at, key",
                (kind, escaped + "%", as_of_day[:10] + "%"),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def log_fetch(self, kind: str, key: str, source: str, n_rows: int, detail: str = "") -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO fetch_log(kind,key,source,fetched_at,n_rows,detail) VALUES (?,?,?,?,?,?)",
                (kind, key, source, utcnow_iso(), n_rows, detail),
            )

    def close(self) -> None:
        with self._lock:
            self.conn.close()
