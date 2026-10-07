"""Point-in-time guarantees of the bitemporal SQLite cache."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from swingcore.bitemporal_cache import (
    Cache,
    as_of_utc,
    is_lookahead,
    lookahead_gap,
    newest_session,
)
from swingcore.calendar import session_day
from swingcore.synthetic import make_bars


def test_roundtrip_and_as_of(cache: Cache) -> None:
    df = make_bars(n=50)
    assert cache.upsert_bars("XYZ", df, "test") == 50
    back = cache.get_bars("XYZ")
    assert len(back) == 50
    assert back["close"].iloc[-1] == df["close"].iloc[-1]
    cutoff = df.index[29].strftime("%Y-%m-%d")
    pit = cache.get_bars("XYZ", as_of=cutoff + "T12:00:00")
    assert len(pit) == 30 and pit.index[-1] == df.index[29]


def test_as_of_beats_end(cache: Cache) -> None:
    df = make_bars(n=20)
    cache.upsert_bars("A", df, "test")
    got = cache.get_bars("A", end=df.index[-1].strftime("%Y-%m-%d"), as_of=df.index[4].strftime("%Y-%m-%d"))
    assert len(got) == 5


def test_coverage_and_payloads(cache: Cache) -> None:
    df = make_bars(n=10)
    cache.upsert_bars("B", df, "test")
    cov = cache.bars_coverage("B")
    assert cov == (df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))
    cache.put_payload("fund", "B", {"eps": 1.0}, "test", as_of="2025-01-01T00:00:00")
    cache.put_payload("fund", "B", {"eps": 2.0}, "test", as_of="2025-06-01T00:00:00")
    early, late = cache.get_payload("fund", "B", as_of="2025-03-01"), cache.get_payload("fund", "B")
    assert early is not None and early[0]["eps"] == 1.0
    assert late is not None and late[0]["eps"] == 2.0
    assert cache.bars_coverage("NOPE") is None


def test_a_payload_carries_when_it_was_really_fetched_not_only_what_it_claims(cache: Cache) -> None:
    """`as_of` is the point in time a payload describes; `fetched_at` is when it was pulled.

    They differ on any back-dated run, and only `fetched_at` can tell a replay that a row labelled
    last month was actually retrieved today.
    """
    cache.put_payload("fund", "C", {"eps": 1.0}, "synthetic", as_of="2025-01-01T00:00:00")
    hit = cache.get_payload("fund", "C")
    assert hit is not None
    assert hit.as_of == "2025-01-01T00:00:00"
    # `fetched_at` is UTC, so it is compared against the UTC day: for the hours either side of
    # midnight UTC the local date is a different one, and asserting on it would fail once a day.
    assert hit.fetched_at > hit.as_of
    assert hit.fetched_at.startswith(datetime.now(UTC).date().isoformat())
    assert hit.data == {"eps": 1.0} and hit[0] == hit.data, "still unpacks as (data, as_of, ...)"


def test_a_replay_only_sees_what_had_been_downloaded_by_its_known_at(cache: Cache) -> None:
    """The second time axis. A value stamped for Monday but downloaded on Wednesday did not exist
    on Tuesday, whatever its `as_of` says, and a Tuesday replay must not see it."""
    monday = "2026-09-14T16:30:00-04:00"
    cache.put_payload(
        "fund", "D", {"eps": 1.0}, "synthetic", as_of=monday, fetched_at="2026-09-16T09:00:00-04:00"
    )
    assert cache.get_payload("fund", "D", as_of=monday) is not None, "valid time alone would serve it"
    assert cache.get_payload("fund", "D", as_of=monday, known_at="2026-09-15T09:00:00-04:00") is None
    hit = cache.get_payload("fund", "D", as_of=monday, known_at="2026-09-16T09:00:00-04:00")
    assert hit is not None and hit.fetched_at == "2026-09-16T13:00:00+00:00", "stored on the UTC clock"


def test_known_at_compares_instants_not_strings(cache: Cache) -> None:
    """20:00 in New York is 00:00 UTC the next day; as text, the local stamp would sort earlier."""
    cache.put_payload(
        "fund", "E", {"eps": 1.0}, "synthetic", as_of="2026-09-14", fetched_at="2026-09-15T00:30:00+00:00"
    )
    assert cache.get_payload("fund", "E", known_at="2026-09-14T20:00:00-04:00") is None
    assert cache.get_payload("fund", "E", known_at="2026-09-14T20:31:00-04:00") is not None


def test_as_of_compares_instants_not_strings(cache: Cache) -> None:
    """18:00 in New York is 22:00 UTC. As text, `...T18:00:00-04:00` sorts before `...T21:00:00+00:00`,
    so a 21:00 UTC read would have been served a value that describes an hour later."""
    cache.put_payload("fund", "G", {"eps": 1.0}, "synthetic", as_of="2026-09-14T18:00:00-04:00")
    assert cache.get_payload("fund", "G", as_of="2026-09-14T21:00:00+00:00") is None, "lookahead"
    hit = cache.get_payload("fund", "G", as_of="2026-09-14T22:00:00+00:00")
    assert hit is not None and hit.as_of == "2026-09-14T18:00:00-04:00", "returned as written"


def test_as_of_does_not_hide_an_earlier_instant_written_in_another_offset(cache: Cache) -> None:
    """20:00 UTC is 16:00 in New York, so a 16:30 New York read has to see it; as text it sorts later."""
    cache.put_payload("fund", "H", {"eps": 1.0}, "synthetic", as_of="2026-09-14T20:00:00+00:00")
    hit = cache.get_payload("fund", "H", as_of="2026-09-14T16:30:00-04:00")
    assert hit is not None and hit.data == {"eps": 1.0}


def test_the_latest_payload_is_the_latest_instant_not_the_largest_string(cache: Cache) -> None:
    cache.put_payload("fund", "I", {"eps": 1.0}, "synthetic", as_of="2026-09-14T21:00:00+00:00")
    cache.put_payload("fund", "I", {"eps": 2.0}, "synthetic", as_of="2026-09-14T18:00:00-04:00")
    hit = cache.get_payload("fund", "I")
    assert hit is not None and hit.data == {"eps": 2.0}, "22:00 UTC is later than 21:00 UTC"


def test_a_bare_date_is_the_start_of_that_new_york_day_not_utc_midnight(cache: Cache) -> None:
    """UTC midnight on the 15th is 20:00 on the 14th in New York, so reading "2026-09-15" as UTC
    would serve a row labelled with the 15th to the evening of the 14th."""
    cache.put_payload("fund", "K", {"eps": 1.0}, "synthetic", as_of="2026-09-15")
    assert cache.get_payload("fund", "K", as_of="2026-09-14T22:00:00-04:00") is None
    assert cache.get_payload("fund", "K", as_of="2026-09-15T00:00:00-04:00") is not None
    assert cache.get_payload("fund", "K", as_of="2026-09-15") is not None


def test_a_stamp_without_an_offset_is_new_york_time(cache: Cache) -> None:
    cache.put_payload("fund", "L", {"eps": 1.0}, "synthetic", as_of="2026-09-14T18:00:00")
    assert cache.get_payload("fund", "L", as_of="2026-09-14T21:30:00+00:00") is None, "17:30 New York"
    assert cache.get_payload("fund", "L", as_of="2026-09-14T22:00:00+00:00") is not None


def test_as_of_keeps_sub_second_order(cache: Cache) -> None:
    cache.put_payload("fund", "M", {"eps": 1.0}, "synthetic", as_of="2026-09-14T20:00:00.900000+00:00")
    assert cache.get_payload("fund", "M", as_of="2026-09-14T20:00:00.100000+00:00") is None
    assert cache.get_payload("fund", "M", as_of="2026-09-14T20:00:00.900000+00:00") is not None
    assert as_of_utc("2026-09-14T16:00:00-04:00") == "2026-09-14T20:00:00.000000+00:00", "fixed width"


def test_the_bar_cut_off_is_the_new_york_date_of_an_offset_stamp(cache: Cache) -> None:
    """01:00 UTC on the 15th is 21:00 on the 14th in New York: the 15th's bar does not exist yet."""
    days = pd.bdate_range("2026-09-14", periods=2)
    bars = pd.DataFrame({c: [1.0, 2.0] for c in ("open", "high", "low", "close", "volume")}, index=days)
    cache.upsert_bars("N", bars, "synthetic")
    got = cache.get_bars("N", as_of="2026-09-15T01:00:00+00:00")
    assert [d.date().isoformat() for d in got.index] == ["2026-09-14"]
    assert session_day("2026-09-15T01:00:00+00:00") == "2026-09-14"
    assert session_day("2026-09-15") == "2026-09-15"


def test_lookahead_uses_the_new_york_date_of_an_offset_stamp() -> None:
    """After the close of the 15th, a stamp at 21:00 New York on the 14th is a past session."""
    after_close = datetime(2026, 9, 15, 17, 0, tzinfo=ZoneInfo("America/New_York"))
    assert is_lookahead("2026-09-15T01:00:00+00:00", now=after_close)
    assert "as_of 2026-09-14" in lookahead_gap("2026-09-15T01:00:00+00:00", "bars", now=after_close)
    assert not is_lookahead("2026-09-15T21:00:00+00:00", now=after_close)


def test_bars_downloaded_after_the_replay_moment_are_refused(cache: Cache) -> None:
    df = make_bars(n=20)
    cache.upsert_bars("F", df.iloc[:10], "synthetic", fetched_at="2025-01-20T00:00:00+00:00")
    cache.upsert_bars("F", df.iloc[10:], "synthetic", fetched_at="2025-02-10T00:00:00+00:00")
    assert len(cache.get_bars("F")) == 20
    assert len(cache.get_bars("F", known_at="2025-01-21")) == 10


def test_a_cache_file_written_before_fetched_at_existed_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE payloads (kind TEXT NOT NULL, key TEXT NOT NULL, as_of TEXT NOT NULL,"
        " source TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY (kind, key, as_of));"
        "INSERT INTO payloads VALUES ('fund','D','2025-01-01','synthetic','{\"eps\": 3.0}');"
    )
    conn.commit()
    conn.close()
    migrated = Cache(path)
    hit = migrated.get_payload("fund", "D")
    assert hit is not None and hit.data == {"eps": 3.0} and hit.fetched_at == ""
    assert migrated.get_payload("fund", "D", known_at="2030-01-01") is None, (
        "unknown fetch time is not 'always'"
    )
    assert migrated.get_payload("fund", "D", as_of="2025-01-01T00:00:00-05:00") is not None, (
        "an old row is given a UTC stamp, so offset-aware cut-offs still find it"
    )
    assert migrated.get_payload("fund", "D", as_of="2024-12-31T23:00:00-05:00") is None, (
        "and a New Year's Eve read still does not see a row dated New Year's Day"
    )


def _legacy_payloads(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """A payloads table as the version before `as_of_utc` wrote it: (key, as_of, as_of_utc) rows."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE payloads (kind TEXT NOT NULL, key TEXT NOT NULL, as_of TEXT NOT NULL,"
        " source TEXT NOT NULL, payload TEXT NOT NULL, fetched_at TEXT NOT NULL DEFAULT '',"
        " as_of_utc TEXT NOT NULL DEFAULT '', PRIMARY KEY (kind, key, as_of))"
    )
    conn.executemany(
        "INSERT INTO payloads VALUES ('fund', ?, ?, 'synthetic', '{}', '', ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_a_legacy_row_with_an_unparseable_as_of_opens_but_is_never_served(tmp_path: Path) -> None:
    """Refusing to open the whole cache over one bad row would be worse; serving it would be wrong."""
    path = tmp_path / "bad.sqlite"
    _legacy_payloads(path, [("BAD", "last tuesday", ""), ("OK", "2025-01-02", "")])
    migrated = Cache(path)
    assert migrated.get_payload("fund", "BAD") is None
    assert migrated.get_payload("fund", "BAD", as_of="2030-01-01") is None
    assert migrated.get_payload("fund", "OK", as_of="2025-01-03") is not None


def test_stamps_written_by_the_previous_as_of_utc_shape_are_recomputed(tmp_path: Path) -> None:
    """The first `as_of_utc` had no microseconds and read a bare date as UTC midnight."""
    path = tmp_path / "v1.sqlite"
    _legacy_payloads(path, [("J", "2026-09-15", "2026-09-15T00:00:00+00:00")])
    migrated = Cache(path)
    assert migrated.get_payload("fund", "J", as_of="2026-09-14T22:00:00-04:00") is None
    stored = migrated.conn.execute("SELECT as_of_utc FROM payloads WHERE key='J'").fetchone()[0]
    assert stored == "2026-09-15T04:00:00.000000+00:00"


def test_payload_keys_like_returns_same_day_rows_only(cache: Cache) -> None:
    """The prefix query the agent cache asks: "did I already ask this agent about this ticker today?"

    The underscore in every agent name is why the prefix is escaped: in SQL `LIKE`, `_` matches any
    single character, so an unescaped `toy_headline_agent|` would also match `toyXheadline_agent|`.
    """
    morning = "2026-09-14T07:30:00-04:00"
    for key in ("recorder|toy_headline_agent|AAA|aaa", "recorder|toy_headline_agent|AAA|bbb"):
        cache.put_payload("agent", key, {"data": key}, "recorder", as_of=morning)
    tomorrow = "2026-09-15T07:30:00-04:00"
    cache.put_payload("agent", "recorder|toy_headline_agent|AAA|ccc", {}, "recorder", as_of=tomorrow)
    cache.put_payload("agent", "recorder|toy_headline_agent|BBB|ddd", {}, "recorder", as_of=morning)
    cache.put_payload("agent", "recorder|toyXheadline_agent|AAA|eee", {}, "recorder", as_of=morning)
    cache.put_payload("agent", "other|toy_headline_agent|AAA|fff", {}, "recorder", as_of=morning)
    cache.put_payload("fund", "recorder|toy_headline_agent|AAA|ggg", {}, "recorder", as_of=morning)

    keys = cache.payload_keys_like("agent", "recorder|toy_headline_agent|AAA|", "2026-09-14")
    assert keys == ["recorder|toy_headline_agent|AAA|aaa", "recorder|toy_headline_agent|AAA|bbb"]
    assert cache.payload_keys_like("agent", "recorder|toy_headline_agent|AAA|", "2026-09-16") == []
    assert len(cache.payload_keys_like("agent", "", "2026-09-14")) == 5, "an empty prefix means all"


def test_payload_keys_like_matches_the_new_york_session_not_the_written_date(cache: Cache) -> None:
    """01:00 UTC on the 15th was written as "2026-09-15" but is the evening of the 14th."""
    cache.put_payload("agent", "r|a|AAA|late", {}, "r", as_of="2026-09-15T01:00:00+00:00")
    cache.put_payload("agent", "r|a|AAA|next", {}, "r", as_of="2026-09-15T09:30:00-04:00")
    assert cache.payload_keys_like("agent", "r|a|AAA|", "2026-09-14") == ["r|a|AAA|late"]
    assert cache.payload_keys_like("agent", "r|a|AAA|", "2026-09-15") == ["r|a|AAA|next"]
    assert cache.payload_keys_like("agent", "r|a|AAA|", "2026-09-15T00:30:00+00:00") == ["r|a|AAA|late"]


def test_a_session_older_than_the_newest_close_is_flagged_as_lookahead_with_an_actionable_gap() -> None:
    newest = newest_session()
    stale = (newest - timedelta(days=30)).isoformat()
    assert is_lookahead(stale + "T16:30:00-04:00")
    assert not is_lookahead((newest + timedelta(days=1)).isoformat())
    gap = lookahead_gap(stale, "fundamentals")
    assert gap.startswith("insufficient_data:") and stale in gap and "fundamentals" in gap


def test_the_newest_completed_session_is_fetchable_even_though_its_date_is_not_today() -> None:
    """A pre-open run on D+1 is stamped D 16:30; that is the run's own data, not lookahead."""
    stamp = newest_session().isoformat() + "T16:30:00-04:00"
    assert not is_lookahead(stamp)
    assert is_lookahead((newest_session() - timedelta(days=7)).isoformat() + "T16:30:00-04:00")


def test_the_guard_flips_at_the_close_of_the_next_session_not_at_midnight() -> None:
    """Pinned to named instants, because against the wall clock this case cannot be tested.

    Monday 2026-09-14's stamp is the run's own data all through Tuesday morning and becomes
    lookahead only once Tuesday's 16:00 ET close has passed.
    """
    ny = ZoneInfo("America/New_York")
    monday = "2026-09-14T16:30:00-04:00"
    tuesday_morning = datetime(2026, 9, 15, 11, 0, tzinfo=ny)
    tuesday_after_close = datetime(2026, 9, 15, 16, 30, tzinfo=ny)
    assert newest_session(tuesday_morning).isoformat() == "2026-09-14"
    assert not is_lookahead(monday, now=tuesday_morning), "yesterday's session is still fetchable"
    assert newest_session(tuesday_after_close).isoformat() == "2026-09-15"
    assert is_lookahead(monday, now=tuesday_after_close), "a newer session has now closed"
    gap = lookahead_gap(monday, "fundamentals", now=tuesday_after_close)
    assert "2026-09-14" in gap and "2026-09-15" in gap


def test_a_holiday_weekend_does_not_move_the_newest_session() -> None:
    """Labor Day 2026 is Monday 09-07: from Saturday to Tuesday's close the newest session is Friday."""
    ny = ZoneInfo("America/New_York")
    for instant in (datetime(2026, 9, 5, 12, tzinfo=ny), datetime(2026, 9, 7, 17, tzinfo=ny)):
        assert newest_session(instant).isoformat() == "2026-09-04"
    assert newest_session(datetime(2026, 9, 8, 16, 1, tzinfo=ny)).isoformat() == "2026-09-08"


def test_a_file_cache_lets_a_second_process_read_while_a_run_writes(tmp_path: Path) -> None:
    """WAL lets a reader through while a cycle writes; the busy timeout makes two writers queue."""
    cache = Cache(tmp_path / "shared.sqlite")
    assert cache.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert cache.conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000
    other = Cache(tmp_path / "shared.sqlite")
    cache.put_payload("fund", "E", {"eps": 4.0}, "test")
    assert other.get_payload("fund", "E") is not None, "the second connection sees the first's write"
    cache.close()
    other.close()
