"""swingcore: the parts of a swing-trading research system that decide what a run is allowed to know.

Four pieces, extracted from a larger private system and trimmed to stand on their own:

* `swingcore.bitemporal_cache` - SQLite store where every row carries `as_of` *and* `fetched_at`;
* `swingcore.calendar` / `swingcore.trading_day_guard` - session arithmetic and the cron guard;
* `swingcore.llm` - one backend protocol, schema validation, exactly one retry, a canonical cache key;
* `swingcore.manifest` - the per-stage run record (ran / not_run / failed) and its cross-check.

`python -m swingcore.demo --dry-run` runs a toy cycle through all of them without touching a network.
"""

__version__ = "0.1.0"
