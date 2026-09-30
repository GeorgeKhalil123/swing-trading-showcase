"""Responsibility: a three-stage toy cycle that exercises every piece of `swingcore` offline.

    python -m swingcore.demo --dry-run

Two synthetic tickers, two toy agents, the `fake` backend replaying recorded fixtures. Nothing here
is the private system's research: the tickers, the prices, the headlines, the agents, their schemas
and their prompts are all made up for the demo, and it calls no model and no network.

Two things go wrong on purpose, because a manifest that has only ever recorded success proves
nothing:

* BBB has **no headline source** in the cache, so `toy_headline_agent` is never asked about it and
  the absence is recorded as a named data gap;
* BBB's recorded `echo_trend_agent` reply is **invalid against its schema** (a trend of
  "sideways"), so the runtime retries once, fails again and records a named agent failure, and the
  research stage says `failed` instead of `ran`.

The research budget is the toy form of the free-slot gate: research is capped at
`free_slots x CANDIDATES_PER_FREE_SLOT`, so a book with no room buys no research at all.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from swingcore.bitemporal_cache import Cache
from swingcore.calendar import is_trading_day
from swingcore.indicators import snapshot
from swingcore.llm import AgentFailed, FakeBackend, Runtime
from swingcore.manifest import RunRecorder
from swingcore.models import TickerMeta
from swingcore.synthetic import make_bars

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "agents"
SESSION = "2026-09-14"
AS_OF = f"{SESSION}T16:30:00-04:00"
FETCHED_AT = f"{SESSION}T16:45:00-04:00"  # when the toy "provider" answered, after the close
MODEL = "toy-model"
CANDIDATES_PER_FREE_SLOT = 2  # TOY value for the demo, not the private configuration
# TOY universe: synthetic seeds, not real companies. BBB deliberately has no sector on file.
UNIVERSE: dict[str, dict[str, Any]] = {
    "AAA": {"seed": 3, "sector": "Toy Industrials", "drift": 0.0012},
    "BBB": {"seed": 11, "sector": None, "drift": -0.0004},
}
HEADLINES: dict[str, list[str]] = {
    "AAA": [
        "AAA (synthetic) reports a quiet quarter in line with its own toy guidance",
        "AAA (synthetic) opens a second toy factory",
    ],
}
INDICATOR_FIELDS = ("close", "sma_20", "sma_50", "atr_pct", "ret_1m")
TREND_SYSTEM = "TOY PROMPT: name the trend of the table you are given; quote only numbers in it."
TONE_SYSTEM = "TOY PROMPT: name the tone of the headlines you are given; cite them verbatim."


def research_budget(free_slots: int, per_slot: int = CANDIDATES_PER_FREE_SLOT) -> int:
    """TOY free-slot gate: how many candidates the book could use today. Zero means no research."""
    return max(0, free_slots) * per_slot


def quoted_numbers_check(indicators: dict[str, float | None]) -> Callable[[dict[str, Any]], str | None]:
    """Reject a reply that quotes a number it was not given. The model reads numbers; it never makes one."""

    def check(data: dict[str, Any]) -> str | None:
        for item in data.get("quoted", []):
            given = indicators.get(item["field"])
            if given is None:
                return f"quoted field {item['field']!r} is not in the table you were given"
            if not math.isclose(float(item["value"]), given, abs_tol=1e-4):
                return f"quoted {item['field']}={item['value']} but the table says {given}"
        return None

    return check


def cited_headlines_check(headlines: list[str]) -> Callable[[dict[str, Any]], str | None]:
    def check(data: dict[str, Any]) -> str | None:
        invented = [c for c in data.get("cited", []) if c not in headlines]
        return f"cited headline(s) not in the input: {invented}" if invented else None

    return check


# ---- stage 1: data ---------------------------------------------------------------------------
def stage_data(cache: Cache, run: RunRecorder) -> dict[str, dict[str, float | None]]:
    """Synthetic bars into the bitemporal cache, indicators computed from an as-of read."""
    tables: dict[str, dict[str, float | None]] = {}
    for ticker, spec in UNIVERSE.items():
        frame = make_bars(n=320, end=SESSION, seed=spec["seed"], drift=spec["drift"])
        frame = frame[[is_trading_day(ts.date()) for ts in frame.index]]
        cache.upsert_bars(ticker, frame, "synthetic", fetched_at=FETCHED_AT)
        bars = cache.get_bars(ticker, as_of=AS_OF, known_at=FETCHED_AT)
        snap = snapshot(bars)
        tables[ticker] = {k: None if snap[k] is None else round(float(snap[k]), 4) for k in INDICATOR_FIELDS}
        run.meta[ticker] = TickerMeta(sector=spec["sector"], last=round(float(snap["close"]), 4))
        if ticker in HEADLINES:
            cache.put_payload(
                "headlines", ticker, HEADLINES[ticker], "synthetic", as_of=AS_OF, fetched_at=FETCHED_AT
            )
    run.ran("data", f"synthetic bars for {len(UNIVERSE)} tickers; indicators computed in Python")
    return tables


# ---- stage 2: research -----------------------------------------------------------------------
def stage_research(
    cache: Cache,
    runtime: Runtime,
    run: RunRecorder,
    tables: dict[str, dict[str, float | None]],
    free_slots: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    budget = research_budget(free_slots)
    candidates = sorted(tables)[:budget]
    if not candidates:
        run.not_run("research", f"free-slot gate closed: {free_slots} free slot(s), no research bought")
        return {}
    answers: dict[str, dict[str, dict[str, Any]]] = {t: {} for t in candidates}
    asked = failed = 0
    for ticker in candidates:
        jobs: list[tuple[str, str, Any, str, Callable[[dict[str, Any]], str | None]]] = [
            (
                "echo_trend_agent",
                TREND_SYSTEM,
                {"ticker": ticker, "as_of": AS_OF, "indicators": tables[ticker]},
                "toy_trend",
                quoted_numbers_check(tables[ticker]),
            )
        ]
        headlines = cache.get_payload("headlines", ticker, as_of=AS_OF, known_at=FETCHED_AT)
        if headlines is None:
            run.gap(
                ticker,
                "headlines",
                "insufficient_data: no headline source for this ticker; toy_headline_agent was not asked",
            )
        else:
            jobs.append(
                (
                    "toy_headline_agent",
                    TONE_SYSTEM,
                    {"ticker": ticker, "as_of": AS_OF, "headlines": headlines.data},
                    "toy_tone",
                    cited_headlines_check(list(headlines.data)),
                )
            )
        for agent, system, payload, schema, check in jobs:
            asked += 1
            try:
                result = runtime.run(agent, ticker, AS_OF, system, payload, schema, extra_check=check)
            except AgentFailed as exc:
                failed += 1
                run.failure(exc.failure)
                continue
            answers[ticker][agent] = result.data
            run.output(ticker, result.data)
    note = (
        f"budget {budget} = {free_slots} free slot(s) x {CANDIDATES_PER_FREE_SLOT}; "
        f"{asked} agent run(s), {failed} failed after its one retry"
    )
    (run.failed if failed else run.ran)("research", note)
    return answers


# ---- stage 3: summary ------------------------------------------------------------------------
def stage_summary(
    run: RunRecorder,
    tables: dict[str, dict[str, float | None]],
    answers: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """A pure-Python roll-up. A ticker without a trend gets a named gap, never a default."""
    for ticker in sorted(tables):
        trend = answers.get(ticker, {}).get("echo_trend_agent")
        if trend is None:
            run.gap(ticker, "summary", "no summary: echo_trend_agent produced no valid output")
            continue
        tone = answers[ticker].get("toy_headline_agent")
        run.summaries[ticker] = {
            "trend": trend["trend"],
            "strength": trend["strength"],
            "tone": tone["tone"] if tone else None,
            "atr_pct": tables[ticker]["atr_pct"],
            "ret_1m": tables[ticker]["ret_1m"],
        }
    run.ran("summary", "per-ticker roll-up computed in Python from validated outputs only")


def cycle(free_slots: int = 1) -> dict[str, Any]:
    cache = Cache(":memory:")
    runtime = Runtime(backend=FakeBackend(FIXTURES), cache=cache, default_model=MODEL)
    run = RunRecorder(session=SESSION, as_of=AS_OF)
    tables = stage_data(cache, run)
    answers = stage_research(cache, runtime, run, tables, free_slots)
    stage_summary(run, tables, answers)
    for private in ("synthesis", "sizing", "report"):
        run.not_run(private, "not part of this showcase; the full system runs it")
    return run.build(runtime.usage(), backend=runtime.backend.name)


def print_stages(manifest: dict[str, Any]) -> None:
    print(f"run {manifest['run_id']}  backend={manifest['backend']}  as_of={manifest['as_of']}")
    for s in manifest["stages"]:
        print(f"  {s['stage']:<10} {s['status']:<8} {s['note']}")
    for f in manifest["agent_failures"]:
        print(f"  FAILED  {f['agent']}[{f['ticker']}] at {f['stage']}: {f['error'][:120]}")
    for g in manifest["data_gaps"]:
        print(f"  GAP     {g['ticker']}.{g['field']}: {g['detail'][:120]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the toy cycle on recorded fixtures and print its manifest.")
    p.add_argument("--dry-run", action="store_true", help="replay recorded fixtures (the only mode here)")
    p.add_argument("--free-slots", type=int, default=1, help="open slots in the toy book (0 closes the gate)")
    p.add_argument("--out", type=Path, help="also write the manifest JSON to this path")
    args = p.parse_args(argv)
    if not args.dry_run:
        print("error: this showcase has no live backend; pass --dry-run", file=sys.stderr)
        return 2
    manifest = cycle(args.free_slots)
    print_stages(manifest)
    text = json.dumps(manifest, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
