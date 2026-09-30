"""Indicator correctness against independent pandas computations and known bounds."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from swingcore import indicators as ind
from swingcore.synthetic import make_bars


def test_sma_matches_rolling_mean(bars: pd.DataFrame) -> None:
    s = ind.sma(bars["close"], 20)
    assert np.isclose(s.iloc[-1], bars["close"].iloc[-20:].mean())
    assert s.iloc[:19].isna().all()


def test_ema_needs_its_window_before_it_reports() -> None:
    e = ind.ema(pd.Series(np.linspace(1, 30, 30)), 21)
    assert e.iloc[:20].isna().all() and not math.isnan(e.iloc[-1])


def test_true_range_uses_the_previous_close_across_a_gap() -> None:
    df = pd.DataFrame(
        {
            "open": [10.0, 12.0],
            "high": [10.5, 12.5],
            "low": [9.5, 11.8],
            "close": [10.0, 12.2],
            "volume": [1, 1],
        }
    )
    assert ind.true_range(df).iloc[-1] == pytest.approx(12.5 - 10.0), "the gap counts, not just the day range"


def test_atr_positive_and_scale(bars: pd.DataFrame) -> None:
    a = ind.atr(bars).iloc[-1]
    assert a > 0
    assert 0.005 < a / bars["close"].iloc[-1] < 0.05  # ~1.5% daily vol synthetic series


def test_atr_of_a_constant_range_is_that_range() -> None:
    n = 40
    close = pd.Series(np.full(n, 100.0))
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0})
    assert ind.atr(df).iloc[-1] == pytest.approx(2.0)
    assert ind.atr(df).iloc[:13].isna().all(), "Wilder's ATR is undefined before n bars"


def test_returns_use_exact_lookback(bars: pd.DataFrame) -> None:
    r = ind.returns(bars["close"])
    assert r["ret_1w"] == pytest.approx(bars["close"].iloc[-1] / bars["close"].iloc[-6] - 1)
    assert r["ret_1y"] == pytest.approx(bars["close"].iloc[-1] / bars["close"].iloc[-253] - 1)


def test_returns_none_when_short() -> None:
    r = ind.returns(make_bars(n=40)["close"])
    assert r["ret_1w"] is not None and r["ret_3m"] is None


def test_slope_is_none_without_enough_history() -> None:
    assert ind.slope_pct(pd.Series([1.0, 2.0, 3.0]), lookback=10) is None
    assert ind.slope_pct(pd.Series(np.linspace(100, 110, 11)), lookback=10) == pytest.approx(0.1)


def test_non_finite_values_become_none_not_zero() -> None:
    assert ind._f(float("nan")) is None and ind._f(float("inf")) is None and ind._f(None) is None
    assert ind._f("1.5") == 1.5


def test_snapshot_keys(bars: pd.DataFrame) -> None:
    s = ind.snapshot(bars)
    for k in ("atr_14", "atr_pct", "sma_20", "sma_200", "sma_50_slope", "ret_3m", "ema_21"):
        assert k in s
    assert s["n_bars"] == len(bars)
    assert s["as_of_bar"] == bars.index[-1].strftime("%Y-%m-%d")


def test_snapshot_short_history_has_none_not_zero() -> None:
    s = ind.snapshot(make_bars(n=40))
    assert s["sma_200"] is None and s["ret_1y"] is None
    assert s["sma_20"] is not None


def test_snapshot_rejects_tiny_frame() -> None:
    with pytest.raises(ValueError):
        ind.snapshot(make_bars(n=10))


def test_no_nan_leaks(bars: pd.DataFrame) -> None:
    s = ind.snapshot(bars)
    for k, v in s.items():
        if isinstance(v, float):
            assert not np.isnan(v), k
