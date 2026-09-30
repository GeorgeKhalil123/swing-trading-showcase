"""Responsibility: deterministic technical indicators computed from daily OHLCV. No LLM, no I/O.

Every function takes a DataFrame indexed by date with columns open/high/low/close/volume (or one of
its Series) and returns plain floats / Series. Values that cannot be computed are returned as None,
never silently defaulted: a model handed `0.0` for "not enough history" would read it as a fact.

This is the generic subset (moving averages, true range, ATR, trailing returns). The private system
computes a much larger snapshot from the same bars; none of its screening logic is here.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

TRADING_DAYS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}


def _f(x: Any) -> float | None:
    """Convert to float; NaN/inf/None -> None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---- moving averages ----------------------------------------------------------
def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def slope_pct(series: pd.Series, lookback: int = 10) -> float | None:
    """Percent change of a series over `lookback` bars (a cheap, explainable slope)."""
    s = series.dropna()
    if len(s) <= lookback:
        return None
    prev = s.iloc[-1 - lookback]
    return _f((s.iloc[-1] - prev) / prev) if prev else None


# ---- volatility ---------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ATR (exponential smoothing with alpha = 1/n)."""
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# ---- trailing returns ---------------------------------------------------------
def returns(close: pd.Series) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for label, n in TRADING_DAYS.items():
        if len(close) > n and close.iloc[-1 - n]:
            out[f"ret_{label}"] = _f(close.iloc[-1] / close.iloc[-1 - n] - 1)
        else:
            out[f"ret_{label}"] = None
    return out


# ---- bundle -------------------------------------------------------------------
def snapshot(df: pd.DataFrame) -> dict[str, Any]:
    """The generic indicators for the last bar. Requires >= 30 bars; longer-window fields are None
    when history is short."""
    if len(df) < 30:
        raise ValueError(f"snapshot needs >= 30 bars, have {len(df)}")
    close = df["close"]
    last = float(close.iloc[-1])
    a = atr(df).iloc[-1]
    out: dict[str, Any] = {
        "as_of_bar": pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d"),
        "close": last,
        "n_bars": int(len(df)),
        "atr_14": _f(a),
        "atr_pct": _f(a / last) if last else None,
        "ema_21": _f(ema(close, 21).iloc[-1]),
    }
    for n in (20, 50, 200):
        s = sma(close, n)
        out[f"sma_{n}"] = _f(s.iloc[-1])
        out[f"sma_{n}_slope"] = slope_pct(s)
    out.update(returns(close))
    return out
