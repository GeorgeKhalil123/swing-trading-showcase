"""Responsibility: deterministic synthetic OHLCV, so the demo and the tests never need a data vendor.

Every price in this repository comes from here. It is a seeded random walk on business days, not a
recording of any real ticker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_bars(
    n: int = 300,
    start: str = "2025-01-02",
    seed: int = 7,
    drift: float = 0.0005,
    vol: float = 0.015,
    price0: float = 100.0,
    volume: float = 2_000_000,
    end: str | None = None,
) -> pd.DataFrame:
    """Random-walk daily bars on business days. Deterministic per seed.

    With `end`, the `n` bars finish on that date instead of starting on `start`.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n) if end else pd.bdate_range(start, periods=n)
    rets = rng.normal(drift, vol, n)
    close = price0 * np.cumprod(1 + rets)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol_s = volume * (1 + np.abs(rng.normal(0, 0.3, n)))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol_s}, index=idx)
    df.index.name = "date"
    return df
