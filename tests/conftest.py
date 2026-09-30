"""Shared fixtures: synthetic OHLCV frames and an in-memory cache."""

from __future__ import annotations

import pandas as pd
import pytest

from swingcore.bitemporal_cache import Cache
from swingcore.synthetic import make_bars


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture
def cache() -> Cache:
    return Cache(":memory:")
