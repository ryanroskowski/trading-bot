from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine import backtest as bt
from types import SimpleNamespace


def test_smoke_backtest_runs(monkeypatch):
    # Provide a tiny synthetic dataset to avoid network
    dates = pd.date_range("2022-01-01", periods=180, freq="B")
    idx = np.arange(len(dates))
    close = pd.DataFrame({
        "SPY": 100 * (1 + 0.0004) ** idx,
        "EFA": 100 * (1 + 0.0002) ** idx,
        "IEF": 100 * (1 + 0.0001) ** idx,
        "BIL": 100.0,
    }, index=dates)
    open_ = close.shift(1).bfill()
    price = SimpleNamespace(close=close, open=open_)

    # Patch both ETF and large-cap fetches to return ETFs-only data during smoke test
    monkeypatch.setattr(bt, "fetch_yf_ohlcv", lambda tickers, start=None, end=None: price)
    res = bt.run_backtest()
    assert not res.daily_returns.empty
    assert res.equity_curve.iloc[-1] > 0


