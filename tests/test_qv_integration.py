from __future__ import annotations

import numpy as np
import pandas as pd
from types import SimpleNamespace
from unittest.mock import patch
from src.config import load_config as real_load_config

from src.engine import backtest as bt


def test_qv_weights_flow_into_composite():
    # ETF prices
    dates = pd.date_range("2022-01-01", periods=120, freq="B")
    idx = np.arange(len(dates))
    close_etf = pd.DataFrame({
        "SPY": 100 * (1 + 0.0005) ** idx,
        "EFA": 100 * (1 + 0.0003) ** idx,
        "IEF": 100 * (1 + 0.0001) ** idx,
        "BIL": 100.0,
    }, index=dates)
    open_etf = close_etf.shift(1).bfill()

    # Stock universe (present in large_cap_csv)
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "BRK.B", "XOM", "JPM", "UNH"]
    rng = np.random.default_rng(0)
    close_stk = pd.DataFrame({s: 50 + np.cumsum(rng.normal(0, 0.5, len(dates))) for s in stocks}, index=dates)
    open_stk = close_stk.shift(1).bfill()

    # Patch fetcher to return ETFs for ETF list, and stocks for large caps
    def fake_fetch(tickers, start=None, end=None):
        if set(tickers) <= set(close_etf.columns):
            return SimpleNamespace(close=close_etf, open=open_etf)
        else:
            subset = [t for t in tickers if t in close_stk.columns]
            return SimpleNamespace(close=close_stk[subset], open=open_stk[subset])

    def patched_load_config(path=None):
        cfg = real_load_config(path)
        # Restrict ETFs to ones provided in close_etf so the fake fetch returns data
        cfg["universe"]["etfs"] = list(close_etf.columns)
        # Force QV to use actual stocks to validate integration of stock symbols
        cfg["strategies"]["qv_trend"]["use_etf_proxies"] = False
        return cfg

    with patch('src.engine.backtest.load_config', side_effect=patched_load_config):
        with patch('src.engine.backtest.fetch_yf_ohlcv', side_effect=fake_fetch):
            with patch('pandas.read_csv', return_value=pd.DataFrame({'Ticker': stocks})):
                res = bt.run_backtest()

    # Composite weights should include some stock columns if QV was active
    has_stock_cols = any(c in res.weights.columns for c in stocks)
    assert has_stock_cols, "Composite weights should include QV-selected stock symbols"


