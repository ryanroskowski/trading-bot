from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


@dataclass
class PriceData:
    close: pd.DataFrame
    open: pd.DataFrame


def fetch_yf_ohlcv(tickers: List[str], start: str = "2005-01-01", end: Optional[str] = None) -> PriceData:
    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="ticker")

    # yfinance returns a Panel-like MultiIndex; normalize to DataFrames
    frames_close = {}
    frames_open = {}
    for t in tickers:
        df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
        frames_close[t] = df["Adj Close"].rename(t)
        frames_open[t] = df["Open"].rename(t)
    close = pd.concat(frames_close.values(), axis=1).sort_index()
    open_ = pd.concat(frames_open.values(), axis=1).sort_index()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    open_.index = pd.DatetimeIndex(open_.index).tz_localize(None)
    return PriceData(close=close, open=open_)


def align_next_bar_execution(signals: pd.DataFrame, open_prices: pd.DataFrame) -> pd.DataFrame:
    # Shift signals one bar forward to ensure next-open execution without lookahead
    return signals.shift(1).reindex(open_prices.index)


