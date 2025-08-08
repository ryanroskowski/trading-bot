from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from ..data.fundamentals import fetch_basic_fundamentals


@dataclass
class QVConfig:
    top_n: int
    rebalance: str  # monthly | weekly


def _quality_proxy(prices: pd.DataFrame) -> pd.Series:
    # Lower realized vol + persistent uptrend
    vol = prices.pct_change().rolling(60).std(ddof=0).iloc[-1]
    trend = prices.rolling(200).mean().iloc[-1]
    last = prices.iloc[-1]
    up = (last > trend).astype(float)
    score = up - vol.rank(pct=True)
    return score


def _value_proxy(prices: pd.DataFrame) -> pd.Series:
    # Fallback: recent drawdown recency as crude value stand-in
    rolling_max = prices.rolling(252).max().iloc[-1]
    dd = (prices.iloc[-1] / rolling_max - 1.0)
    # Higher negative dd -> cheaper; invert rank so deeper dd gets higher value score
    val = (-dd).rank(pct=True)
    return val


def compute_weights(prices: pd.DataFrame, cfg: QVConfig, universe: List[str]) -> pd.DataFrame:
    present = [t for t in universe if t in prices.columns]
    if len(present) == 0:
        # Nothing to do; return all zeros over the provided index
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    close = prices[present].dropna(how="all", axis=1)
    index = close.index
    # monthly only for simplicity; weekly optional but default monthly
    month_ends = pd.DatetimeIndex(sorted(set((idx + pd.offsets.BMonthEnd(0)).normalize() for idx in index)))
    dates = month_ends.intersection(index)

    w_df = pd.DataFrame(0.0, index=index, columns=close.columns)

    for d in dates:
        sub = close.loc[:d].dropna(how="any")
        if len(sub) < 252:
            continue
        qual = _quality_proxy(sub)
        val = _value_proxy(sub)
        composite = 0.5 * qual + 0.5 * val

        # trend filter: above 200d SMA
        trend_ok = (sub.iloc[-1] > sub.rolling(200).mean().iloc[-1])
        composite = composite[trend_ok]
        top = composite.sort_values(ascending=False).head(cfg.top_n)

        if len(top) < max(5, cfg.top_n // 2):
            # park remainder in BIL if present
            if "BIL" in w_df.columns:
                w_df.loc[d, "BIL"] = 1.0
            continue

        w = pd.Series(0.0, index=w_df.columns)
        w.loc[top.index] = 1.0 / len(top)
        w_df.loc[d, :] = w

    return w_df.replace(0.0, np.nan).ffill().fillna(0.0)


