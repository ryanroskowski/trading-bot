from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .common import monthly_rebalance_dates


@dataclass
class CryptoETFTrendConfig:
    rebalance: str  # "monthly"
    lookback_months: int  # e.g., 6 or 12
    universe: List[str]  # e.g., ["IBIT", "FBTC", "BITO"]
    cash: str = "BIL"
    max_weight: float = 0.10  # optional composite cap suggestion; enforced in portfolio later


def compute_weights(prices: pd.DataFrame, cfg: CryptoETFTrendConfig) -> pd.DataFrame:
    """Compute a simple absolute-momentum trend sleeve on crypto ETFs.

    - Monthly rebalance on last business day
    - If trailing lookback return <= 0 for an ETF, that ETF gets 0 weight
    - Allocate inverse-vol among positive-return ETFs
    - If none are positive, park to cash symbol (e.g., BIL)
    - Output a daily index aligned DataFrame with weights forward-filled between rebalances
    """
    close = prices
    if close.empty:
        return pd.DataFrame(index=close.index)

    # Filter to instruments present
    symbols = [s for s in cfg.universe if s in close.columns]
    if not symbols:
        # Nothing tradable; return zeros but include cash if present
        cols = [cfg.cash] if cfg.cash in close.columns else []
        return pd.DataFrame(0.0, index=close.index, columns=cols)

    lookback = int(round(cfg.lookback_months * 21))
    dates = monthly_rebalance_dates(close.index)
    dates = dates.intersection(close.index)

    # Precompute vol for inverse-vol weighting
    daily_rets = close[symbols].pct_change().fillna(0.0)
    vol = daily_rets.rolling(63, min_periods=20).std(ddof=0)

    w = pd.DataFrame(0.0, index=close.index, columns=list(set(symbols + ([cfg.cash] if cfg.cash in close.columns else []))))

    for d in dates:
        sub = close.loc[:d, symbols]
        if len(sub) <= lookback + 1:
            continue
        # trailing return
        r = sub.iloc[-1] / sub.iloc[-lookback - 1] - 1.0
        pos = r[r > 0].index.tolist()
        if not pos:
            # All negative -> cash if present
            if cfg.cash in w.columns:
                w.loc[d, cfg.cash] = 1.0
            continue
        v = vol.loc[d, pos]
        v = v.replace(0.0, np.nan).fillna(v[v > 0].median() if (v > 0).any() else 1.0)
        inv = 1.0 / v
        ww = inv / inv.sum()
        # Write row at rebalance date
        w.loc[d, pos] = ww.values

    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # Forward-fill rebalances to daily index
    w = w.where(w.ne(0.0)).ffill().fillna(0.0)
    # Ensure rows sum <= 1 (no leverage here)
    row_sum = w.sum(axis=1)
    over = row_sum > 1.0
    if over.any():
        w.loc[over] = w.loc[over].div(row_sum[over], axis=0)
    return w


