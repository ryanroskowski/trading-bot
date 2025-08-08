from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.risk import realized_vol_annualized
from .common import monthly_rebalance_dates, weekly_rebalance_dates, apply_vol_targeting


@dataclass
class VMDMConfig:
    lookbacks_months: List[int]
    rebalance: str  # "monthly" or "weekly"
    abs_filter: bool
    target_vol_annual: float


def compute_weights(prices: pd.DataFrame, cfg: VMDMConfig, bonds_symbol: str = "IEF", cash_symbol: str = "BIL") -> pd.DataFrame:
    close = prices
    returns = close.pct_change().fillna(0.0)
    index = close.index

    if cfg.rebalance == "weekly":
        dates = weekly_rebalance_dates(index)
    else:
        dates = monthly_rebalance_dates(index)
    dates = dates.intersection(index)

    # Compute momentum scores
    def rel_mom(series: pd.Series) -> float:
        vals = []
        for m in cfg.lookbacks_months:
            w = int(m * 21)
            if len(series) <= w:
                vals.append(np.nan)
            else:
                vals.append(series.pct_change(w).iloc[-1])
        arr = np.array(vals, dtype=float)
        if np.isnan(arr).all():
            return np.nan
        return float(np.nanmean(arr))

    tickers = [t for t in ["SPY", "EFA", bonds_symbol, cash_symbol] if t in close.columns]
    w_df = pd.DataFrame(0.0, index=index, columns=tickers)

    for d in dates:
        if d not in index:
            continue
        sub = close.loc[:d]
        spy_m = rel_mom(sub["SPY"]) if "SPY" in sub else np.nan
        efa_m = rel_mom(sub["EFA"]) if "EFA" in sub else np.nan
        bonds_m = rel_mom(sub[bonds_symbol]) if bonds_symbol in sub else np.nan
        winner = None
        if np.nan_to_num(spy_m) >= np.nan_to_num(efa_m):
            winner = "SPY" if "SPY" in close.columns else None
            win_m = spy_m
        else:
            winner = "EFA" if "EFA" in close.columns else None
            win_m = efa_m

        target = None
        if cfg.abs_filter and (np.isnan(win_m) or win_m <= 0.0):
            # go to bonds, else cash
            target = bonds_symbol if bonds_symbol in tickers else cash_symbol
            if target not in tickers:
                target = None
        else:
            target = winner

        if target is None:
            continue
        w_df.loc[d, :] = 0.0
        w_df.loc[d, target] = 1.0

    w_df = w_df.replace(0.0, np.nan).ffill().fillna(0.0)
    # Vol targeting applied later at portfolio level; here we return base weights
    return w_df


