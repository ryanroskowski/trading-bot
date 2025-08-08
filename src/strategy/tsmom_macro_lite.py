from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .common import monthly_rebalance_dates, weekly_rebalance_dates


@dataclass
class TSMOMConfig:
    lookback_months: int
    rebalance: str  # monthly | weekly


def compute_weights(prices: pd.DataFrame, cfg: TSMOMConfig, universe: List[str] | None = None) -> pd.DataFrame:
    close = prices
    index = close.index
    if universe is None:
        universe = [t for t in ["SPY", "TLT", "IEF", "DBC", "UUP", "BIL"] if t in close.columns]

    if cfg.rebalance == "weekly":
        dates = weekly_rebalance_dates(index)
    else:
        dates = monthly_rebalance_dates(index)
    dates = dates.intersection(index)

    w = int(cfg.lookback_months * 21)
    w_df = pd.DataFrame(0.0, index=index, columns=universe)

    for d in dates:
        sub = close.loc[:d]
        mom = sub.pct_change(w).iloc[-1]
        long_only = mom[mom > 0].index.tolist()
        if len(long_only) == 0:
            # park in BIL if available
            if "BIL" in universe:
                w_df.loc[d, "BIL"] = 1.0
            continue
        # inverse vol weights (use 60-day ex-post as proxy)
        vols = sub[long_only].pct_change().rolling(60).std(ddof=0).iloc[-1]
        inv = 1.0 / vols.replace(0, np.nan)
        inv = inv.fillna(0.0)
        if inv.sum() == 0:
            eq = pd.Series(1.0, index=long_only)
            weights = eq / eq.sum()
        else:
            weights = inv / inv.sum()
        w_df.loc[d, :] = 0.0
        w_df.loc[d, weights.index] = weights.values

    return w_df.replace(0.0, np.nan).ffill().fillna(0.0)


