from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .common import monthly_rebalance_dates


@dataclass
class CarryConfig:
    rebalance: str  # monthly only for now
    universe: List[str]


def compute_weights(prices: pd.DataFrame, cfg: CarryConfig) -> pd.DataFrame:
    close = prices
    index = close.index
    dates = monthly_rebalance_dates(index)
    dates = dates.intersection(index)

    # Simple carry proxies using trailing 12m total return as a stand-in
    w_df = pd.DataFrame(0.0, index=index, columns=[t for t in cfg.universe if t in close.columns])
    if w_df.shape[1] == 0:
        return w_df

    look = 252
    for d in dates:
        sub = close.loc[:d]
        if len(sub) <= look:
            continue
        mom = sub.pct_change(look).iloc[-1]
        # Long top half by 12m return (proxy for carry), inverse vol weights
        top = mom.sort_values(ascending=False).head(max(1, len(mom) // 2)).index.tolist()
        vols = sub[top].pct_change().rolling(60).std(ddof=0).iloc[-1]
        inv = 1.0 / vols.replace(0, np.nan)
        inv = inv.fillna(0.0)
        if inv.sum() == 0:
            eq = pd.Series(1.0, index=top)
            weights = eq / eq.sum()
        else:
            weights = inv / inv.sum()
        w_df.loc[d, :] = 0.0
        w_df.loc[d, weights.index] = weights.values

    return w_df.replace(0.0, np.nan).ffill().fillna(0.0)


