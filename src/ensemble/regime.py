from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class RegimeState:
    trend_breadth: float
    spy_vol_20d: float
    vix_level: float | None
    label: str  # risk_on | mixed | risk_off


def compute_trend_breadth(prices: pd.DataFrame) -> float:
    last = prices.iloc[-1]
    ma200 = prices.rolling(200).mean().iloc[-1]
    breadth = (last > ma200).mean()
    return float(breadth)


def classify_regime(etf_prices: pd.DataFrame, vix_level: float | None = None) -> RegimeState:
    spy = etf_prices["SPY"].pct_change()
    vol20 = spy.rolling(20).std(ddof=0).iloc[-1] * np.sqrt(252)
    breadth = compute_trend_breadth(etf_prices[[c for c in etf_prices.columns if c != "BIL"]])

    # VIX proxy via SPY vol if VIX unavailable
    vix = vix_level if vix_level is not None else float(vol20 * 16)  # rough proxy

    if breadth > 0.6 and vix < 22:
        label = "risk_on"
    elif breadth < 0.4 or vix > 28:
        label = "risk_off"
    else:
        label = "mixed"

    return RegimeState(trend_breadth=float(breadth), spy_vol_20d=float(vol20), vix_level=vix, label=label)


