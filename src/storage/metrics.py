from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass
class Metrics:
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_dd: float
    longest_dd_days: int
    stdev_ann: float
    exposure: float
    turnover: float | None
    psr: float | None
    dsr: float | None


def compute_report(returns: pd.Series) -> Metrics:
    r = returns.dropna()
    if r.empty:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, None, None, None)
    equity = (1 + r).cumprod()
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = float(equity.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else 0.0
    mu = r.mean() * TRADING_DAYS_PER_YEAR
    sigma = r.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = float(mu / sigma) if sigma > 0 else 0.0
    downside = r[r < 0].std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = float(mu / downside) if downside > 0 else 0.0
    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    max_dd = float(dd.min())
    longest_dd_days = int(_longest_drawdown_days(equity))
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    stdev_ann = sigma
    exposure = float((r != 0).mean())
    psr = _probabilistic_sharpe_ratio(r)
    dsr = _deflated_sharpe_ratio(r)
    return Metrics(cagr, sharpe, sortino, calmar, max_dd, longest_dd_days, stdev_ann, exposure, None, psr, dsr)


def _longest_drawdown_days(equity: pd.Series) -> int:
    peak = equity.cummax()
    underwater = equity < peak
    longest = 0
    current = 0
    for is_uw in underwater:
        if is_uw:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _probabilistic_sharpe_ratio(r: pd.Series, sr_bench: float = 0.0) -> float:
    # Bailey, López de Prado (2012)
    if r.std(ddof=0) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    n = len(r)
    k = (1 + sr**2 / 2) / (n - 1)
    z = (sr - sr_bench) * math.sqrt((n - 1) / (1 - k))
    # Convert Z to prob ~ N(0,1)
    return float(0.5 * (1 + math.erf(z / math.sqrt(2))))


def _deflated_sharpe_ratio(r: pd.Series, sr_bench: float = 0.0, num_trials: int = 10) -> float:
    # Simplified DSR assuming a small trials count; see López de Prado (2018)
    if r.std(ddof=0) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    # Adjust for multiple testing
    sr_deflated = sr - (0.5 * (num_trials - 1) / (len(r) - 1))
    return float(sr_deflated)


