from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def realized_vol_annualized(returns: pd.Series | pd.DataFrame, window: int = 20) -> pd.Series:
    rolling_std = returns.rolling(window).std(ddof=0)
    return rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)


def vol_target_scale(portfolio_vol_ann: float, target_vol_ann: float) -> float:
    if portfolio_vol_ann <= 0 or np.isnan(portfolio_vol_ann):
        return 0.0
    return float(np.clip(target_vol_ann / portfolio_vol_ann, 0.0, 10.0))


def apply_drawdown_derisking(equity_curve: pd.Series, threshold_pct: float, scale: float) -> float:
    peak = equity_curve.cummax()
    dd = (equity_curve / peak - 1.0).iloc[-1] * -100.0
    if dd >= threshold_pct:
        return float(scale)
    return 1.0


def cap_total_allocation(weights: pd.Series, cap: float) -> pd.Series:
    total = weights.clip(lower=0).sum()
    if total <= cap or total == 0:
        return weights
    return weights * (cap / total)


def enforce_max_positions(weights: pd.Series, max_positions: int) -> pd.Series:
    non_zero = weights[weights.abs() > 1e-9]
    if len(non_zero) <= max_positions:
        return weights
    # Keep largest absolute weights
    keep = non_zero.abs().sort_values(ascending=False).head(max_positions).index
    result = weights.copy()
    result.loc[~result.index.isin(keep)] = 0.0
    return result


def pdt_guard(estimated_round_trips_today: int, account_equity_usd: float) -> Tuple[bool, str]:
    if account_equity_usd >= 25000:
        return True, "PDT OK"
    if estimated_round_trips_today >= 3:
        return False, "PDT guard triggered (<$25k equity and >=3 round-trips)"
    return True, "PDT OK"


