from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.risk import realized_vol_annualized, vol_target_scale


def total_return(prices: pd.DataFrame, months: int) -> pd.Series:
    window = int(months * 21)
    return prices.pct_change(window).iloc[-1]


def next_bar_weights(weights_raw: pd.DataFrame, open_prices: pd.DataFrame) -> pd.DataFrame:
    # Shift signals to honor next-bar execution (weights decided at close, applied next open)
    return weights_raw.shift(1).reindex(open_prices.index).fillna(0.0)


def monthly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return rebalance dates aligned to the last available trading timestamp per month.

    Using calendar month-ends normalized to midnight can fail to intersect with
    session timestamps (e.g., 09:30). This implementation groups by year-month
    and takes the max index value actually present for each month.
    """
    if len(index) == 0:
        return pd.DatetimeIndex([])
    periods = index.to_period("M")
    # Get last timestamp present in the index for each month
    last_per_month = []
    for period in pd.PeriodIndex(periods.unique()).sort_values():
        mask = periods == period
        # Safety: ensure at least one entry
        if mask.any():
            last_per_month.append(index[mask][-1])
    return pd.DatetimeIndex(last_per_month)


def weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Fridays
    return pd.DatetimeIndex([d for d in index if d.weekday() == 5 - 1]).unique()


def apply_vol_targeting(weights: pd.DataFrame, returns: pd.DataFrame, target_vol_ann: float, window: int = 20) -> pd.DataFrame:
    # Estimate realized vol of portfolio based on component vols and weights path
    # Simple approximation: daily portfolio ret = sum_i w_i * r_i
    port_ret = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    port_vol_ann = realized_vol_annualized(port_ret, window=window)
    scale = vol_target_scale(port_vol_ann.iloc[-1] if len(port_vol_ann) else np.nan, target_vol_ann)
    return weights * scale


