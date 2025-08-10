from __future__ import annotations

import numpy as np
import pandas as pd


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Compute total portfolio turnover per day as sum of absolute weight changes.

    Turnover_t = sum_i |w_{t,i} - w_{t-1,i}|, with first day equal to sum |w_{0,i}|.
    """
    if weights.empty:
        return pd.Series(dtype=float)
    dw = weights.diff().abs().sum(axis=1)
    # Handle first row explicitly
    first_idx = weights.index[0]
    if first_idx in dw.index:
        dw.loc[first_idx] = weights.iloc[0].abs().sum()
    return dw


def compute_turnover_by_symbol(weights: pd.DataFrame) -> pd.DataFrame:
    """Compute per-symbol absolute weight change per day.

    Returns a DataFrame aligned to weights with abs(w_t - w_{t-1}). First row is abs(w_0).
    """
    if weights.empty:
        return pd.DataFrame(index=weights.index, columns=weights.columns, dtype=float)
    dw = weights.diff().abs()
    first_idx = weights.index[0]
    if first_idx in dw.index:
        dw.loc[first_idx] = weights.iloc[0].abs()
    return dw.fillna(0.0)


def costs_from_turnover(turnover: pd.Series, bps: float) -> pd.Series:
    """Legacy aggregate cost: turnover * bps/1e4 as fraction of equity per day."""
    return turnover * (bps / 1e4)


def costs_from_turnover_by_symbol(turnover_by_symbol: pd.DataFrame, bps_by_symbol: pd.Series) -> pd.Series:
    """Compute daily cost using per-symbol bps.

    Args:
        turnover_by_symbol: abs weight change per symbol per day.
        bps_by_symbol: Series mapping symbol -> bps. Unknown symbols default to median of provided bps.

    Returns:
        Series of daily cost as fraction of equity.
    """
    if turnover_by_symbol.empty:
        return pd.Series(0.0, index=turnover_by_symbol.index)
    # Align and fill missing bps with median
    bps_map = bps_by_symbol.copy()
    if bps_map.empty:
        # default to 3 bps if nothing provided
        bps_map = pd.Series(3.0, index=turnover_by_symbol.columns)
    # Broadcast bps across columns
    bps_broadcast = pd.Series(bps_map, index=turnover_by_symbol.columns).fillna(bps_map.median())
    # Daily costs: sum_i turnover_i * (bps_i/1e4)
    daily = (turnover_by_symbol * (bps_broadcast / 1e4)).sum(axis=1)
    return daily.fillna(0.0)


def apply_costs_to_returns(portfolio_returns: pd.Series, weights: pd.DataFrame, bps: float) -> pd.Series:
    """Apply flat bps cost to portfolio returns given weights path (aggregate model)."""
    if portfolio_returns.empty or weights.empty:
        return portfolio_returns
    turnover = compute_turnover(weights)
    costs = costs_from_turnover(turnover, bps)
    after_cost = portfolio_returns - costs
    return after_cost


def apply_symbol_costs_to_returns(
    portfolio_returns: pd.Series,
    weights: pd.DataFrame,
    bps_by_symbol: pd.Series,
) -> pd.Series:
    """Apply per-symbol bps costs to portfolio returns.

    Costs are computed from per-symbol turnover and subtracted from daily returns.
    """
    if portfolio_returns.empty or weights.empty:
        return portfolio_returns
    turn_by_sym = compute_turnover_by_symbol(weights)
    daily_costs = costs_from_turnover_by_symbol(turn_by_sym, bps_by_symbol)
    return (portfolio_returns - daily_costs).reindex_like(portfolio_returns).fillna(0.0)


