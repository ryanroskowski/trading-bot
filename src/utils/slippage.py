from __future__ import annotations

import numpy as np
import pandas as pd


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    dw = weights.diff().abs().sum(axis=1)
    dw.iloc[0] = weights.iloc[0].abs().sum()
    return dw


def costs_from_turnover(turnover: pd.Series, bps: float) -> pd.Series:
    # cost as fraction of equity per day
    return turnover * (bps / 1e4)


def apply_costs_to_returns(portfolio_returns: pd.Series, weights: pd.DataFrame, bps: float) -> pd.Series:
    turnover = compute_turnover(weights)
    costs = costs_from_turnover(turnover, bps)
    after_cost = portfolio_returns - costs
    return after_cost


