from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.slippage import (
    compute_turnover,
    compute_turnover_by_symbol,
    costs_from_turnover_by_symbol,
    apply_symbol_costs_to_returns,
)


def test_turnover_and_symbol_costs_basic():
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    cols = ["SPY", "AAPL"]
    w = pd.DataFrame(
        [
            [0.0, 0.0],
            [0.5, 0.2],
            [0.5, 0.0],
            [0.3, 0.2],
            [0.3, 0.2],
        ],
        index=dates,
        columns=cols,
    )

    # Per-symbol turnover
    per_sym = compute_turnover_by_symbol(w)
    assert per_sym.shape == w.shape
    # First day equals abs(weights)
    assert abs(per_sym.iloc[0].sum() - w.iloc[0].abs().sum()) < 1e-12

    # Costs with per-symbol bps
    bps = pd.Series({"SPY": 3.0, "AAPL": 10.0})
    daily_costs = costs_from_turnover_by_symbol(per_sym, bps)
    assert (daily_costs >= 0).all()

    # Apply to returns
    r = pd.Series([0.0, 0.01, -0.005, 0.002, 0.0], index=dates)
    after = apply_symbol_costs_to_returns(r, w, bps)
    # Costs reduce returns (except possibly days with zero turnover)
    assert (after <= r + 1e-12).all()


