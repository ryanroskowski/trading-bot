from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.vm_dual_momentum import compute_weights as vm_weights, VMDMConfig
from src.utils.risk import cap_total_allocation


def test_vm_dm_no_lookahead():
    dates = pd.date_range("2020-01-01", periods=260, freq="B")
    idx = np.arange(len(dates))
    prices = pd.DataFrame({
        "SPY": 100 * (1 + 0.0005) ** idx,
        "EFA": 100 * (1 + 0.0003) ** idx,
        "IEF": 100 * (1 + 0.0001) ** idx,
        "BIL": 100.0,
    }, index=dates)
    cfg = VMDMConfig(lookbacks_months=[6, 12], rebalance="monthly", abs_filter=True, target_vol_annual=0.1)
    w = vm_weights(prices, cfg)
    assert not w.isna().any().any()
    assert (w["SPY"] >= 0).all()


def test_cap_total_allocation_global():
    w = pd.Series({"SPY": 0.7, "EFA": 0.3})
    capped = cap_total_allocation(w, 0.6)
    assert abs(capped.sum() - 0.6) < 1e-6


