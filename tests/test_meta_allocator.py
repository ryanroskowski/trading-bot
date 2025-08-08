from __future__ import annotations

import numpy as np
import pandas as pd

from src.ensemble.meta_allocator import MetaAllocator, MetaConfig
import os


def test_meta_weights_normalize():
    dates = pd.date_range("2022-01-01", periods=200, freq="B")
    r1 = pd.Series(np.random.normal(0.0004, 0.01, len(dates)), index=dates)
    r2 = pd.Series(np.random.normal(0.0002, 0.01, len(dates)), index=dates)
    r3 = pd.Series(np.random.normal(0.0003, 0.01, len(dates)), index=dates)
    etfs = pd.DataFrame({"SPY": (1 + r1).cumprod(), "EFA": (1 + r2).cumprod(), "BIL": (1 + 0*r3).cumprod()}, index=dates)
    cfg = MetaConfig(enabled=True, method="ewma-sharpe", lookbacks=[63, 126, 252], weights=[0.5, 0.3, 0.2], cooldown_days=21, min_hold_days=21, max_switches_per_quarter=2, blend_top_k=2)
    meta = MetaAllocator(cfg)
    w = meta.step(dates[-1], {"vm_dm": r1, "tsmom": r2, "qv_trend": r3}, etfs)
    assert abs(w.sum() - 1.0) < 1e-6


