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


def test_meta_allocator_risk_parity_method():
    dates = pd.date_range("2022-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    # lower vol vs higher vol strat
    r_low = pd.Series(rng.normal(0.0003, 0.005, len(dates)), index=dates)
    r_high = pd.Series(rng.normal(0.0003, 0.015, len(dates)), index=dates)
    etfs = pd.DataFrame({"SPY": 100 * (1 + 0.0002) ** np.arange(len(dates))}, index=dates)
    cfg = MetaConfig(enabled=True, method="risk_parity", lookbacks=[63], weights=[1.0], cooldown_days=5, min_hold_days=5, max_switches_per_quarter=10, blend_top_k=2)
    meta = MetaAllocator(cfg)
    w = meta.step(dates[-1], {"low": r_low, "high": r_high}, etfs)
    assert w.get("low", 0.0) >= w.get("high", 0.0)


def test_meta_allocator_hrp_method():
    dates = pd.date_range("2022-01-01", periods=200, freq="B")
    rng = np.random.default_rng(1)
    r_a = pd.Series(rng.normal(0.0003, 0.010, len(dates)), index=dates)
    r_b = pd.Series(rng.normal(0.0003, 0.010, len(dates)), index=dates)
    r_c = pd.Series(rng.normal(0.0003, 0.010, len(dates)), index=dates)
    etfs = pd.DataFrame({"SPY": 100 * (1 + 0.0002) ** np.arange(len(dates))}, index=dates)
    cfg = MetaConfig(enabled=True, method="hrp", lookbacks=[63], weights=[1.0], cooldown_days=5, min_hold_days=5, max_switches_per_quarter=10, blend_top_k=3)
    meta = MetaAllocator(cfg)
    w = meta.step(dates[-1], {"A": r_a, "B": r_b, "C": r_c}, etfs)
    # HRP should produce a valid normalized weighting across all strategies
    assert abs(w.sum() - 1.0) < 1e-6

