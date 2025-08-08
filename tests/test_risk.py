from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.risk import realized_vol_annualized, vol_target_scale, cap_total_allocation


def test_vol_target_scale_basic():
    scale = vol_target_scale(0.20, 0.10)
    assert 0 < scale < 1


def test_realized_vol():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 100))
    vol = realized_vol_annualized(r, window=20)
    assert vol.iloc[-1] > 0


def test_cap_total_allocation():
    w = pd.Series({"A": 0.5, "B": 0.5, "C": 0.5})
    capped = cap_total_allocation(w, 0.6)
    assert capped.sum() <= 0.6000001


