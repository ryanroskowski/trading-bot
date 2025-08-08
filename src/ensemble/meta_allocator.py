from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .regime import classify_regime


@dataclass
class MetaConfig:
    enabled: bool
    method: str  # "ewma-sharpe"
    lookbacks: List[int]
    weights: List[float]
    cooldown_days: int
    min_hold_days: int
    max_switches_per_quarter: int
    blend_top_k: int


def ewma_sharpe(returns: pd.Series, halflife: int = 63) -> float:
    if returns.dropna().empty:
        return 0.0
    w = 0.5 ** (np.arange(len(returns))[::-1] / halflife)
    w = w / w.sum()
    r = returns.values
    mu = (r * w).sum()
    sigma = np.sqrt(((r - mu) ** 2 * w).sum())
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float(mu / sigma * np.sqrt(252))


def score_strategies(daily_returns: Dict[str, pd.Series], lookbacks: List[int], weights: List[float]) -> pd.Series:
    scores = {}
    for name, ret in daily_returns.items():
        win_scores = []
        for lb, w in zip(lookbacks, weights):
            s = ewma_sharpe(ret.tail(lb), halflife=max(10, lb // 2))
            win_scores.append(w * s)
        scores[name] = float(np.nansum(win_scores))
    return pd.Series(scores).sort_values(ascending=False)


def allocate(scores: pd.Series, regime_label: str, top_k: int) -> pd.Series:
    # Base: normalize top_k scores to sum 1
    top = scores.head(top_k)
    if top.abs().sum() == 0:
        w = pd.Series(0.0, index=scores.index)
        if len(top) > 0:
            w.loc[top.index] = 1.0 / len(top)
        return w
    w = (top - top.min()).clip(lower=0)  # shift to non-negative
    if w.sum() == 0:
        w = pd.Series(1.0, index=top.index)
    w = w / w.sum()

    # Regime tilt
    if regime_label == "risk_on":
        tilt = {"vm_dm": 0.4, "tsmom": 0.4, "qv_trend": 0.2}
    elif regime_label == "risk_off":
        tilt = {"vm_dm": 0.6, "tsmom": 0.2, "qv_trend": 0.2}
    else:  # mixed
        tilt = {"vm_dm": 0.4, "tsmom": 0.3, "qv_trend": 0.3}

    # Map scores index to canonical strategy keys
    canon = {n: ("vm_dm" if "vm" in n else ("tsmom" if "tsmom" in n else ("qv_trend" if "qv" in n else n))) for n in w.index}
    tilted = {}
    for n, val in w.items():
        key = canon[n]
        tilted[n] = float(val * tilt.get(key, 1.0))
    tilted = pd.Series(tilted)
    if tilted.sum() > 0:
        tilted = tilted / tilted.sum()
    return tilted


def enforce_constraints(weights: pd.Series, min_weight: float = 0.0, max_weight: float = 0.60) -> pd.Series:
    w = weights.clip(lower=min_weight, upper=max_weight)
    if w.sum() > 0:
        w = w / w.sum()
    return w


def smooth_weights(prev: pd.Series | None, new: pd.Series, decay: float = 0.7) -> pd.Series:
    if prev is None:
        return new
    idx = new.index.union(prev.index)
    prev = prev.reindex(idx).fillna(0.0)
    new = new.reindex(idx).fillna(0.0)
    sm = decay * prev + (1 - decay) * new
    if sm.sum() > 0:
        sm = sm / sm.sum()
    return sm


class MetaAllocator:
    def __init__(self, cfg: MetaConfig):
        self.cfg = cfg
        self.prev_weights: pd.Series | None = None
        self.last_switch_date: pd.Timestamp | None = None
        self.switch_count_in_quarter: int = 0

    def step(self, date: pd.Timestamp, strategy_returns: Dict[str, pd.Series], etf_prices: pd.DataFrame) -> pd.Series:
        scores = score_strategies(strategy_returns, self.cfg.lookbacks, self.cfg.weights)
        regime = classify_regime(etf_prices)
        w_raw = allocate(scores, regime.label, self.cfg.blend_top_k)
        # enforce min_hold_days and cooldown_days with simple damping
        if self.last_switch_date is not None:
            days_since = (pd.Timestamp(date) - pd.Timestamp(self.last_switch_date)).days
            if days_since < self.cfg.min_hold_days:
                # Heavily smooth to previous weights to reduce switching
                w_raw = smooth_weights(self.prev_weights, w_raw, decay=0.9)
        w_c = enforce_constraints(w_raw, min_weight=0.0, max_weight=0.60)
        w_s = smooth_weights(self.prev_weights, w_c, decay=0.7)
        # detect switch events (argmax change)
        if self.prev_weights is not None:
            prev_top = self.prev_weights.idxmax()
            new_top = w_s.idxmax()
            if prev_top != new_top:
                # cooldown
                if self.last_switch_date is not None:
                    days_since = (pd.Timestamp(date) - pd.Timestamp(self.last_switch_date)).days
                    if days_since < self.cfg.cooldown_days:
                        # revert to previous to respect cooldown
                        w_s = self.prev_weights
                    else:
                        self.last_switch_date = pd.Timestamp(date)
                        # TODO: track per-quarter cap; simplified here
                else:
                    self.last_switch_date = pd.Timestamp(date)
        self.prev_weights = w_s
        return w_s


