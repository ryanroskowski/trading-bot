from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger

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
        self.current_quarter: Optional[str] = None
        self.weight_history: List[Tuple[pd.Timestamp, pd.Series]] = []

    def _get_quarter(self, date: pd.Timestamp) -> str:
        """Get quarter string in format YYYY-Q."""
        return f"{date.year}-Q{date.quarter}"

    def _reset_quarter_tracking(self, date: pd.Timestamp) -> None:
        """Reset quarterly switch tracking if we've moved to a new quarter."""
        quarter = self._get_quarter(date)
        if self.current_quarter != quarter:
            self.current_quarter = quarter
            self.switch_count_in_quarter = 0
            logger.info(f"Meta-allocator: New quarter {quarter}, reset switch count")

    def _detect_meaningful_switch(self, prev_weights: pd.Series, new_weights: pd.Series, threshold: float = 0.1) -> bool:
        """Detect if there's a meaningful change in allocation weights."""
        if prev_weights is None:
            return True
        
        # Check if any weight changed by more than threshold
        diff = (new_weights - prev_weights).abs()
        return (diff > threshold).any()

    def _apply_weight_smoothing(self, prev_weights: pd.Series, new_weights: pd.Series, days_since_switch: int) -> pd.Series:
        """Apply exponential smoothing based on time since last switch."""
        if prev_weights is None:
            return new_weights
        
        # Adaptive decay based on time since last switch
        base_decay = 0.7
        if days_since_switch < 7:
            decay = 0.9  # High smoothing for recent switches
        elif days_since_switch < 21:
            decay = 0.8  # Medium smoothing
        else:
            decay = base_decay  # Normal smoothing
            
        return smooth_weights(prev_weights, new_weights, decay=decay)

    def step(self, date: pd.Timestamp, strategy_returns: Dict[str, pd.Series], etf_prices: pd.DataFrame) -> pd.Series:
        """
        Execute one step of meta-allocation with full constraint enforcement.
        
        Args:
            date: Current date
            strategy_returns: Dictionary of strategy returns series
            etf_prices: ETF price data for regime classification
            
        Returns:
            Series of strategy allocation weights
        """
        date = pd.Timestamp(date)
        
        # Reset quarterly tracking if needed
        self._reset_quarter_tracking(date)
        
        # Compute raw strategy scores and regime
        scores = score_strategies(strategy_returns, self.cfg.lookbacks, self.cfg.weights)
        regime = classify_regime(etf_prices)
        
        # Get raw allocation weights
        w_raw = allocate(scores, regime.label, self.cfg.blend_top_k)
        
        # Apply constraints and smoothing
        w_constrained = enforce_constraints(w_raw, min_weight=0.0, max_weight=0.60)
        
        # Calculate time since last switch
        days_since_switch = 0
        if self.last_switch_date is not None:
            days_since_switch = (date - self.last_switch_date).days
        
        # Check if we're in min_hold_days period
        in_min_hold_period = (
            self.last_switch_date is not None and 
            days_since_switch < self.cfg.min_hold_days
        )
        
        # Check if we're in cooldown period for new switches
        in_cooldown_period = (
            self.last_switch_date is not None and 
            days_since_switch < self.cfg.cooldown_days
        )
        
        # Check quarterly switch limit
        quarterly_limit_reached = (
            self.switch_count_in_quarter >= self.cfg.max_switches_per_quarter
        )
        
        # Apply smoothing
        w_smoothed = self._apply_weight_smoothing(self.prev_weights, w_constrained, days_since_switch)
        
        # Check if this would be a meaningful switch
        would_switch = self._detect_meaningful_switch(self.prev_weights, w_smoothed)
        
        # Decide whether to allow the switch
        final_weights = w_smoothed
        
        if would_switch:
            # Check all constraints
            constraints_violated = []
            
            if in_min_hold_period:
                constraints_violated.append(f"min_hold_days ({self.cfg.min_hold_days})")
            
            if in_cooldown_period:
                constraints_violated.append(f"cooldown_days ({self.cfg.cooldown_days})")
                
            if quarterly_limit_reached:
                constraints_violated.append(f"max_switches_per_quarter ({self.cfg.max_switches_per_quarter})")
            
            if constraints_violated:
                # Revert to previous weights with heavy smoothing
                logger.info(f"Meta-allocator switch blocked: {', '.join(constraints_violated)}")
                if self.prev_weights is not None:
                    final_weights = smooth_weights(self.prev_weights, w_smoothed, decay=0.95)
                else:
                    final_weights = w_smoothed
            else:
                # Allow the switch
                self.last_switch_date = date
                self.switch_count_in_quarter += 1
                logger.info(f"Meta-allocator switch executed on {date.date()}, "
                           f"quarter switches: {self.switch_count_in_quarter}/{self.cfg.max_switches_per_quarter}")
        
        # Store weight history for analysis (keep last 100 entries)
        self.weight_history.append((date, final_weights.copy()))
        if len(self.weight_history) > 100:
            self.weight_history.pop(0)
        
        # Update state
        self.prev_weights = final_weights.copy()
        
        logger.debug(f"Meta-allocator weights: {final_weights.to_dict()}")
        logger.debug(f"Regime: {regime.label}, Days since switch: {days_since_switch}")
        
        return final_weights

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the meta-allocator state."""
        return {
            'current_quarter': self.current_quarter,
            'switch_count_in_quarter': self.switch_count_in_quarter,
            'max_switches_per_quarter': self.cfg.max_switches_per_quarter,
            'last_switch_date': self.last_switch_date.isoformat() if self.last_switch_date else None,
            'days_since_last_switch': (pd.Timestamp.now() - self.last_switch_date).days if self.last_switch_date else None,
            'current_weights': self.prev_weights.to_dict() if self.prev_weights is not None else None,
            'weight_history_length': len(self.weight_history)
        }


