from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def realized_vol_annualized(returns: pd.Series | pd.DataFrame, window: int = 20) -> pd.Series:
    rolling_std = returns.rolling(window).std(ddof=0)
    return rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)


def vol_target_scale(portfolio_vol_ann: float, target_vol_ann: float) -> float:
    if portfolio_vol_ann <= 0 or np.isnan(portfolio_vol_ann):
        return 0.0
    return float(np.clip(target_vol_ann / portfolio_vol_ann, 0.0, 10.0))


def scale_to_target_vol(daily_returns: pd.Series, target_vol_ann: float, window: int = 20) -> float:
    """Compute a bounded scaling factor to reach target annualized volatility.

    Uses a rolling window estimate of realized volatility; clamps the scale
    between 0.2x and 1.5x to avoid abrupt swings during live trading.
    """
    if daily_returns is None or daily_returns.empty:
        return 1.0
    try:
        vol_ann_series = realized_vol_annualized(daily_returns, window=window)
        vol_ann = float(vol_ann_series.iloc[-1])
        if vol_ann <= 1e-12 or np.isnan(vol_ann):
            return 1.0
        scale = float(target_vol_ann / vol_ann)
        return float(np.clip(scale, 0.2, 1.5))
    except Exception:
        return 1.0


def apply_drawdown_derisking(equity_curve: pd.Series, threshold_pct: float, scale: float) -> float:
    peak = equity_curve.cummax()
    dd = (equity_curve / peak - 1.0).iloc[-1] * -100.0
    if dd >= threshold_pct:
        return float(scale)
    return 1.0


def cap_total_allocation(weights: pd.Series, cap: float) -> pd.Series:
    total = weights.clip(lower=0).sum()
    if total <= cap or total == 0:
        return weights
    return weights * (cap / total)


def enforce_max_positions(weights: pd.Series, max_positions: int) -> pd.Series:
    non_zero = weights[weights.abs() > 1e-9]
    if len(non_zero) <= max_positions:
        return weights
    # Keep largest absolute weights
    keep = non_zero.abs().sort_values(ascending=False).head(max_positions).index
    result = weights.copy()
    result.loc[~result.index.isin(keep)] = 0.0
    return result


def pdt_guard(estimated_round_trips_today: int, account_equity_usd: float) -> Tuple[bool, str]:
    if account_equity_usd >= 25000:
        return True, "PDT OK"
    if estimated_round_trips_today >= 3:
        return False, "PDT guard triggered (<$25k equity and >=3 round-trips)"
    return True, "PDT OK"


def check_daily_loss_circuit_breaker(
    current_equity: float, 
    start_of_day_equity: float, 
    threshold_pct: float = 5.0
) -> Tuple[bool, str]:
    """Check if daily loss exceeds circuit breaker threshold."""
    if start_of_day_equity <= 0:
        return False, "Invalid start of day equity"
    
    daily_pnl_pct = (current_equity - start_of_day_equity) / start_of_day_equity * 100
    
    if daily_pnl_pct < -threshold_pct:
        return True, f"Daily loss circuit breaker triggered: {daily_pnl_pct:.2f}% < -{threshold_pct:.2f}%"
    
    return False, f"Daily PnL: {daily_pnl_pct:.2f}%"


def check_rolling_drawdown_circuit_breaker(
    equity_curve: pd.Series, 
    max_drawdown_pct: float = 20.0,
    lookback_days: int = 90
) -> Tuple[bool, str]:
    """Check if rolling drawdown exceeds circuit breaker threshold."""
    if equity_curve.empty:
        return False, "No equity data"
    
    # Use last N days or all available data
    recent_equity = equity_curve.tail(lookback_days)
    
    # Calculate rolling max and current drawdown
    peak = recent_equity.cummax()
    current_dd_pct = (recent_equity.iloc[-1] / peak.iloc[-1] - 1.0) * 100
    
    if current_dd_pct < -max_drawdown_pct:
        return True, f"Rolling drawdown circuit breaker triggered: {current_dd_pct:.2f}% < -{max_drawdown_pct:.2f}%"
    
    return False, f"Current drawdown: {current_dd_pct:.2f}%"


def check_volatility_circuit_breaker(
    returns: pd.Series,
    max_vol_annual: float = 50.0,
    window: int = 20
) -> Tuple[bool, str]:
    """Check if recent volatility exceeds circuit breaker threshold."""
    if returns.empty or len(returns) < window:
        return False, "Insufficient return data"
    
    recent_vol = realized_vol_annualized(returns, window).iloc[-1] * 100
    
    if recent_vol > max_vol_annual:
        return True, f"Volatility circuit breaker triggered: {recent_vol:.2f}% > {max_vol_annual:.2f}%"
    
    return False, f"Current volatility: {recent_vol:.2f}%"


@dataclass
class CircuitBreakerStatus:
    """Container for circuit breaker status."""
    daily_loss_triggered: bool
    daily_loss_message: str
    drawdown_triggered: bool
    drawdown_message: str
    volatility_triggered: bool
    volatility_message: str
    
    @property
    def any_triggered(self) -> bool:
        return self.daily_loss_triggered or self.drawdown_triggered or self.volatility_triggered
    
    def get_summary(self) -> str:
        if not self.any_triggered:
            return "All circuit breakers OK"
        
        messages = []
        if self.daily_loss_triggered:
            messages.append(self.daily_loss_message)
        if self.drawdown_triggered:
            messages.append(self.drawdown_message)
        if self.volatility_triggered:
            messages.append(self.volatility_message)
            
        return "; ".join(messages)


def check_all_circuit_breakers(
    current_equity: float,
    start_of_day_equity: float,
    equity_curve: pd.Series,
    returns: pd.Series,
    daily_loss_threshold: float = 5.0,
    max_drawdown_threshold: float = 20.0,
    max_volatility_threshold: float = 50.0
) -> CircuitBreakerStatus:
    """Check all circuit breakers and return comprehensive status."""
    
    daily_loss_triggered, daily_loss_msg = check_daily_loss_circuit_breaker(
        current_equity, start_of_day_equity, daily_loss_threshold
    )
    
    drawdown_triggered, drawdown_msg = check_rolling_drawdown_circuit_breaker(
        equity_curve, max_drawdown_threshold
    )
    
    volatility_triggered, volatility_msg = check_volatility_circuit_breaker(
        returns, max_volatility_threshold
    )
    
    return CircuitBreakerStatus(
        daily_loss_triggered=daily_loss_triggered,
        daily_loss_message=daily_loss_msg,
        drawdown_triggered=drawdown_triggered,
        drawdown_message=drawdown_msg,
        volatility_triggered=volatility_triggered,
        volatility_message=volatility_msg
    )


