from __future__ import annotations

import datetime
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pandas_market_calendars as mcal
from loguru import logger

from ..config import load_config, project_root
from ..data.marketdata import fetch_yf_ohlcv
from ..execution.alpaca import AlpacaConnector, AlpacaConfig
from ..execution.order_router import BrokerContext, reconcile_and_route
from ..strategy.vm_dual_momentum import compute_weights as vm_weights, VMDMConfig
from ..strategy.tsmom_macro_lite import compute_weights as ts_weights, TSMOMConfig
from ..strategy.quality_value_trend import compute_weights as qv_weights, QVConfig
from ..ensemble.meta_allocator import MetaAllocator, MetaConfig
from ..utils.risk import (
    realized_vol_annualized,
    vol_target_scale,
    cap_total_allocation,
    enforce_max_positions,
    apply_drawdown_derisking,
    pdt_guard,
)
from ..storage.db import init_db, insert_equity


@dataclass
class LiveContext:
    equity_usd: float
    last_config_mtime: Optional[float] = None
    pdt_trades_today: int = 0
    last_pdt_reset_date: Optional[str] = None
    circuit_breaker_active: bool = False
    daily_pnl_start_equity: Optional[float] = None


def check_market_open(timezone: str = "America/New_York") -> bool:
    """Check if NYSE is open using pandas_market_calendars."""
    try:
        nyse = mcal.get_calendar('NYSE')
        now = pd.Timestamp.now(tz=timezone)
        
        # Get today's schedule
        schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
        if schedule.empty:
            return False  # Market closed (holiday/weekend)
            
        market_open = schedule.iloc[0]['market_open'].tz_convert(timezone)
        market_close = schedule.iloc[0]['market_close'].tz_convert(timezone)
        
        return market_open <= now <= market_close
        
    except Exception as e:
        logger.warning(f"Market hours check failed: {e}. Assuming closed.")
        return False


def check_kill_switch() -> bool:
    """Check if KILL_SWITCH file exists."""
    kill_switch_path = project_root() / "KILL_SWITCH"
    if kill_switch_path.exists():
        logger.critical("KILL_SWITCH detected! Stopping all trading operations.")
        return True
    return False


def update_pdt_tracking(ctx: LiveContext) -> None:
    """Update PDT tracking for new trading day."""
    today = datetime.date.today().isoformat()
    if ctx.last_pdt_reset_date != today:
        ctx.pdt_trades_today = 0
        ctx.last_pdt_reset_date = today
        ctx.daily_pnl_start_equity = None
        logger.info(f"Reset PDT tracking for new trading day: {today}")


def check_circuit_breakers(ctx: LiveContext, current_equity: float, cfg: dict) -> bool:
    """Check if circuit breakers should be triggered."""
    # Initialize daily PnL tracking
    if ctx.daily_pnl_start_equity is None:
        ctx.daily_pnl_start_equity = current_equity
        return False
        
    # Daily PnL circuit breaker
    daily_pnl_pct = (current_equity - ctx.daily_pnl_start_equity) / ctx.daily_pnl_start_equity
    daily_loss_threshold = -abs(float(cfg["risk"].get("daily_loss_circuit_breaker_pct", 5.0))) / 100.0
    
    if daily_pnl_pct < daily_loss_threshold:
        if not ctx.circuit_breaker_active:
            logger.critical(f"CIRCUIT BREAKER: Daily loss {daily_pnl_pct:.2%} exceeds threshold {daily_loss_threshold:.2%}")
            ctx.circuit_breaker_active = True
        return True
        
    # Reset circuit breaker if we're back above threshold
    if ctx.circuit_breaker_active and daily_pnl_pct > daily_loss_threshold * 0.5:  # 50% recovery
        logger.info("Circuit breaker reset - losses recovered")
        ctx.circuit_breaker_active = False
    
    return ctx.circuit_breaker_active


def get_live_prices(symbols: list, alpaca: AlpacaConnector) -> pd.Series:
    """Get current live prices from Alpaca."""
    prices = {}
    for symbol in symbols:
        try:
            price = alpaca.get_last_price(symbol)
            if price is not None:
                prices[symbol] = price
            else:
                logger.warning(f"No live price available for {symbol}")
        except Exception as e:
            logger.error(f"Failed to get live price for {symbol}: {e}")
    
    return pd.Series(prices)


def run_live_loop() -> None:
    """Main live trading loop with full functionality."""
    logger.info("Starting live trading engine...")
    
    # Initialize
    init_db()
    cfg = load_config()
    alpaca = AlpacaConnector(AlpacaConfig(
        allow_fractional=cfg["risk"].get("allow_fractional", True),
        max_retries=3
    ))
    
    # Get initial account info
    try:
        account_info = alpaca.get_account()
        initial_equity = account_info['equity']
        is_pdt = account_info.get('pattern_day_trader', False)
        logger.info(f"Account equity: ${initial_equity:,.2f}, PDT status: {is_pdt}")
    except Exception as e:
        logger.error(f"Failed to get account info: {e}")
        initial_equity = 10000.0
        is_pdt = False
    
    ctx = LiveContext(
        equity_usd=initial_equity,
        last_config_mtime=(project_root() / "config.yaml").stat().st_mtime,
        daily_pnl_start_equity=initial_equity
    )
    
    # Initialize meta-allocator
    mcfg = MetaConfig(**cfg["ensemble"]["meta_allocator"]) if cfg["ensemble"]["meta_allocator"]["enabled"] else None
    meta = MetaAllocator(mcfg) if mcfg else None
    
    logger.info("Live trading engine initialized. Starting main loop...")
    
    while True:
        try:
            # Check kill switch
            if check_kill_switch():
                logger.critical("Kill switch activated. Exiting live loop.")
                break
            
            # Update PDT tracking
            update_pdt_tracking(ctx)
            
            # Hot reload config if changed
            try:
                cfg_path = project_root() / "config.yaml"
                mtime = cfg_path.stat().st_mtime
                if mtime > ctx.last_config_mtime:
                    old_cfg = cfg
                    cfg = load_config()
                    ctx.last_config_mtime = mtime
                    logger.info("Config hot-reloaded successfully")
                    
                    # Reinitialize meta-allocator if config changed
                    if cfg["ensemble"]["meta_allocator"] != old_cfg["ensemble"]["meta_allocator"]:
                        mcfg = MetaConfig(**cfg["ensemble"]["meta_allocator"]) if cfg["ensemble"]["meta_allocator"]["enabled"] else None
                        meta = MetaAllocator(mcfg) if mcfg else None
                        logger.info("Meta-allocator reinitialized")
                        
            except Exception as e:
                logger.warning(f"Config hot-reload failed: {e}")
            
            # Check market hours
            if cfg["schedule"]["market_hours_only"] and not check_market_open(cfg.get("timezone", "America/New_York")):
                logger.debug("Market closed, sleeping...")
                time.sleep(int(cfg["schedule"]["check_interval_seconds"]))
                continue
            
            # Get current account info
            try:
                account_info = alpaca.get_account()
                current_equity = account_info['equity']
                ctx.equity_usd = current_equity
                
                # Check circuit breakers
                if check_circuit_breakers(ctx, current_equity, cfg):
                    logger.warning("Circuit breaker active, skipping trading")
                    time.sleep(int(cfg["schedule"]["check_interval_seconds"]))
                    continue
                    
            except Exception as e:
                logger.error(f"Failed to get account info: {e}")
                time.sleep(int(cfg["schedule"]["check_interval_seconds"]))
                continue
            
            # Fetch EOD data for strategies (yfinance)
            etfs = cfg["universe"]["etfs"]
            try:
                px = fetch_yf_ohlcv(etfs, start="2020-01-01")
                close = px.close.dropna(how="all")
                open_ = px.open.dropna(how="all")
                rets = close.pct_change().fillna(0.0)
            except Exception as e:
                logger.error(f"Failed to fetch EOD data: {e}")
                time.sleep(int(cfg["schedule"]["check_interval_seconds"]))
                continue
            
            # Get live prices from Alpaca
            try:
                all_symbols = list(etfs)
                if cfg["strategies"]["qv_trend"]["enabled"]:
                    try:
                        large = pd.read_csv(project_root() / cfg["universe"]["large_cap_csv"])['Ticker'].tolist()
                        all_symbols.extend(large)
                    except Exception:
                        pass
                
                live_prices = get_live_prices(all_symbols, alpaca)
                if live_prices.empty:
                    logger.warning("No live prices available, using last close prices")
                    live_prices = close.iloc[-1]
                else:
                    # Blend live prices with last close for missing symbols
                    live_prices = live_prices.combine_first(close.iloc[-1])
                    
            except Exception as e:
                logger.error(f"Failed to get live prices: {e}")
                live_prices = close.iloc[-1]
            
            # Compute strategy weights
            w_all = []
            
            if cfg["strategies"]["vm_dual_momentum"]["enabled"]:
                try:
                    vm_cfg = VMDMConfig(
                        lookbacks_months=cfg["strategies"]["vm_dual_momentum"]["lookbacks_months"],
                        rebalance=cfg["strategies"]["vm_dual_momentum"]["rebalance"],
                        abs_filter=cfg["strategies"]["vm_dual_momentum"]["abs_filter"],
                        target_vol_annual=cfg["risk"]["target_vol_annual"],
                    )
                    w_vm = vm_weights(close, vm_cfg)
                    w_all.append(("vm_dm", w_vm))
                    logger.debug("VM-DM weights computed")
                except Exception as e:
                    logger.error(f"VM-DM strategy failed: {e}")
            
            if cfg["strategies"]["tsmom_macro_lite"]["enabled"]:
                try:
                    ts_cfg = TSMOMConfig(
                        lookback_months=cfg["strategies"]["tsmom_macro_lite"]["lookback_months"],
                        rebalance=cfg["strategies"]["tsmom_macro_lite"]["rebalance"],
                    )
                    w_ts = ts_weights(close, ts_cfg)
                    w_all.append(("tsmom", w_ts))
                    logger.debug("TSMOM weights computed")
                except Exception as e:
                    logger.error(f"TSMOM strategy failed: {e}")
            
            if cfg["strategies"]["qv_trend"]["enabled"]:
                try:
                    large = pd.read_csv(project_root() / cfg["universe"]["large_cap_csv"])['Ticker'].tolist()
                    if large:
                        stock_px = fetch_yf_ohlcv(large[:50])  # Limit to avoid timeouts
                        stock_close = stock_px.close.dropna(how="all")
                        qv_cfg = QVConfig(
                            top_n=cfg["strategies"]["qv_trend"]["top_n"],
                            rebalance=cfg["strategies"]["qv_trend"]["rebalance"]
                        )
                        all_prices_for_qv = pd.concat([stock_close, close.reindex(columns=["BIL"])], axis=1)
                        w_qv = qv_weights(all_prices_for_qv, qv_cfg, universe=large)
                        w_all.append(("qv_trend", w_qv))
                        logger.debug("QV-Trend weights computed")
                except Exception as e:
                    logger.error(f"QV-Trend strategy failed: {e}")
            
            # Skip overnight drift in live mode
            if cfg["strategies"]["overnight_drift_demo"]["enabled"]:
                logger.warning("Overnight drift strategy disabled in live mode")
            
            if not w_all:
                logger.warning("No strategy weights available, skipping this cycle")
                time.sleep(int(cfg["schedule"]["check_interval_seconds"]))
                continue
            
            # Apply next-bar execution alignment
            strat_weights = {}
            for name, w in w_all:
                aligned_w = w.shift(1).reindex(open_.index).fillna(0.0)
                # Ensure all ETF columns are present
                aligned_w = aligned_w.reindex(columns=close.columns, fill_value=0.0)
                strat_weights[name] = aligned_w
            
            # Meta-allocator ensemble
            if meta and len(strat_weights) > 1:
                try:
                    # Compute per-strategy returns
                    per_strat_returns = {}
                    for name, w in strat_weights.items():
                        strat_rets = (w * rets.reindex_like(w).fillna(0.0)).sum(axis=1)
                        per_strat_returns[name] = strat_rets.iloc[:-1]  # Exclude today
                    
                    # Get meta weights
                    current_date = open_.index[-1]
                    meta_weights = meta.step(current_date, per_strat_returns, close)
                    
                    # Blend strategies
                    comp_w = pd.DataFrame(0.0, index=open_.index, columns=close.columns)
                    for name, w in strat_weights.items():
                        weight = float(meta_weights.get(name, 0.0))
                        comp_w = comp_w.add(w.mul(weight), fill_value=0.0)
                        
                    logger.info(f"Meta-allocator weights: {meta_weights}")
                    
                except Exception as e:
                    logger.error(f"Meta-allocator failed: {e}, using equal weights")
                    comp_w = sum(w for _, w in strat_weights.items()) / len(strat_weights)
            else:
                # Equal weight blend
                comp_w = sum(w for _, w in strat_weights.items()) / max(1, len(strat_weights))
            
            # Get current target allocation
            target = comp_w.iloc[-1].copy()
            
            # Apply risk overlays
            risk = cfg["risk"]
            
            # 1. Position limits
            target = enforce_max_positions(target, int(risk["max_positions_total"]))
            
            # 2. Total allocation cap
            target = cap_total_allocation(target, float(risk["max_total_allocation_pct"]))
            
            # 3. Per-trade cap
            per_trade_cap = risk.get("per_trade_cap_usd")
            if per_trade_cap:
                for symbol in target.index:
                    if target[symbol] > 0:
                        max_weight = per_trade_cap / ctx.equity_usd
                        target[symbol] = min(target[symbol], max_weight)
            
            # 4. Volatility targeting
            try:
                base_port_ret = (comp_w.fillna(0.0) * rets.reindex_like(comp_w).fillna(0.0)).sum(axis=1)
                vol_ann = realized_vol_annualized(base_port_ret, window=20).iloc[-1]
                target_vol = float(risk["target_vol_annual"])
                scale = vol_target_scale(float(vol_ann), target_vol)
                target = target * scale
                logger.debug(f"Vol targeting: realized={vol_ann:.3f}, target={target_vol:.3f}, scale={scale:.3f}")
            except Exception as e:
                logger.warning(f"Vol targeting failed: {e}")
            
            # 5. Drawdown de-risking
            try:
                equity_curve = (1 + base_port_ret.tail(90).fillna(0.0)).cumprod()
                derisk_scale = apply_drawdown_derisking(
                    equity_curve, 
                    float(risk["dd_derisk_threshold_pct"]), 
                    float(risk["dd_derisk_scale"])
                )
                target = target * derisk_scale
                if derisk_scale < 1.0:
                    logger.info(f"Drawdown de-risking applied: scale={derisk_scale:.3f}")
            except Exception as e:
                logger.warning(f"Drawdown de-risking failed: {e}")
            
            # 6. Final allocation cap
            target = cap_total_allocation(target, float(risk["max_total_allocation_pct"]))
            
            # 7. PDT guard
            if not is_pdt and ctx.equity_usd < 25000:
                max_round_trips = 2  # Conservative limit
                if ctx.pdt_trades_today >= max_round_trips:
                    logger.warning(f"PDT guard: Already made {ctx.pdt_trades_today} round trips today, limiting new orders")
                    # Don't place new orders that would increase positions significantly
                    target = target * 0.1  # Minimal adjustments only
            
            # Execute orders
            try:
                broker_ctx = BrokerContext(
                    equity_usd=ctx.equity_usd,
                    allow_fractional=risk.get("allow_fractional", True),
                    dust_threshold_usd=risk.get("dust_threshold_usd", 3.0),
                    per_trade_cap_usd=per_trade_cap
                )
                
                order_results, submitted_deltas = reconcile_and_route(
                    target_weights=target,
                    last_prices=live_prices,
                    ctx=broker_ctx,
                    alpaca=alpaca,
                    timestamp=datetime.datetime.now().isoformat()
                )
                
                # Update PDT tracking
                for result in order_results:
                    if result.submitted:
                        ctx.pdt_trades_today += 1
                
                logger.info(f"Orders executed: {len([r for r in order_results if r.submitted])} successful, "
                           f"{len([r for r in order_results if not r.submitted])} failed")
                
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
            
            # Persist equity snapshot
            try:
                insert_equity(pd.Timestamp.utcnow().isoformat(), float(ctx.equity_usd))
            except Exception as e:
                logger.warning(f"Equity persistence failed: {e}")
            
            logger.info(f"Live loop cycle complete. Equity: ${ctx.equity_usd:,.2f}, "
                       f"PDT trades today: {ctx.pdt_trades_today}")
            
        except Exception as e:
            logger.error(f"Unexpected error in live loop: {e}")
        
        # Sleep until next cycle
        time.sleep(int(cfg["schedule"]["check_interval_seconds"]))


if __name__ == "__main__":
    run_live_loop()


