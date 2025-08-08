from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..storage.db import (
    init_db,
    get_open_orders,
    get_recent_fills,
    mark_order_closed,
)
from ..storage.orders import save_order, get_order_by_client_id
from .alpaca import AlpacaConnector, get_positions


@dataclass
class BrokerContext:
    equity_usd: float
    allow_fractional: bool
    dust_threshold_usd: float = 3.0
    per_trade_cap_usd: Optional[float] = None


@dataclass
class Position:
    symbol: str
    qty: float
    market_value: float
    unrealized_pl: float


@dataclass
class OrderResult:
    symbol: str
    client_order_id: str
    qty: float
    side: str
    submitted: bool
    error: Optional[str] = None


def generate_client_order_id(symbol: str, qty: float, side: str, timestamp: str) -> str:
    """Generate deterministic client order ID for idempotency."""
    content = f"{symbol}_{qty}_{side}_{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def get_current_positions() -> Dict[str, Position]:
    """Get current positions from broker."""
    try:
        positions_data = get_positions()
        positions = {}
        for pos_data in positions_data:
            symbol = pos_data.get('symbol', '')
            qty = float(pos_data.get('qty', 0))
            market_value = float(pos_data.get('market_value', 0))
            unrealized_pl = float(pos_data.get('unrealized_pl', 0))
            positions[symbol] = Position(symbol, qty, market_value, unrealized_pl)
        return positions
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        return {}


def compute_order_deltas(
    target_weights: pd.Series,
    current_positions: Dict[str, Position],
    last_prices: pd.Series,
    equity_usd: float
) -> Dict[str, float]:
    """Compute order deltas: target_qty - current_qty for each symbol."""
    target_dollars = (target_weights.clip(lower=0) * equity_usd).fillna(0.0)
    deltas = {}
    
    # Get all symbols we need to consider
    all_symbols = set(target_weights.index) | set(current_positions.keys())
    
    for symbol in all_symbols:
        # Target quantity
        target_dollar = target_dollars.get(symbol, 0.0)
        price = float(last_prices.get(symbol, np.nan))
        
        if np.isnan(price) or price <= 0:
            logger.warning(f"Skip {symbol}: bad price {price}")
            continue
            
        target_qty = target_dollar / price
        
        # Current quantity
        current_qty = current_positions.get(symbol, Position(symbol, 0, 0, 0)).qty
        
        # Delta
        delta = target_qty - current_qty
        
        if abs(delta) > 1e-6:  # Only include meaningful deltas
            deltas[symbol] = delta
    
    return deltas


def apply_trade_filters(
    deltas: Dict[str, float],
    last_prices: pd.Series,
    ctx: BrokerContext
) -> Dict[str, float]:
    """Apply dust threshold, fractional rounding, and per-trade caps."""
    filtered_deltas = {}
    
    for symbol, delta in deltas.items():
        price = float(last_prices.get(symbol, np.nan))
        if np.isnan(price) or price <= 0:
            continue
            
        # Apply dust threshold
        notional = abs(delta * price)
        if notional < ctx.dust_threshold_usd:
            logger.debug(f"Skip {symbol}: notional ${notional:.2f} below dust threshold")
            continue
            
        # Apply per-trade cap
        if ctx.per_trade_cap_usd and notional > ctx.per_trade_cap_usd:
            cap_qty = ctx.per_trade_cap_usd / price
            if delta > 0:
                delta = min(delta, cap_qty)
            else:
                delta = max(delta, -cap_qty)
            logger.info(f"Capped {symbol} trade to ${ctx.per_trade_cap_usd}")
            
        # Handle fractional shares
        if not ctx.allow_fractional:
            if abs(delta) < 1.0:
                logger.debug(f"Skip {symbol}: fractional qty {delta:.3f} not allowed")
                continue
            delta = float(int(delta))  # Truncate to integer
            
        filtered_deltas[symbol] = delta
    
    return filtered_deltas


def submit_orders_with_idempotency(
    deltas: Dict[str, float],
    alpaca: AlpacaConnector,
    timestamp: str
) -> List[OrderResult]:
    """Submit orders with idempotency checking."""
    init_db()  # Ensure database is initialized
    results = []
    
    for symbol, delta in deltas.items():
        if abs(delta) < 1e-6:
            continue
            
        side = "buy" if delta > 0 else "sell"
        qty = abs(delta)
        
        # Generate client order ID
        client_order_id = generate_client_order_id(symbol, qty, side, timestamp)
        
        # Check if order already exists (idempotency)
        existing_order = get_order_by_client_id(client_order_id)
        if existing_order:
            logger.info(f"Order {client_order_id} already exists, skipping")
            results.append(OrderResult(symbol, client_order_id, qty, side, False, "Already exists"))
            continue
            
        # Submit order
        try:
            order_result = alpaca.submit_market_order(symbol, qty, side, client_order_id)
            
            if order_result.get('status') == 'accepted':
                # Save to database
                save_order(
                    client_order_id=client_order_id,
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type="market",
                    status="submitted",
                    broker_order_id=order_result.get('id'),
                    submitted_at=timestamp
                )
                
                results.append(OrderResult(symbol, client_order_id, qty, side, True))
                logger.info(f"Submitted {side} {qty} {symbol} (ID: {client_order_id})")
            else:
                error_msg = order_result.get('message', 'Unknown error')
                results.append(OrderResult(symbol, client_order_id, qty, side, False, error_msg))
                logger.error(f"Failed to submit {side} {qty} {symbol}: {error_msg}")
                
        except Exception as e:
            error_msg = str(e)
            results.append(OrderResult(symbol, client_order_id, qty, side, False, error_msg))
            logger.error(f"Exception submitting {side} {qty} {symbol}: {error_msg}")
    
    return results


def reconcile_and_route(
    target_weights: pd.Series,
    last_prices: pd.Series,
    ctx: BrokerContext,
    alpaca: Optional[AlpacaConnector] = None,
    timestamp: Optional[str] = None
) -> Tuple[List[OrderResult], Dict[str, float]]:
    """
    Complete order reconciliation and routing.
    
    Args:
        target_weights: Target allocation weights (sum should be <= 1.0)
        last_prices: Last known prices for each symbol
        ctx: Broker context with equity and settings
        alpaca: Alpaca connector instance
        timestamp: Current timestamp for idempotency
        
    Returns:
        Tuple of (order_results, submitted_deltas)
    """
    if timestamp is None:
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
    if alpaca is None:
        alpaca = AlpacaConnector()
    
    logger.info(f"Starting order reconciliation at {timestamp}")
    
    # Step 1: Get current positions
    current_positions = get_current_positions()
    logger.info(f"Current positions: {len(current_positions)} symbols")
    
    # Step 2: Compute order deltas
    deltas = compute_order_deltas(target_weights, current_positions, last_prices, ctx.equity_usd)
    logger.info(f"Computed deltas for {len(deltas)} symbols")

    # Step 2.5: Partial-fill reconciliation and open orders consideration
    open_orders = get_open_orders()
    recent_fills = get_recent_fills(since_minutes=60)
    open_by_symbol = {}
    for o in open_orders:
        sym = o["symbol"]
        s = open_by_symbol.setdefault(sym, {"buy": [], "sell": []})
        side = (o.get("side") or "").lower()
        filled_qty = float(o.get("filled_qty") or 0.0)
        qty = float(o.get("qty") or 0.0)
        rem = max(0.0, qty - filled_qty)
        if side == "buy":
            s["buy"].append(rem)
        elif side == "sell":
            s["sell"].append(rem)

    effective_deltas: Dict[str, float] = {}
    for sym, delta in deltas.items():
        working_buy = sum(open_by_symbol.get(sym, {}).get("buy", []))
        working_sell = sum(open_by_symbol.get(sym, {}).get("sell", []))
        eff = delta - working_buy + working_sell
        logger.info(
            f"partial_reconcile: symbol={sym}, delta={delta:.6f}, working_buy={working_buy:.6f}, "
            f"working_sell={working_sell:.6f}, effective={eff:.6f}"
        )
        effective_deltas[sym] = eff

    # Direction flip protection: if existing open orders are opposite sign of effective delta, cancel first
    for sym, eff in list(effective_deltas.items()):
        s = open_by_symbol.get(sym)
        if not s:
            continue
        has_buy = sum(s.get("buy", [])) > 1e-9
        has_sell = sum(s.get("sell", [])) > 1e-9
        if eff < 0 and has_buy:
            logger.warning(f"cancel_open_orders due to direction flip: symbol={sym}")
            try:
                alpaca.cancel_open_orders(sym)  # best effort
            except Exception:
                pass
            # Do not submit opposite order in the same loop
            effective_deltas[sym] = 0.0
        elif eff > 0 and has_sell:
            logger.warning(f"cancel_open_orders due to direction flip: symbol={sym}")
            try:
                alpaca.cancel_open_orders(sym)
            except Exception:
                pass
            effective_deltas[sym] = 0.0
    
    # Step 3: Apply trade filters
    filtered_deltas = apply_trade_filters(effective_deltas, last_prices, ctx)
    logger.info(f"After filtering: {len(filtered_deltas)} orders to submit")
    
    # Step 4: Submit orders with idempotency
    if not filtered_deltas:
        logger.info("No orders to submit")
        return [], {}
        
    results = submit_orders_with_idempotency(filtered_deltas, alpaca, timestamp)
    
    # Step 5: Extract successfully submitted deltas
    submitted_deltas = {}
    for result in results:
        if result.submitted:
            delta = filtered_deltas.get(result.symbol, 0)
            if result.side == "sell":
                delta = -abs(delta)
            submitted_deltas[result.symbol] = delta
    
    # Mark orders closed if fully filled this loop
    for o in get_open_orders():
        try:
            mark_order_closed(o["client_order_id"])
        except Exception:
            pass

    # Optional: cancel stale orders older than 15 minutes
    try:
        from datetime import datetime, timezone
        cutoff_min = 15
        for o in get_open_orders():
            try:
                ts = o.get("submitted_at")
                if not ts:
                    continue
                # parse without strict tz handling
                submit_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                age_min = (datetime.now(timezone.utc) - submit_dt.astimezone(timezone.utc)).total_seconds() / 60.0
                if age_min >= cutoff_min:
                    logger.warning(f"stale_open_order_cancel: client_order_id={o['client_order_id']}, age_min={age_min:.1f}")
                    try:
                        alpaca.cancel_order(o["client_order_id"])
                    except Exception:
                        pass
            except Exception:
                continue
    except Exception:
        pass

    logger.info(f"Successfully submitted {len(submitted_deltas)} orders")
    return results, submitted_deltas


