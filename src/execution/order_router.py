from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from .alpaca import submit_market_order


@dataclass
class BrokerContext:
    equity_usd: float
    allow_fractional: bool


def reconcile_and_route(target_weights: pd.Series, last_prices: pd.Series, ctx: BrokerContext) -> Dict[str, float]:
    """Convert target weights to order quantities and submit market orders.
    Returns dict of symbol->qty submitted. Simplified netting; assumes full rebalance.
    """
    target_dollars = (target_weights.clip(lower=0) * ctx.equity_usd).fillna(0.0)
    qty = {}
    for symbol, dollars in target_dollars.items():
        price = float(last_prices.get(symbol, np.nan))
        if np.isnan(price) or price <= 0:
            logger.warning(f"Skip {symbol}: bad price")
            continue
        q = dollars / price
        if not ctx.allow_fractional:
            q = int(q)
        qty[symbol] = q
        if q > 0:
            submit_market_order(symbol, q, side="buy")
        else:
            # In a full implementation, we would also send sells to flatten to zero
            pass
    return qty


