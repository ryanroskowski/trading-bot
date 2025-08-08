from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from loguru import logger

from ..config import env


@dataclass
class AlpacaConfig:
    allow_fractional: bool


def _client() -> TradingClient:
    key = env("ALPACA_API_KEY", "")
    secret = env("ALPACA_SECRET_KEY", "")
    base_url = env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not key or not secret:
        raise RuntimeError("Alpaca API keys missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
    return TradingClient(key, secret, paper="paper" in base_url)


def submit_market_order(symbol: str, qty: float, side: str = "buy", tif: TimeInForce = TimeInForce.DAY, max_retries: int = 3) -> None:
    client = _client()
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    for attempt in range(max_retries):
        try:
            order = MarketOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=tif)
            client.submit_order(order)
            logger.info(f"Submitted market order: {side} {qty} {symbol}")
            return
        except Exception as e:  # pylint: disable=broad-except
            wait = min(2 ** attempt, 10)
            logger.warning(f"Order attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    logger.error(f"Failed to submit order for {symbol} after {max_retries} retries")


