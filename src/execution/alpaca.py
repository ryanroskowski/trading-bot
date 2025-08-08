from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.historical import StockHistoricalDataClient
from loguru import logger

from ..config import env


@dataclass
class AlpacaConfig:
    allow_fractional: bool = True
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 10.0
    limit_fallback_percent: float = 0.01  # 1% slippage for limit fallback


class AlpacaConnector:
    """Enhanced Alpaca connector with retry logic, limit fallback, and comprehensive API coverage."""
    
    def __init__(self, config: Optional[AlpacaConfig] = None):
        self.config = config or AlpacaConfig()
        self._trading_client: Optional[TradingClient] = None
        self._data_client: Optional[StockHistoricalDataClient] = None
        
    @property
    def trading_client(self) -> TradingClient:
        if self._trading_client is None:
            key = env("ALPACA_API_KEY", "")
            secret = env("ALPACA_SECRET_KEY", "")
            base_url = env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            if not key or not secret:
                raise RuntimeError("Alpaca API keys missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
            self._trading_client = TradingClient(key, secret, paper="paper" in base_url)
        return self._trading_client
    
    @property
    def data_client(self) -> StockHistoricalDataClient:
        if self._data_client is None:
            key = env("ALPACA_API_KEY", "")
            secret = env("ALPACA_SECRET_KEY", "")
            if not key or not secret:
                raise RuntimeError("Alpaca API keys missing")
            self._data_client = StockHistoricalDataClient(key, secret)
        return self._data_client
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                account = self.trading_client.get_account()
                return {
                    'equity': float(account.equity),
                    'buying_power': float(account.buying_power),
                    'cash': float(account.cash),
                    'portfolio_value': float(account.portfolio_value),
                    'pattern_day_trader': account.pattern_day_trader,
                    'trading_blocked': account.trading_blocked,
                    'account_blocked': account.account_blocked,
                    'currency': account.currency
                }
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to get account after {self.config.max_retries} attempts: {e}")
                    raise
                wait_time = self._calculate_retry_delay(attempt)
                logger.warning(f"Account request attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                positions = self.trading_client.get_all_positions()
                return [
                    {
                        'symbol': pos.symbol,
                        'qty': float(pos.qty),
                        'market_value': float(pos.market_value),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'unrealized_plpc': float(pos.unrealized_plpc),
                        'avg_entry_price': float(pos.avg_entry_price),
                        'side': pos.side.value if pos.side else None
                    }
                    for pos in positions
                ]
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to get positions after {self.config.max_retries} attempts: {e}")
                    raise
                wait_time = self._calculate_retry_delay(attempt)
                logger.warning(f"Positions request attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last price for a symbol with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.data_client.get_stock_latest_quote(request)
                
                if symbol in quotes:
                    quote = quotes[symbol]
                    # Use mid-price between bid and ask
                    bid_price = float(quote.bid_price) if quote.bid_price else 0
                    ask_price = float(quote.ask_price) if quote.ask_price else 0
                    
                    if bid_price > 0 and ask_price > 0:
                        return (bid_price + ask_price) / 2
                    elif ask_price > 0:
                        return ask_price
                    elif bid_price > 0:
                        return bid_price
                        
                logger.warning(f"No valid price data for {symbol}")
                return None
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to get price for {symbol} after {self.config.max_retries} attempts: {e}")
                    return None
                wait_time = self._calculate_retry_delay(attempt)
                logger.warning(f"Price request attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
    
    def submit_market_order(
        self, 
        symbol: str, 
        qty: float, 
        side: str, 
        client_order_id: Optional[str] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> Dict[str, Any]:
        """Submit market order with retry and limit fallback."""
        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        
        # First try market order
        for attempt in range(self.config.max_retries):
            try:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side_enum,
                    time_in_force=time_in_force,
                    client_order_id=client_order_id
                )
                
                order = self.trading_client.submit_order(order_request)
                logger.info(f"Submitted market order: {side} {qty} {symbol} (ID: {client_order_id})")
                
                return {
                    'id': order.id,
                    'client_order_id': order.client_order_id,
                    'status': 'accepted',
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': order.side.value,
                    'order_type': 'market',
                    'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None
                }
                
            except Exception as e:
                error_msg = str(e)
                is_final_attempt = attempt == self.config.max_retries - 1
                
                # Check if we should try limit fallback
                if is_final_attempt and self._should_use_limit_fallback(error_msg):
                    logger.warning(f"Market order failed, attempting limit fallback for {symbol}")
                    return self._submit_limit_fallback(symbol, qty, side_enum, client_order_id, time_in_force)
                
                if is_final_attempt:
                    logger.error(f"Failed to submit market order for {symbol} after {self.config.max_retries} attempts: {e}")
                    return {
                        'status': 'rejected',
                        'message': error_msg,
                        'symbol': symbol,
                        'qty': qty,
                        'side': side
                    }
                
                wait_time = self._calculate_retry_delay(attempt)
                logger.warning(f"Market order attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
    
    def _submit_limit_fallback(
        self,
        symbol: str,
        qty: float,
        side_enum: OrderSide,
        client_order_id: Optional[str],
        time_in_force: TimeInForce
    ) -> Dict[str, Any]:
        """Submit limit order as fallback when market order fails."""
        try:
            # Get current price for limit calculation
            current_price = self.get_last_price(symbol)
            if not current_price:
                raise ValueError(f"Cannot get current price for {symbol}")
            
            # Calculate limit price with slippage buffer
            slippage = current_price * self.config.limit_fallback_percent
            if side_enum == OrderSide.BUY:
                limit_price = current_price + slippage  # Buy higher
            else:
                limit_price = current_price - slippage  # Sell lower
            
            # Round to reasonable precision
            limit_price = round(limit_price, 2)
            
            limit_order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=time_in_force,
                limit_price=limit_price,
                client_order_id=client_order_id
            )
            
            order = self.trading_client.submit_order(limit_order_request)
            logger.info(f"Submitted limit fallback order: {side_enum.value} {qty} {symbol} @ ${limit_price}")
            
            return {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'status': 'accepted',
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side.value,
                'order_type': 'limit',
                'limit_price': limit_price,
                'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None
            }
            
        except Exception as e:
            logger.error(f"Limit fallback also failed for {symbol}: {e}")
            return {
                'status': 'rejected',
                'message': f"Both market and limit fallback failed: {e}",
                'symbol': symbol,
                'qty': qty,
                'side': side_enum.value
            }
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        base_delay = self.config.retry_base_delay * (2 ** attempt)
        max_delay = self.config.retry_max_delay
        delay = min(base_delay, max_delay)
        
        # Add jitter (±20%)
        jitter = delay * 0.2 * (2 * random.random() - 1)
        return max(0.1, delay + jitter)
    
    def _should_use_limit_fallback(self, error_msg: str) -> bool:
        """Determine if error warrants limit order fallback."""
        error_msg_lower = error_msg.lower()
        fallback_triggers = [
            'insufficient liquidity',
            'market closed',
            'halted',
            'rejected',
            'timeout'
        ]
        return any(trigger in error_msg_lower for trigger in fallback_triggers)


# Legacy functions for backward compatibility
def _client() -> TradingClient:
    """Legacy client function."""
    connector = AlpacaConnector()
    return connector.trading_client


def submit_market_order(symbol: str, qty: float, side: str = "buy", tif: TimeInForce = TimeInForce.DAY, max_retries: int = 3) -> None:
    """Legacy market order function."""
    connector = AlpacaConnector()
    result = connector.submit_market_order(symbol, qty, side, time_in_force=tif)
    if result.get('status') != 'accepted':
        raise RuntimeError(f"Order failed: {result.get('message', 'Unknown error')}")


def get_positions() -> List[Dict[str, Any]]:
    """Get positions using the connector."""
    connector = AlpacaConnector()
    return connector.get_positions()


def get_account() -> Dict[str, Any]:
    """Get account info using the connector."""
    connector = AlpacaConnector()
    return connector.get_account()


