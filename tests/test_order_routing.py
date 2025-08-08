from __future__ import annotations

import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.execution.order_router import (
    BrokerContext,
    Position,
    OrderResult,
    compute_order_deltas,
    apply_trade_filters,
    reconcile_and_route,
    generate_client_order_id,
)
from src.execution.alpaca import AlpacaConnector


class TestOrderRouting:
    def test_compute_order_deltas_basic(self):
        """Test basic delta computation."""
        # Target weights
        target_weights = pd.Series({
            "SPY": 0.5,
            "IEF": 0.3,
            "BIL": 0.2
        })
        
        # Current positions
        current_positions = {
            "SPY": Position("SPY", 10.0, 4500.0, 100.0),
            "IEF": Position("IEF", 5.0, 1000.0, -50.0),
            # No BIL position
        }
        
        # Last prices
        last_prices = pd.Series({
            "SPY": 450.0,
            "IEF": 100.0,
            "BIL": 100.0
        })
        
        equity_usd = 10000.0
        
        deltas = compute_order_deltas(target_weights, current_positions, last_prices, equity_usd)
        
        # SPY: target = 5000/450 = 11.11, current = 10, delta = 1.11
        # IEF: target = 3000/100 = 30, current = 5, delta = 25
        # BIL: target = 2000/100 = 20, current = 0, delta = 20
        
        assert abs(deltas["SPY"] - 1.111) < 0.01
        assert abs(deltas["IEF"] - 25.0) < 0.01
        assert abs(deltas["BIL"] - 20.0) < 0.01

    def test_apply_trade_filters_dust_threshold(self):
        """Test dust threshold filtering."""
        deltas = {
            "SPY": 0.1,    # $45 notional, above threshold
            "IEF": 0.01,   # $1 notional, below threshold
            "BIL": -0.02   # $2 notional, below threshold
        }
        
        last_prices = pd.Series({
            "SPY": 450.0,
            "IEF": 100.0,
            "BIL": 100.0
        })
        
        ctx = BrokerContext(equity_usd=10000.0, allow_fractional=True, dust_threshold_usd=3.0)
        
        filtered = apply_trade_filters(deltas, last_prices, ctx)
        
        # Only SPY should remain
        assert "SPY" in filtered
        assert "IEF" not in filtered
        assert "BIL" not in filtered

    def test_apply_trade_filters_fractional(self):
        """Test fractional share handling."""
        deltas = {
            "SPY": 1.5,
            "IEF": 0.3,    # Less than 1 share
            "BIL": -2.7
        }
        
        last_prices = pd.Series({
            "SPY": 450.0,
            "IEF": 100.0,
            "BIL": 100.0
        })
        
        # Test with fractional disabled
        ctx = BrokerContext(equity_usd=10000.0, allow_fractional=False)
        
        filtered = apply_trade_filters(deltas, last_prices, ctx)
        
        # SPY should be truncated to 1.0, IEF should be excluded, BIL should be -2.0
        assert filtered["SPY"] == 1.0
        assert "IEF" not in filtered  # Below 1.0 share
        assert filtered["BIL"] == -2.0

    def test_apply_trade_filters_per_trade_cap(self):
        """Test per-trade cap."""
        deltas = {
            "SPY": 20.0,   # $9000 notional, over cap
            "IEF": 10.0,   # $1000 notional, under cap
        }
        
        last_prices = pd.Series({
            "SPY": 450.0,
            "IEF": 100.0
        })
        
        ctx = BrokerContext(equity_usd=10000.0, allow_fractional=True, per_trade_cap_usd=5000.0)
        
        filtered = apply_trade_filters(deltas, last_prices, ctx)
        
        # SPY should be capped to 5000/450 = 11.11 shares
        assert abs(filtered["SPY"] - 11.111) < 0.01
        assert filtered["IEF"] == 10.0

    def test_generate_client_order_id_deterministic(self):
        """Test that client order ID generation is deterministic."""
        symbol = "SPY"
        qty = 10.5
        side = "buy"
        timestamp = "2023-01-01T12:00:00"
        
        id1 = generate_client_order_id(symbol, qty, side, timestamp)
        id2 = generate_client_order_id(symbol, qty, side, timestamp)
        
        assert id1 == id2
        assert len(id1) == 16  # MD5 hash truncated to 16 chars

    @patch('src.execution.order_router.get_current_positions')
    @patch('src.execution.order_router.submit_orders_with_idempotency')
    def test_reconcile_and_route_integration(self, mock_submit_orders, mock_get_positions):
        """Test full reconcile and route integration."""
        # Mock current positions
        mock_get_positions.return_value = {
            "SPY": Position("SPY", 5.0, 2250.0, 50.0)
        }
        
        # Mock order submission
        mock_submit_orders.return_value = [
            OrderResult("SPY", "order123", 5.0, "buy", True),
            OrderResult("IEF", "order124", 10.0, "buy", True)
        ]
        
        # Set up inputs
        target_weights = pd.Series({
            "SPY": 0.5,   # Target: 10 shares, current: 5, delta: +5
            "IEF": 0.3,   # Target: 10 shares, current: 0, delta: +10
            "BIL": 0.0    # No position wanted
        })
        
        last_prices = pd.Series({
            "SPY": 500.0,
            "IEF": 300.0,
            "BIL": 100.0
        })
        
        ctx = BrokerContext(equity_usd=10000.0, allow_fractional=True)
        
        # Create mock alpaca connector
        mock_alpaca = Mock(spec=AlpacaConnector)
        
        # Execute
        order_results, submitted_deltas = reconcile_and_route(
            target_weights=target_weights,
            last_prices=last_prices,
            ctx=ctx,
            alpaca=mock_alpaca,
            timestamp="2023-01-01T12:00:00"
        )
        
        # Verify results
        assert len(order_results) == 2
        assert len(submitted_deltas) == 2
        
        # Verify positions were queried
        mock_get_positions.assert_called_once()
        
        # Verify orders were submitted
        mock_submit_orders.assert_called_once()

    @patch('src.execution.order_router.get_open_orders')
    @patch('src.execution.order_router.get_current_positions')
    @patch('src.execution.order_router.submit_orders_with_idempotency')
    def test_partial_fills_reduce_effective_delta(self, mock_submit, mock_get_pos, mock_get_open):
        """If there is an open BUY(10) with 3 filled, and target delta is +10, effective delta should be +7."""
        mock_get_pos.return_value = {}
        # Open order: BUY 10, filled 3
        mock_get_open.return_value = [{
            'client_order_id': 'abc', 'symbol': 'SPY', 'side': 'buy',
            'qty': 10.0, 'filled_qty': 3.0, 'status': 'PARTIALLY_FILLED', 'submitted_at': '2023-01-01T12:00:00'
        }]
        mock_submit.return_value = [OrderResult('SPY', 'cid', 7.0, 'buy', True)]

        target_weights = pd.Series({'SPY': 1.0})
        last_prices = pd.Series({'SPY': 100.0})
        ctx = BrokerContext(equity_usd=1000.0, allow_fractional=True, dust_threshold_usd=0.0)
        mock_alpaca = Mock(spec=AlpacaConnector)

        _, submitted = reconcile_and_route(target_weights, last_prices, ctx, alpaca=mock_alpaca, timestamp='2023-01-01T12:00:01')
        # 1000$ * 1.0 / 100 = 10 target qty; filled=3 so remaining needed is 7
        assert abs(submitted.get('SPY', 0.0) - 7.0) < 1e-6

    @patch('src.execution.order_router.get_open_orders')
    @patch('src.execution.order_router.get_current_positions')
    @patch('src.execution.order_router.submit_orders_with_idempotency')
    def test_direction_flip_cancels_then_waits(self, mock_submit, mock_get_pos, mock_get_open):
        """If target flips direction while opposite open orders exist, router cancels and does not submit new in same loop."""
        mock_get_pos.return_value = {}
        # Open order BUY 10 remaining
        mock_get_open.return_value = [{
            'client_order_id': 'abc', 'symbol': 'SPY', 'side': 'buy',
            'qty': 10.0, 'filled_qty': 0.0, 'status': 'ACCEPTED', 'submitted_at': '2023-01-01T12:00:00'
        }]
        mock_submit.return_value = []  # should not be called with opposite side same loop

        target_weights = pd.Series({'SPY': -0.5})  # sell target
        last_prices = pd.Series({'SPY': 100.0})
        ctx = BrokerContext(equity_usd=1000.0, allow_fractional=True, dust_threshold_usd=0.0)
        mock_alpaca = Mock(spec=AlpacaConnector)

        results, submitted = reconcile_and_route(target_weights, last_prices, ctx, alpaca=mock_alpaca, timestamp='2023-01-01T12:01:00')
        # No orders submitted due to direction flip cancel-first policy
        assert submitted == {}
        mock_submit.assert_not_called()


class TestOrderRoutingEdgeCases:
    def test_empty_target_weights(self):
        """Test handling of empty target weights."""
        target_weights = pd.Series(dtype=float)
        current_positions = {}
        last_prices = pd.Series(dtype=float)
        
        deltas = compute_order_deltas(target_weights, current_positions, last_prices, 10000.0)
        
        assert len(deltas) == 0

    def test_missing_price_data(self):
        """Test handling of missing price data."""
        target_weights = pd.Series({"SPY": 0.5, "MISSING": 0.3})
        current_positions = {}
        last_prices = pd.Series({"SPY": 450.0})  # Missing price for MISSING
        
        deltas = compute_order_deltas(target_weights, current_positions, last_prices, 10000.0)
        
        # Should only compute delta for SPY
        assert "SPY" in deltas
        assert "MISSING" not in deltas

    def test_zero_equity(self):
        """Test handling of zero equity."""
        target_weights = pd.Series({"SPY": 0.5})
        current_positions = {}
        last_prices = pd.Series({"SPY": 450.0})
        
        deltas = compute_order_deltas(target_weights, current_positions, last_prices, 0.0)
        
        assert len(deltas) == 0

    def test_negative_prices(self):
        """Test handling of negative or zero prices."""
        target_weights = pd.Series({"SPY": 0.5, "INVALID": 0.3})
        current_positions = {}
        last_prices = pd.Series({"SPY": 450.0, "INVALID": -10.0})
        
        deltas = compute_order_deltas(target_weights, current_positions, last_prices, 10000.0)
        
        # Should only compute delta for SPY (positive price)
        assert "SPY" in deltas
        assert "INVALID" not in deltas


@pytest.fixture
def sample_positions():
    """Sample positions for testing."""
    return {
        "SPY": Position("SPY", 10.0, 4500.0, 100.0),
        "IEF": Position("IEF", 5.0, 1500.0, -25.0),
        "BIL": Position("BIL", 20.0, 2000.0, 0.0)
    }


@pytest.fixture
def sample_prices():
    """Sample prices for testing."""
    return pd.Series({
        "SPY": 450.0,
        "IEF": 300.0,
        "BIL": 100.0,
        "EFA": 80.0
    })


def test_position_handling(sample_positions, sample_prices):
    """Test position data handling."""
    target_weights = pd.Series({
        "SPY": 0.4,   # Reduce from current
        "IEF": 0.4,   # Increase from current  
        "EFA": 0.2,   # New position
        "BIL": 0.0    # Close position
    })
    
    equity_usd = 10000.0
    
    deltas = compute_order_deltas(target_weights, sample_positions, sample_prices, equity_usd)
    
    # SPY: target = 4000/450 = 8.89, current = 10, delta = -1.11 (sell)
    # IEF: target = 4000/300 = 13.33, current = 5, delta = +8.33 (buy)
    # EFA: target = 2000/80 = 25, current = 0, delta = +25 (buy)
    # BIL: target = 0, current = 20, delta = -20 (sell)
    
    assert deltas["SPY"] < 0  # Sell
    assert deltas["IEF"] > 0  # Buy
    assert deltas["EFA"] > 0  # Buy
    assert deltas["BIL"] < 0  # Sell
    
    assert abs(deltas["SPY"] + 1.111) < 0.01
    assert abs(deltas["IEF"] - 8.333) < 0.01
    assert abs(deltas["EFA"] - 25.0) < 0.01
    assert abs(deltas["BIL"] + 20.0) < 0.01
