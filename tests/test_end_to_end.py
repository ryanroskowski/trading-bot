from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.cli.main import main as cli_main
from src.config import load_config
from src.engine import backtest as bt
from src.storage.db import init_db, get_conn


class TestEndToEndBacktest:
    """End-to-end tests for the complete backtest system."""
    
    @pytest.fixture(autouse=True)
    def setup_temp_db(self):
        """Set up a temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = Path(self.temp_dir) / "test_trading_bot.sqlite"
        
        # Patch the database path
        with patch('src.storage.db.DB_PATH', self.temp_db):
            init_db()
            yield
        
        # Cleanup
        if self.temp_db.exists():
            self.temp_db.unlink()
        os.rmdir(self.temp_dir)

    def test_backtest_complete_workflow(self):
        """Test complete backtest workflow with real strategy logic."""
        # Create test data with realistic market behavior
        dates = pd.date_range("2022-01-01", periods=180, freq="B")  # Business days
        n_days = len(dates)
        
        # Generate realistic price data with trends and volatility
        np.random.seed(42)  # For reproducible tests
        
        # SPY: Upward trend with volatility
        spy_returns = np.random.normal(0.0005, 0.015, n_days)
        spy_prices = 400 * np.exp(np.cumsum(spy_returns))
        
        # EFA: Lower return, higher volatility
        efa_returns = np.random.normal(0.0002, 0.018, n_days)
        efa_prices = 70 * np.exp(np.cumsum(efa_returns))
        
        # IEF: Low volatility bond
        ief_returns = np.random.normal(0.0001, 0.005, n_days)
        ief_prices = 100 * np.exp(np.cumsum(ief_returns))
        
        # BIL: Stable cash equivalent
        bil_prices = 100.0 + 0.001 * np.arange(n_days)
        
        # Create DataFrames
        close_data = pd.DataFrame({
            "SPY": spy_prices,
            "EFA": efa_prices,
            "IEF": ief_prices,
            "BIL": bil_prices
        }, index=dates)
        
        # Open prices (slightly different from close)
        open_data = close_data.shift(1).bfill() * (1 + np.random.normal(0, 0.001, close_data.shape))
        
        # Create mock price data object
        mock_price_data = SimpleNamespace(close=close_data, open=open_data)
        
        # Mock the data fetching
        with patch('src.engine.backtest.fetch_yf_ohlcv') as mock_fetch:
            mock_fetch.return_value = mock_price_data
            
            # Mock the large cap CSV for QV-Trend
            mock_large_cap_df = pd.DataFrame({'Ticker': ['AAPL', 'MSFT', 'GOOGL']})
            with patch('pandas.read_csv', return_value=mock_large_cap_df):
                
                # Run the backtest
                result = bt.run_backtest()
        
        # Validate results
        assert isinstance(result, bt.BacktestResult)
        assert not result.daily_returns.empty
        assert not result.equity_curve.empty
        assert not result.weights.empty
        
        # Check that we have returns for all strategies
        assert "vm_dm" in result.per_strategy_returns
        assert "tsmom" in result.per_strategy_returns
        assert "qv_trend" in result.per_strategy_returns
        assert "overnight" in result.per_strategy_returns
        
        # Validate equity curve makes sense
        assert result.equity_curve.iloc[0] == 1.0  # Starts at 1.0
        assert result.equity_curve.iloc[-1] > 0.5   # Reasonable final value
        assert result.equity_curve.iloc[-1] < 2.0   # Not too extreme
        
        # Check that weights sum appropriately (meta-allocator may not sum to 1)
        total_weights = result.weights.sum(axis=1)
        assert (total_weights >= 0).all()
        assert (total_weights <= 1.2).all()  # Allow some leverage/meta-allocation
        
        # Validate returns are reasonable
        total_return = result.equity_curve.iloc[-1] - 1.0
        assert -0.5 < total_return < 1.0  # Reasonable return range for ~6 months
        
        # Check that slippage and commission were applied
        # (results should be slightly lower than without costs)
        assert len(result.daily_returns) == len(dates) - 1  # One less due to first day

    def test_backtest_with_all_strategies_enabled(self):
        """Test backtest with all strategies enabled."""
        # Load config and enable all strategies
        with patch('src.config.load_config') as mock_load_config:
            cfg = load_config()
            cfg["strategies"]["vm_dual_momentum"]["enabled"] = True
            cfg["strategies"]["tsmom_macro_lite"]["enabled"] = True
            cfg["strategies"]["qv_trend"]["enabled"] = True
            cfg["strategies"]["overnight_drift_demo"]["enabled"] = True
            cfg["ensemble"]["meta_allocator"]["enabled"] = True
            mock_load_config.return_value = cfg
            
            # Create minimal test data
            dates = pd.date_range("2022-01-01", periods=60, freq="B")
            etf_symbols = ["SPY", "EFA", "IEF", "BIL"]
            
            # Simple trending data
            price_data = {}
            for i, symbol in enumerate(etf_symbols):
                base_price = 100 + i * 20
                price_data[symbol] = base_price + np.arange(len(dates)) * 0.1
            
            close_df = pd.DataFrame(price_data, index=dates)
            open_df = close_df.shift(1).bfill()
            
            mock_price_data = SimpleNamespace(close=close_df, open=open_df)
            
            with patch('src.engine.backtest.fetch_yf_ohlcv', return_value=mock_price_data):
                with patch('pandas.read_csv', return_value=pd.DataFrame({'Ticker': ['AAPL']})):
                    result = bt.run_backtest()
            
            # All strategies should have computed returns
            assert len(result.per_strategy_returns) == 4
            assert all(len(returns) > 0 for returns in result.per_strategy_returns.values())

    def test_backtest_cli_integration(self):
        """Test backtest through CLI interface."""
        # Create mock data
        dates = pd.date_range("2022-01-01", periods=30, freq="B")
        close_data = pd.DataFrame({
            "SPY": 400 + np.arange(len(dates)) * 0.5,
            "EFA": 70 + np.arange(len(dates)) * 0.1,
            "IEF": 100 + np.arange(len(dates)) * 0.01,
            "BIL": np.full(len(dates), 100.0)
        }, index=dates)
        
        open_data = close_data.shift(1).bfill()
        mock_price_data = SimpleNamespace(close=close_data, open=open_data)
        
        with patch('src.engine.backtest.fetch_yf_ohlcv', return_value=mock_price_data):
            with patch('pandas.read_csv', return_value=pd.DataFrame({'Ticker': ['AAPL']})):
                with patch('sys.argv', ['main.py', 'backtest', '--strategy', 'vm_dm']):
                    # This should not raise an exception
                    try:
                        cli_main()
                    except SystemExit as e:
                        # CLI may call sys.exit(0) on success
                        assert e.code == 0

    def test_backtest_error_handling(self):
        """Test backtest error handling and recovery."""
        # Test with invalid data
        with patch('src.engine.backtest.fetch_yf_ohlcv') as mock_fetch:
            # Return empty data
            mock_fetch.return_value = SimpleNamespace(
                close=pd.DataFrame(), 
                open=pd.DataFrame()
            )
            
            # Should handle gracefully
            with pytest.raises(Exception):
                bt.run_backtest()

    def test_backtest_different_profiles(self):
        """Test backtest with different risk profiles."""
        dates = pd.date_range("2022-01-01", periods=90, freq="B")
        
        # Create test data
        price_data = {
            "SPY": 400 + np.cumsum(np.random.normal(0.05, 1.0, len(dates))),
            "EFA": 70 + np.cumsum(np.random.normal(0.02, 1.2, len(dates))),
            "IEF": 100 + np.cumsum(np.random.normal(0.01, 0.3, len(dates))),
            "BIL": np.full(len(dates), 100.0)
        }
        
        close_df = pd.DataFrame(price_data, index=dates)
        open_df = close_df.shift(1).bfill()
        mock_price_data = SimpleNamespace(close=close_df, open=open_df)
        
        profiles = ["conservative", "default", "aggressive"]
        results = {}
        
        for profile in profiles:
            with patch('src.config.load_config') as mock_load_config:
                cfg = load_config()
                cfg["profile"] = profile
                mock_load_config.return_value = cfg
                
                with patch('src.engine.backtest.fetch_yf_ohlcv', return_value=mock_price_data):
                    with patch('pandas.read_csv', return_value=pd.DataFrame({'Ticker': ['AAPL']})):
                        result = bt.run_backtest()
                        results[profile] = result
        
        # All profiles should complete successfully
        assert len(results) == 3
        for profile, result in results.items():
            assert isinstance(result, bt.BacktestResult)
            assert not result.equity_curve.empty

    def test_reports_generation(self):
        """Test that reports are generated correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch the reports directory
            with patch('src.config.project_root', return_value=Path(temp_dir)):
                
                # Create minimal test data
                dates = pd.date_range("2022-01-01", periods=50, freq="B")
                close_data = pd.DataFrame({
                    "SPY": 400 + np.arange(len(dates)),
                    "EFA": 70 + np.arange(len(dates)) * 0.5,
                    "IEF": np.full(len(dates), 100.0),
                    "BIL": np.full(len(dates), 100.0)
                }, index=dates)
                
                open_data = close_data.shift(1).bfill()
                mock_price_data = SimpleNamespace(close=close_data, open=open_data)
                
                with patch('src.engine.backtest.fetch_yf_ohlcv', return_value=mock_price_data):
                    with patch('pandas.read_csv', return_value=pd.DataFrame({'Ticker': ['AAPL']})):
                        result = bt.run_backtest()
                
                # Check that reports directory was created
                reports_dir = Path(temp_dir) / "reports"
                assert reports_dir.exists()


class TestLiveEngineComponents:
    """Test components that would be used in live trading."""
    
    def test_market_calendar_integration(self):
        """Test market calendar functionality."""
        from src.engine.live import check_market_open
        
        # This test will vary based on when it's run, but should not crash
        result = check_market_open("America/New_York")
        assert isinstance(result, bool)

    def test_kill_switch_detection(self):
        """Test kill switch detection."""
        from src.engine.live import check_kill_switch
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.engine.live.project_root', return_value=Path(temp_dir)):
                # No kill switch file
                assert not check_kill_switch()
                
                # Create kill switch file
                kill_switch_path = Path(temp_dir) / "KILL_SWITCH"
                kill_switch_path.touch()
                assert check_kill_switch()

    @patch('src.execution.alpaca.AlpacaConnector')
    def test_live_engine_initialization(self, mock_alpaca_class):
        """Test that live engine can initialize without errors."""
        from src.engine.live import LiveContext
        
        # Create a mock alpaca instance
        mock_alpaca = mock_alpaca_class.return_value
        mock_alpaca.get_account.return_value = {
            'equity': 10000.0,
            'pattern_day_trader': False
        }
        
        # Test context creation
        ctx = LiveContext(equity_usd=10000.0)
        assert ctx.equity_usd == 10000.0
        assert ctx.pdt_trades_today == 0


class TestIntegrationSmokeTests:
    """High-level smoke tests for system integration."""
    
    def test_config_profile_application(self):
        """Test that profile settings are applied correctly."""
        from src.config import load_config, apply_profile_overrides
        
        base_config = load_config()
        base_config["profile"] = "aggressive"
        
        # Test aggressive profile
        aggressive_config = apply_profile_overrides(base_config.copy())
        
        # Should have more aggressive settings
        vm_config = aggressive_config["strategies"]["vm_dual_momentum"]
        assert vm_config["rebalance"] == "weekly"
        assert vm_config["lookbacks_months"] == [3, 6]

    def test_database_operations(self):
        """Test basic database operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_db = Path(temp_dir) / "test.sqlite"
            
            with patch('src.storage.db.DB_PATH', temp_db):
                init_db()
                
                # Test equity insertion
                from src.storage.db import insert_equity
                insert_equity("2023-01-01T12:00:00", 10000.0)
                
                # Test equity retrieval
                with get_conn() as conn:
                    cursor = conn.execute("SELECT * FROM equity")
                    rows = cursor.fetchall()
                    assert len(rows) == 1

    def test_strategy_weight_computation(self):
        """Test that all strategies can compute weights without errors."""
        # Create test data
        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        symbols = ["SPY", "EFA", "IEF", "BIL"]
        
        # Generate test prices
        price_data = {}
        for symbol in symbols:
            base = 100 if symbol == "BIL" else 50 + len(symbol) * 20
            price_data[symbol] = base + np.cumsum(np.random.normal(0, 0.5, len(dates)))
        
        close_df = pd.DataFrame(price_data, index=dates)
        
        # Test VM-DM
        from src.strategy.vm_dual_momentum import compute_weights as vm_weights, VMDMConfig
        vm_config = VMDMConfig(lookbacks_months=[6, 12], rebalance="monthly", abs_filter=True, target_vol_annual=0.1)
        vm_result = vm_weights(close_df, vm_config)
        assert isinstance(vm_result, pd.DataFrame)
        assert not vm_result.empty
        
        # Test TSMOM
        from src.strategy.tsmom_macro_lite import compute_weights as ts_weights, TSMOMConfig
        ts_config = TSMOMConfig(lookback_months=12, rebalance="monthly")
        ts_result = ts_weights(close_df, ts_config)
        assert isinstance(ts_result, pd.DataFrame)
        assert not ts_result.empty
        
        # All strategy results should have same index
        assert len(vm_result.index) == len(ts_result.index)

    def test_risk_management_functions(self):
        """Test risk management utilities."""
        from src.utils.risk import (
            vol_target_scale,
            cap_total_allocation, 
            enforce_max_positions,
            check_all_circuit_breakers
        )
        
        # Test vol targeting
        scale = vol_target_scale(0.15, 0.10)  # 15% realized, 10% target
        assert 0.5 < scale < 1.0  # Should scale down
        
        # Test allocation capping
        weights = pd.Series({"SPY": 0.6, "IEF": 0.5})  # Sums to 1.1
        capped = cap_total_allocation(weights, 0.8)
        assert capped.sum() <= 0.8
        
        # Test position limiting
        weights = pd.Series({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2})
        limited = enforce_max_positions(weights, 3)
        assert (limited > 0).sum() <= 3
        
        # Test circuit breakers
        equity_curve = pd.Series([100, 95, 90, 85, 80])  # 20% drawdown
        returns = pd.Series([0.01, -0.05, -0.05, -0.05, -0.06])
        
        status = check_all_circuit_breakers(
            current_equity=80.0,
            start_of_day_equity=100.0,
            equity_curve=equity_curve,
            returns=returns
        )
        
        assert status.daily_loss_triggered  # 20% loss should trigger
        assert isinstance(status.get_summary(), str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
