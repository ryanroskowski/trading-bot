from __future__ import annotations

import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from ..config import load_config, project_root
from ..storage.db import get_conn
from ..execution.alpaca import AlpacaConnector, get_account, get_positions


app = FastAPI(title="Trading Bot Status", version="0.1.0", description="Trading Bot Status API")

# Add CORS middleware for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint."""
    try:
        # Test database connection
        with get_conn() as conn:
            conn.execute("SELECT 1")
        
        # Check if KILL_SWITCH exists
        kill_switch_active = (project_root() / "KILL_SWITCH").exists()
        
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "database": "connected",
            "kill_switch_active": kill_switch_active
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "error": str(e)
        }


@app.get("/status")
def status():
    """Get overall system status."""
    try:
        cfg = load_config()
        mode = cfg.get("mode", "paper")
        profile = cfg.get("profile", "default")
        
        # Get latest equity
        equity = None
        with get_conn() as conn:
            cur = conn.execute("SELECT ts, value FROM equity ORDER BY ts DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                equity = {"timestamp": row[0], "value": float(row[1])}
        
        # Get kill switch status
        kill_switch_active = (project_root() / "KILL_SWITCH").exists()
        
        # Get strategy status
        strategies = {}
        for strategy_name, strategy_config in cfg.get("strategies", {}).items():
            strategies[strategy_name] = {
                "enabled": strategy_config.get("enabled", False),
                "rebalance": strategy_config.get("rebalance", "monthly")
            }
        
        return {
            "mode": mode,
            "profile": profile,
            "equity": equity,
            "kill_switch_active": kill_switch_active,
            "strategies": strategies,
            "meta_allocator_enabled": cfg.get("ensemble", {}).get("meta_allocator", {}).get("enabled", False),
            "market_hours_only": cfg.get("schedule", {}).get("market_hours_only", True),
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/equity")
def equity():
    """Get equity curve data."""
    try:
        with get_conn() as conn:
            df = pd.read_sql_query("SELECT ts, value FROM equity ORDER BY ts", conn)
        
        # Convert to list of dictionaries
        records = df.to_dict(orient="records")
        
        # Calculate basic statistics
        if not df.empty:
            values = df['value'].astype(float)
            stats = {
                "current": float(values.iloc[-1]),
                "start": float(values.iloc[0]),
                "min": float(values.min()),
                "max": float(values.max()),
                "total_return_pct": float((values.iloc[-1] / values.iloc[0] - 1) * 100) if values.iloc[0] > 0 else 0,
                "count": len(values)
            }
        else:
            stats = None
        
        return {
            "data": records,
            "statistics": stats,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/positions")
def positions():
    """Get current positions from broker."""
    try:
        # Try to get live positions from Alpaca
        try:
            positions_data = get_positions()
            if positions_data is None:
                positions_data = []
            
            # Also get any cached positions from database
            cached_positions = []
            try:
                with get_conn() as conn:
                    df = pd.read_sql_query("""
                        SELECT symbol, qty, market_value, unrealized_pl, ts
                        FROM positions 
                        WHERE ts = (SELECT MAX(ts) FROM positions)
                    """, conn)
                    cached_positions = df.to_dict(orient="records")
            except:
                pass
            
            return {
                "live_positions": positions_data,
                "cached_positions": cached_positions,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            # Fallback to cached positions only
            with get_conn() as conn:
                df = pd.read_sql_query("""
                    SELECT symbol, qty, market_value, unrealized_pl, ts
                    FROM positions 
                    WHERE ts = (SELECT MAX(ts) FROM positions)
                """, conn)
                
            return {
                "live_positions": [],
                "cached_positions": df.to_dict(orient="records"),
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orders")
def orders():
    """Get recent orders."""
    try:
        with get_conn() as conn:
            # Get orders from both old and new tables
            orders_basic = pd.read_sql_query("""
                SELECT ts as timestamp, symbol, side, qty, status 
                FROM orders 
                ORDER BY ts DESC 
                LIMIT 50
            """, conn)
            
            orders_extended = pd.read_sql_query("""
                SELECT submitted_at as timestamp, symbol, side, qty, status, 
                       client_order_id, broker_order_id, order_type
                FROM orders_extended 
                ORDER BY submitted_at DESC 
                LIMIT 50
            """, conn)
        
        return {
            "recent_orders": orders_extended.to_dict(orient="records"),
            "legacy_orders": orders_basic.to_dict(orient="records"),
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "recent_orders": [],
            "legacy_orders": [],
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }


@app.get("/fills")
def fills():
    """Get recent fills."""
    try:
        with get_conn() as conn:
            fills_extended = pd.read_sql_query("""
                SELECT client_order_id, symbol, qty, price, fill_time as timestamp
                FROM fills_extended 
                ORDER BY fill_time DESC 
                LIMIT 50
            """, conn)
            
        return {
            "recent_fills": fills_extended.to_dict(orient="records"),
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "recent_fills": [],
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }


@app.get("/targets")
def targets():
    """Get last computed target weights."""
    try:
        with get_conn() as conn:
            df = pd.read_sql_query(
                """
                SELECT ts, symbol, weight FROM targets 
                WHERE ts = (SELECT MAX(ts) FROM targets)
                ORDER BY symbol
                """,
                conn
            )
            meta_df = pd.read_sql_query(
                """
                SELECT ts, strategy, weight FROM meta_targets
                WHERE ts = (SELECT MAX(ts) FROM meta_targets)
                ORDER BY strategy
                """,
                conn
            )
            strat_df = pd.read_sql_query(
                """
                SELECT ts, strategy, symbol, weight FROM strategy_targets
                WHERE ts = (SELECT MAX(ts) FROM strategy_targets)
                ORDER BY strategy, symbol
                """,
                conn
            )
        if df.empty:
            return {
                "target_weights": {},
                "computation_time": None,
                "timestamp": datetime.datetime.now().isoformat(),
                "note": "No targets available yet",
                "meta_weights": {},
                "per_strategy_targets": {}
            }
        ts = df['ts'].iloc[0]
        weights = {row['symbol']: float(row['weight']) for _, row in df.iterrows()}
        meta_weights = {row['strategy']: float(row['weight']) for _, row in meta_df.iterrows()} if not meta_df.empty else {}
        per_strategy_targets = {}
        if not strat_df.empty:
            for strat, grp in strat_df.groupby('strategy'):
                per_strategy_targets[strat] = {row['symbol']: float(row['weight']) for _, row in grp.iterrows()}
        return {
            "target_weights": weights,
            "computation_time": ts,
            "timestamp": datetime.datetime.now().isoformat(),
            "meta_weights": meta_weights,
            "per_strategy_targets": per_strategy_targets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/account")
def account():
    """Get account information from broker."""
    try:
        account_info = get_account()
        return {
            "account": account_info,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
def config():
    """Get current configuration (read-only)."""
    try:
        cfg = load_config()
        # Remove sensitive information
        safe_cfg = cfg.copy()
        if "notifier" in safe_cfg:
            if "discord_webhook_url" in safe_cfg["notifier"]:
                safe_cfg["notifier"]["discord_webhook_url"] = "***REDACTED***"
        
        return {
            "config": safe_cfg,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Get basic performance metrics."""
    try:
        with get_conn() as conn:
            # Get equity curve
            equity_df = pd.read_sql_query("SELECT ts, value FROM equity ORDER BY ts", conn)
        
        if equity_df.empty:
            return {"error": "No equity data available"}
        
        # Calculate basic metrics
        equity_values = equity_df['value'].astype(float)
        returns = equity_values.pct_change().dropna()
        
        if len(returns) < 2:
            return {"error": "Insufficient data for metrics calculation"}
        
        # Calculate metrics
        total_return = (equity_values.iloc[-1] / equity_values.iloc[0] - 1) * 100
        annualized_return = (equity_values.iloc[-1] / equity_values.iloc[0]) ** (252 / len(equity_values)) - 1
        volatility = returns.std() * (252 ** 0.5) * 100
        sharpe = (annualized_return * 100) / volatility if volatility > 0 else 0
        
        # Max drawdown
        peak = equity_values.cummax()
        drawdown = (equity_values / peak - 1) * 100
        max_drawdown = drawdown.min()
        
        metrics = {
            "total_return_pct": float(total_return),
            "annualized_return_pct": float(annualized_return * 100),
            "volatility_pct": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": float(max_drawdown),
            "current_drawdown_pct": float(drawdown.iloc[-1]),
            "days_tracked": len(equity_values),
            "start_date": equity_df['ts'].iloc[0],
            "end_date": equity_df['ts'].iloc[-1]
        }
        
        return {
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


