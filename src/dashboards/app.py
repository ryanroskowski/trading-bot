from __future__ import annotations

import datetime
import requests
from typing import Dict, List, Any

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from ..config import load_config, save_config, project_root
from ..storage.db import get_conn


st.set_page_config(page_title="Trading Bot Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0px;
    }
    .status-ok { color: #00ff00; }
    .status-warn { color: #ffaa00; }
    .status-error { color: #ff0000; }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", [
    "Overview",
    "Positions & Orders", 
    "Performance Metrics",
    "Strategy Controls",
    "System Status"
])

# Helper functions
@st.cache_data(ttl=30)
def get_api_data(endpoint: str) -> Dict[str, Any]:
    """Get data from the API with caching."""
    try:
        response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=30)
def load_equity_from_db():
    """Load equity data directly from database."""
    try:
        with get_conn() as conn:
            df = pd.read_sql_query("SELECT ts, value FROM equity ORDER BY ts", conn, parse_dates=["ts"])
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def format_currency(value: float) -> str:
    """Format currency with commas and dollar sign."""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format percentage with sign."""
    return f"{value:+.2f}%"

# Main title
st.title("🤖 Trading Bot Dashboard")

# Check API health
health_data = get_api_data("/health")
if "error" in health_data:
    st.error(f"⚠️ API Connection Failed: {health_data['error']}")
    st.info("Falling back to direct database access where possible.")
    api_available = False
else:
    api_available = True

# Overview Page
if page == "Overview":
    st.header("📊 System Overview")
    
    # Get status data
    if api_available:
        status_data = get_api_data("/status")
        equity_data = get_api_data("/equity")
    else:
        status_data = {"error": "API unavailable"}
        equity_data = {"error": "API unavailable"}
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if "error" not in status_data:
        with col1:
            mode = status_data.get("mode", "unknown")
            mode_color = "🟢" if mode == "paper" else "🔴" if mode == "live" else "🟡"
            st.metric("Mode", f"{mode_color} {mode.title()}")
        
        with col2:
            profile = status_data.get("profile", "unknown")
            st.metric("Profile", profile.title())
        
        with col3:
            kill_switch = status_data.get("kill_switch_active", False)
            kill_status = "🔴 ACTIVE" if kill_switch else "🟢 OK"
            st.metric("Kill Switch", kill_status)
        
        with col4:
            equity_info = status_data.get("equity")
            if equity_info:
                current_equity = equity_info.get("value", 0)
                st.metric("Current Equity", format_currency(current_equity))
            else:
                st.metric("Current Equity", "No data")
    
    # Kill switch warning
    if api_available and status_data.get("kill_switch_active", False):
        st.error("🚨 **KILL SWITCH ACTIVE** - All trading operations are halted!")
    
    # Equity chart + Drawdown
    st.subheader("💰 Equity Curve")
    
    equity_df = None
    if "error" not in equity_data and equity_data.get("data"):
        equity_df = pd.DataFrame(equity_data["data"])
        equity_df['ts'] = pd.to_datetime(equity_df['ts'])
    else:
        equity_df = load_equity_from_db()
    
    if not equity_df.empty:
        # Create plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df['ts'],
            y=equity_df['value'],
            mode='lines',
            name='Equity',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Equity Over Time",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown
        equity_series = equity_df.set_index('ts')['value']
        peak = equity_series.cummax()
        drawdown = (equity_series / peak - 1.0) * 100
        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            line=dict(color='#d62728', width=2)
        ))
        dd_fig.update_layout(title="Drawdown (%)", xaxis_title="Date", yaxis_title="%",
                             height=250, hovermode='x unified')
        st.plotly_chart(dd_fig, use_container_width=True)
        
        # Stats below chart
        if len(equity_df) > 1:
            start_value = equity_df['value'].iloc[0]
            current_value = equity_df['value'].iloc[-1]
            total_return = (current_value / start_value - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", format_percentage(total_return))
            with col2:
                st.metric("Start Value", format_currency(start_value))
            with col3:
                st.metric("Days Tracked", len(equity_df))
    else:
        st.info("📈 No equity data available yet. Start trading to see your performance!")

elif page == "Strategy Controls":
    st.header("⚙️ Strategy Controls")
    
    cfg = load_config()
    
    # Warning for live mode
    if cfg.get("mode") == "live":
        st.warning("⚠️ **LIVE MODE ACTIVE** - Changes will affect real trading!")
    
    # Profile selection
    st.subheader("Risk Profile")
    current_profile = cfg.get("profile", "default")
    profile = st.selectbox(
        "Select Profile", 
        ["conservative", "default", "aggressive", "turbo"], 
        index=["conservative", "default", "aggressive", "turbo"].index(current_profile),
        help="Conservative: Lower vol/risk, Default: Balanced, Aggressive: Higher turnover, Turbo: Maximum frequency"
    )
    
    # Mode selection
    st.subheader("Trading Mode")
    current_mode = cfg.get("mode", "paper")
    mode = st.selectbox(
        "Select Mode",
        ["backtest", "paper", "live"],
        index=["backtest", "paper", "live"].index(current_mode),
        help="Backtest: Historical simulation, Paper: Live paper trading, Live: Real money trading"
    )
    
    # Strategy toggles
    st.subheader("Strategy Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        vm_enabled = st.checkbox(
            "VM Dual Momentum",
            value=cfg["strategies"]["vm_dual_momentum"]["enabled"],
            help="Volatility-managed dual momentum strategy"
        )
        
        ts_enabled = st.checkbox(
            "TSMOM Macro Lite",
            value=cfg["strategies"]["tsmom_macro_lite"]["enabled"],
            help="Time-series momentum macro strategy"
        )
    
    with col2:
        qv_enabled = st.checkbox(
            "QV-Trend",
            value=cfg["strategies"]["qv_trend"]["enabled"],
            help="Quality-value trend strategy for stocks"
        )
        
        on_enabled = st.checkbox(
            "Overnight Drift (Research Only)",
            value=cfg["strategies"]["overnight_drift_demo"]["enabled"],
            help="Research strategy - disabled in live mode"
        )
    
    # Apply button
    if st.button("🚀 Apply Configuration", type="primary"):
        # Update configuration
        cfg["profile"] = profile
        cfg["mode"] = mode
        cfg["strategies"]["vm_dual_momentum"]["enabled"] = vm_enabled
        cfg["strategies"]["tsmom_macro_lite"]["enabled"] = ts_enabled
        cfg["strategies"]["qv_trend"]["enabled"] = qv_enabled
        cfg["strategies"]["overnight_drift_demo"]["enabled"] = on_enabled
        
        # Save configuration
        try:
            save_config(cfg)
            st.success("✅ Configuration updated successfully! Changes will be applied on the next trading cycle.")
            
            # Show what changed
            changes = []
            if profile != current_profile:
                changes.append(f"Profile: {current_profile} → {profile}")
            if mode != current_mode:
                changes.append(f"Mode: {current_mode} → {mode}")
            
            if changes:
                st.info("Changes: " + ", ".join(changes))
                
        except Exception as e:
            st.error(f"❌ Failed to save configuration: {e}")

elif page == "Performance Metrics":
    st.header("📈 Performance & Signals")

    # Load reports if present (from backtest)
    reports_dir = project_root() / "reports"
    meta_weights_path = reports_dir / "weights.csv"
    metrics_path = reports_dir / "metrics.csv"

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Strategy Blend Weights (Backtest)")
        try:
            if meta_weights_path.exists():
                wdf = pd.read_csv(meta_weights_path, index_col=0, parse_dates=True)
                st.area_chart(wdf)
            else:
                st.info("No backtest weights.csv found yet.")
        except Exception as e:
            st.error(f"Failed to load weights: {e}")
    with cols[1]:
        st.subheader("Summary Metrics")
        try:
            if metrics_path.exists():
                mdf = pd.read_csv(metrics_path)
                st.dataframe(mdf)
            else:
                st.info("No metrics.csv found yet.")
        except Exception as e:
            st.error(f"Failed to load metrics: {e}")

    st.subheader("Live Targets & Signals (Latest)")
    targets_data = get_api_data("/targets")
    if "target_weights" in targets_data and targets_data["target_weights"]:
        lw_col, mw_col = st.columns(2)
        with lw_col:
            st.caption("Composite Target Weights")
            tdf = pd.DataFrame(list(targets_data["target_weights"].items()), columns=["Symbol", "Weight"]).set_index("Symbol")
            st.bar_chart(tdf)
        with mw_col:
            st.caption("Meta Weights by Strategy")
            mw = targets_data.get("meta_weights", {})
            if mw:
                mw_df = pd.DataFrame(list(mw.items()), columns=["Strategy", "Weight"]).set_index("Strategy")
                st.bar_chart(mw_df)
            else:
                st.info("No meta weights yet.")

        # Per-strategy targets snapshot table
        st.caption("Per-Strategy Per-Symbol Targets")
        pst = targets_data.get("per_strategy_targets", {})
        if pst:
            # Flatten into a display-friendly dataframe
            rows = []
            for strat, sym_map in pst.items():
                for sym, w in sym_map.items():
                    rows.append({"Strategy": strat, "Symbol": sym, "Weight": w})
            pst_df = pd.DataFrame(rows)
            st.dataframe(pst_df.sort_values(["Strategy", "Symbol"]))
        else:
            st.info("No per-strategy targets yet.")
    else:
        st.info("No live targets available yet.")

elif page == "System Status":
    st.header("🔧 System Status")
    
    # API health check
    health_data = get_api_data("/health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Status")
        if "error" in health_data:
            st.error(f"❌ API Unavailable: {health_data['error']}")
        else:
            st.success("✅ API Connected")
            if health_data.get("kill_switch_active"):
                st.error("🚨 Kill Switch Active")
            else:
                st.success("✅ Kill Switch OK")
    
    with col2:
        st.subheader("Database Status")
        try:
            with get_conn() as conn:
                conn.execute("SELECT 1")
            st.success("✅ Database Connected")
        except Exception as e:
            st.error(f"❌ Database Error: {e}")

    # Live runtime status banner (PDT and circuit breakers)
    try:
        with get_conn() as conn:
            rtd = pd.read_sql_query(
                "SELECT ts, pdt_trades_today, circuit_breaker_active FROM runtime_status ORDER BY ts DESC LIMIT 1",
                conn
            )
        if not rtd.empty:
            last = rtd.iloc[0]
            if int(last["circuit_breaker_active"]) == 1:
                st.error("🚨 CIRCUIT BREAKER ACTIVE — Trading Paused")
            st.info(f"PDT trades today: {int(last['pdt_trades_today'])}")
    except Exception as e:
        st.warning(f"Runtime status unavailable: {e}")
    
    # Kill switch controls
    st.subheader("Emergency Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚨 ACTIVATE KILL SWITCH", type="secondary"):
            try:
                kill_switch_path = project_root() / "KILL_SWITCH"
                kill_switch_path.touch()
                st.error("🚨 KILL SWITCH ACTIVATED - All trading stopped!")
            except Exception as e:
                st.error(f"Failed to activate kill switch: {e}")
    
    with col2:
        if st.button("✅ DEACTIVATE KILL SWITCH", type="primary"):
            try:
                kill_switch_path = project_root() / "KILL_SWITCH"
                if kill_switch_path.exists():
                    kill_switch_path.unlink()
                    st.success("✅ Kill switch deactivated - Trading can resume")
                else:
                    st.info("Kill switch was not active")
            except Exception as e:
                st.error(f"Failed to deactivate kill switch: {e}")

else:
    st.header("🔧 Coming Soon")
    st.info(f"The {page} page is under development.")

# Auto-refresh
time_to_refresh = st.sidebar.slider("Auto-refresh interval (seconds)", 10, 60, 30)
st.sidebar.markdown(f"Page will auto-refresh every {time_to_refresh} seconds")

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("*Trading Bot Dashboard - Built with Streamlit*")