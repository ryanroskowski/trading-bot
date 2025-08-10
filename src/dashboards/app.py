from __future__ import annotations

import datetime
import os
import base64
import requests
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from src.config import load_config, save_config, project_root
from src.storage.db import get_conn


st.set_page_config(page_title="Trading Bot Dashboard", page_icon="✨", layout="wide", initial_sidebar_state="expanded")
pio.templates.default = "simple_white"
px.defaults.template = "simple_white"
# Unified colors
px.defaults.color_discrete_sequence = [
    "#4f46e5", "#22c55e", "#0ea5e9", "#f59e0b", "#ef4444", "#9333ea", "#14b8a6"
]

# Light and airy theme inspired by the mockup
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    :root {
        --bg: #fcfcfd;
        --card: #ffffff;
        --accent: #6366f1; /* indigo accent to match header gradient */
        --accentRing: #e0e7ff;
        --muted: #6b7280;
        --success: #22c55e;
        --warn: #f59e0b;
        --error: #ef4444;
        --keyline: #eef2f7;
        --sbw: 0px; /* sidebar width */
    }
    html, body, [class*="css"], .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', Arial, sans-serif; }
    .stApp { background-color: var(--bg); }
    /* Ensure primary buttons always have readable text */
    .stButton>button[kind="primary"], .stButton>button[data-baseweb="button"] {
        color: #fff !important;
    }
    .status-ok { color: var(--success); }
    .status-warn { color: var(--warn); }
    .status-error { color: var(--error); }
    h1, h2, h3 { color: #0f172a; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background: var(--card); border-radius: 10px; padding: 8px 12px; border: 1px solid var(--keyline); }
    /* Keep rounded corners but let theme control colors */
    .stButton>button { border-radius: 10px; padding: 0.5rem 0.9rem; }
    .stDownloadButton>button { border-radius: 10px; padding: 0.4rem 0.8rem; }
    .stSelectbox, .stNumberInput, .stTextInput { background: var(--card); }
    .stMetric { background: var(--card); border-radius: 8px; padding: 8px; }
    .css-1dp5vir { background: var(--card); }
    .small { color: var(--muted); font-size: 0.9rem; }
    .badge { display: inline-block; border: 1px solid #e5e7eb; border-radius: 999px; padding: 2px 8px; background: var(--card); color: #0f172a; }
    .muted { color: var(--muted); }
    .divider { height: 1px; background: #e5e7eb; margin: 8px 0; }
    .pill { border-radius: 999px; background: #e0f2fe; color: #0369a1; padding: 2px 8px; }
    .kpi { font-weight: 600; }
    .panel { background: var(--card); border-radius: 12px; padding: 12px; border: 1px solid var(--keyline); }
    .header-space { margin-top: -10px; }
    .spacer { height: 8px; }
    .btn-row { display: flex; gap: 8px; align-items: center; }
    .success { color: var(--success); }
    .warning { color: var(--warn); }
    .danger { color: var(--error); }
    .accent { color: var(--accent); }
    .shadow-sm { box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .rounded { border-radius: 12px; }
    .border { border: 1px solid #e5e7eb; }
    .p-2 { padding: 8px; }
    .p-3 { padding: 12px; }
    .mt-1 { margin-top: 4px; }
    .mt-2 { margin-top: 8px; }
    .mt-3 { margin-top: 12px; }
    .mb-2 { margin-bottom: 8px; }
    .mb-3 { margin-bottom: 12px; }
    .w-100 { width: 100%; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .center { display: flex; align-items: center; gap: 6px; }
    .tiny { font-size: 0.8rem; opacity: 0.8; }
    .code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .note { background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }
    .glass { backdrop-filter: blur(4px); }
    .rounded-pill { border-radius: 999px; }
    .badge-accent { background: #e0f2fe; color: #075985; }
    .badge-muted { background: #f1f5f9; color: #334155; }
    .text-muted { color: #64748b; }
    .text-strong { color: #0f172a; font-weight: 600; }
    .text-accent { color: var(--accent); }
    .surface { background: var(--card); border: 1px solid var(--keyline); border-radius: 12px; }
    .chip { border: 1px solid #e2e8f0; padding: 2px 8px; border-radius: 9999px; background: #eff6ff; color: #1e40af; font-size: 0.85rem; }
    .title { color: #0f172a; }
    .subtitle { color: #334155; }
    .footnote { color: #94a3b8; font-size: 0.8rem; }
    /* Table polish */
    .stDataFrame { border: 1px solid #e5e7eb; border-radius: 8px; }
    .stDataFrame table { border-collapse: collapse; }
    .stDataFrame th, .stDataFrame td { border-bottom: 1px solid #f1f5f9; }
    .stAlert { border-radius: 8px; }
    .stProgress > div > div > div { background-color: var(--accent); }
    .st-emotion-cache-1r4qj8v { background: transparent; }
    .st-emotion-cache-1r6slb0 { background: var(--card); }
    .stMetric > label { color: #475569; }
    .stCaption { color: #64748b; }
    .stMarkdown { color: #0f172a; }
    .keyline { border-top: 1px solid var(--keyline); margin: 6px 0; }
    .rounded-8 { border-radius: 8px; }
    .rounded-10 { border-radius: 10px; }
    .rounded-12 { border-radius: 12px; }
    .bg-card { background: var(--card); }
    .bg-muted { background: #fafbff; }
    .border-muted { border-color: var(--keyline); }
    .shadow-none { box-shadow: none; }
    .shadow-xs { box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .shadow-md { box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .shadow-lg { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
    .opacity-70 { opacity: 0.7; }
    .opacity-50 { opacity: 0.5; }
    .hover-underline:hover { text-decoration: underline; }
    .hover-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); transition: box-shadow 120ms ease; }
    .divider-soft { height: 1px; background: #edf2f7; margin: 6px 0; }
    .soft { color: #64748b; }
    .soft-bg { background: #f8fafc; }
    /* Button hover handled by theme */
    .muted-link { color: #64748b; }
    .muted-link:hover { color: #0ea5e9; }
    .chip-outline { border: 1px dashed #93c5fd; color: #1d4ed8; background: #eff6ff; }
    .tinycaps { text-transform: uppercase; letter-spacing: .08em; font-size: .75rem; color: #64748b; }
    .fineprint { color: #9ca3af; font-size: .8rem; }
    .btn-ghost > button { background: transparent; border: 1px solid #e5e7eb; }
    .btn-ghost > button:hover { border-color: var(--accent); }
    .btn-accent > button { background: var(--accent); color: white; }
    .btn-accent > button:hover { filter: brightness(0.95); }

    /* Global accent for native inputs */
    input[type="checkbox"], input[type="radio"] { accent-color: var(--accent); }

    /* Minimal header restyle: color + custom brand only (no spacing changes) */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #efe9ff 0%, #eaf2ff 100%) !important;
        position: relative; /* allow positioning the title without affecting layout */
    }
    /* Brand title/image are injected via ::before/::after in a separate style block below */

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: #f6f8fc;
        border-right: 1px solid var(--keyline);
    }
    /* Slider: hide duplicate tick bar to ensure only one rail is visible */
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { display: none !important; }
    /* (Other slider/select visual styles come from Streamlit theme) */
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h1 { color:#0f172a; }

    /* Checkbox/select/button/tabs colors now come from Streamlit theme */
</style>
""",
    unsafe_allow_html=True,
)

## Header brand: icon + title using CSS pseudo-elements (no DOM injection)
logo_b64 = ""
brand_img_path = project_root() / "src" / "assets" / "logo.png"
if brand_img_path.exists():
    try:
        with open(brand_img_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception:
        logo_b64 = ""

icon_css = (
    f"""
header[data-testid=\"stHeader\"]::before {{
  content: "Trading Bot Dashboard";
  position: absolute; left: calc(var(--sbw) + 16px); top: 50%; transform: translateY(-50%);
  font-weight: 700; font-size: 20px; letter-spacing: -0.02em; color: #0f172a;
  pointer-events: none; z-index: 10000;
  padding-left: 60px;
  background-image: url('data:image/png;base64,{logo_b64}');
  background-size: 65px 65px; background-repeat: no-repeat; background-position: left center;
}}
"""
) if logo_b64 else ""

style_block = (
    "<style>\n"
    "/* Header brand: icon + title, positioned with CSS var --sbw (sidebar width) */\n"
    + icon_css + "\n"
    "/* Fallback if JS hasn't updated --sbw yet but :has() is supported */\n"
    ".stApp:has(section[data-testid=\"stSidebar\"][aria-expanded=\"true\"]) header[data-testid=\"stHeader\"]::before {\n"
    "  left: calc(260px + 16px);\n"
    "}\n"
    "</style>"
)
st.markdown(style_block, unsafe_allow_html=True)

st.markdown(
    """
<script>
(function(){
  function sync(){
    const root = document.documentElement;
    const sb = document.querySelector('section[data-testid="stSidebar"]');
    if (!sb) { root.style.setProperty('--sbw','0px'); return; }
    // Measure actual width instead of relying on aria-expanded
    let w = sb.getBoundingClientRect().width || 0;
    // Treat tiny widths as collapsed
    if (w < 40) w = 0;
    root.style.setProperty('--sbw', (Math.round(w) || 0) + 'px');
  }
  const sb = document.querySelector('section[data-testid=\"stSidebar\"]');
  if (sb) new MutationObserver(sync).observe(sb, { attributes:true, attributeFilter:['aria-expanded','style','class'] });
  window.addEventListener('resize', sync);
  setTimeout(sync, 0); setTimeout(sync, 200); setTimeout(sync, 600);
})();
</script>
""",
    unsafe_allow_html=True,
)

# Use the default Streamlit header (styled via CSS above) instead of a custom one

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    [
    "Overview",
    "Positions & Orders", 
    "Performance Metrics",
    "Strategy Controls",
        "System Status",
        "Funding Planner",
    ],
    help="Switch between major sections of the trading dashboard",
    label_visibility="collapsed"
)

# Helper functions
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


@st.cache_data(ttl=30)
def get_api_data(endpoint: str) -> Dict[str, Any]:
    """Get data from the API with caching."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def post_api(endpoint: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload or {}, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned {response.status_code}", "text": response.text}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=30)
def load_equity_from_db():
    """Load equity data directly from database."""
    try:
        # If DB file doesn't exist yet (read-only mount in dashboard), skip
        db_path = project_root() / "db" / "trading_bot.sqlite"
        if not db_path.exists():
            return pd.DataFrame()
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

# Main title removed in favor of gradient header

# Check API health
health_data = get_api_data("/health")
if "error" in health_data:
    st.error(f"⚠️ API Connection Failed: {health_data['error']}")
    st.info("Falling back to direct database access where possible.")
    api_available = False
else:
    api_available = True

reports_dir = project_root() / "reports"

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
            mode_dot = "🟢" if mode == "paper" else ("🔴" if mode == "live" else "🟡")
            st.markdown(f"<div class='panel'><div class='tinycaps soft'>Mode</div><div style='font-size:20px;font-weight:600'>{mode_dot} {mode.title()}</div></div>", unsafe_allow_html=True)
        
        with col2:
            profile = status_data.get("profile", "unknown")
            st.markdown(f"<div class='panel'><div class='tinycaps soft'>Profile</div><div><span class='chip'>{profile.title()}</span> <span class='badge-muted'>Ok</span></div></div>", unsafe_allow_html=True)
        
        with col3:
            kill_switch = status_data.get("kill_switch_active", False)
            kill_status = "🔴 ACTIVE" if kill_switch else "🟢 OK"
            st.markdown(f"<div class='panel'><div class='tinycaps soft'>Status</div><div style='font-weight:600'>{kill_status}</div></div>", unsafe_allow_html=True)
        
        with col4:
            equity_info = status_data.get("equity")
            if equity_info:
                current_equity = equity_info.get("value", 0)
                st.markdown(f"<div class='panel'><div class='tinycaps soft'>Current Equity</div><div style='font-size:28px;font-weight:700'>${current_equity:,.2f}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='panel'><div class='tinycaps soft'>Current Equity</div><div class='soft'>No data</div></div>", unsafe_allow_html=True)
    
    # Market-hours + Kill switch banners
    if api_available:
        if status_data.get("kill_switch_active", False):
            st.error("🚨 **KILL SWITCH ACTIVE** - All trading operations are halted!")
        market_hours_only = status_data.get("market_hours_only", True)
        market_open_now = status_data.get("market_open_now", False)
        if market_hours_only and not market_open_now:
            st.warning("⏸️ Market is currently CLOSED. Paper/live orders will queue but not execute until the next market session.")
    
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
        # KPI cards (from reports/metrics.csv if present)
        metrics_path = reports_dir / "metrics.csv"
        try:
            if metrics_path.exists():
                mdf = pd.read_csv(metrics_path)
                if not mdf.empty:
                    m = mdf.iloc[0]
                    k1, k2, k3, k4, k5 = st.columns(5)
                    with k1:
                        st.metric("CAGR", format_percentage(100 * float(m.get("cagr", 0.0))))
                    with k2:
                        st.metric("Sharpe", f"{float(m.get('sharpe', 0.0)):.2f}")
                    with k3:
                        st.metric("Sortino", f"{float(m.get('sortino', 0.0)):.2f}")
                    with k4:
                        st.metric("Max Drawdown", format_percentage(100 * float(m.get("max_dd", 0.0))))
                    with k5:
                        st.metric("Calmar", f"{float(m.get('calmar', 0.0)):.2f}")
        except Exception:
            pass
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
    
    # Advanced ensemble / risk settings expander
    with st.expander("Advanced Ensemble & Risk Settings", expanded=False):
        # Ensemble method selection
        ensemble_cfg = cfg.get("ensemble", {}).get("meta_allocator", {})
        method = st.selectbox(
            "Meta-Allocator Method",
            options=["risk_parity", "ewma-sharpe", "hrp"],
            index=["risk_parity", "ewma-sharpe", "hrp"].index(ensemble_cfg.get("method", "risk_parity")),
            help="How to blend strategies: inverse-vol risk parity, Sharpe-based, or hierarchical risk parity"
        )
        # Risk target and caps
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            target_vol = st.number_input(
                "Target Vol (annual)",
                min_value=0.02, max_value=0.40, value=float(cfg.get("risk", {}).get("target_vol_annual", 0.12)), step=0.01,
                help="Annualized volatility target used for scaling the composite portfolio"
            )
        with col_r2:
            max_gross = st.number_input(
                "Max Gross Exposure",
                min_value=0.5, max_value=2.0, value=float(cfg.get("risk", {}).get("max_gross_exposure", 1.0)), step=0.1,
                help="Maximum gross allocation cap across all assets after scaling"
            )
        # Carry sleeve toggle
        carry_enabled = st.checkbox(
            "Enable Carry Macro Sleeve",
            value=bool(cfg.get("strategies", {}).get("carry_macro", {}).get("enabled", False)),
            help="Optional sleeve using simple carry proxies across macro ETFs"
        )
        carry_universe = st.text_input(
            "Carry Universe (comma-separated)",
            value=",".join(cfg.get("strategies", {}).get("carry_macro", {}).get("universe", ["IEF", "TLT", "DBC", "UUP"]))
        )
        # QV ETF proxies toggle and list
        st.markdown("---")
        qv_proxies_enabled = st.checkbox(
            "QV: Use ETF Proxies",
            value=bool(cfg.get("strategies", {}).get("qv_trend", {}).get("use_etf_proxies", True)),
            help="Use ETF factor proxies instead of individual stocks for QV to reduce costs and survivorship bias"
        )
        qv_proxies_list = st.text_input(
            "QV Proxies (comma-separated)",
            value=",".join(cfg.get("strategies", {}).get("qv_trend", {}).get("proxies", ["QUAL", "USMV", "MTUM", "VLUE"]))
        )

        st.markdown("---")
        crypto_enabled = st.checkbox(
            "Enable Crypto ETF Trend Sleeve",
            value=bool(cfg.get("strategies", {}).get("crypto_etf_trend", {}).get("enabled", False)),
            help="Opt-in sleeve using spot/futures Bitcoin ETFs with absolute momentum and inverse-vol weighting"
        )
        crypto_universe = st.text_input(
            "Crypto ETF Universe (comma-separated)",
            value=",".join(cfg.get("strategies", {}).get("crypto_etf_trend", {}).get("universe", ["IBIT", "FBTC", "BITO"]))
        )
        crypto_lookback = st.number_input(
            "Crypto Lookback (months)", min_value=3, max_value=12, value=int(cfg.get("strategies", {}).get("crypto_etf_trend", {}).get("lookback_months", 6)), step=1
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
        # Advanced updates
        cfg.setdefault("ensemble", {}).setdefault("meta_allocator", {})["method"] = method
        cfg.setdefault("risk", {})["target_vol_annual"] = float(target_vol)
        cfg.setdefault("risk", {})["max_gross_exposure"] = float(max_gross)
        cfg.setdefault("strategies", {}).setdefault("carry_macro", {})["enabled"] = bool(carry_enabled)
        try:
            parsed = [s.strip().upper() for s in carry_universe.split(",") if s.strip()]
        except Exception:
            parsed = ["IEF", "TLT", "DBC", "UUP"]
        cfg["strategies"]["carry_macro"]["universe"] = parsed
        # QV proxies
        cfg.setdefault("strategies", {}).setdefault("qv_trend", {})["use_etf_proxies"] = bool(qv_proxies_enabled)
        try:
            qv_parsed = [s.strip().upper() for s in qv_proxies_list.split(",") if s.strip()]
        except Exception:
            qv_parsed = ["QUAL", "USMV", "MTUM", "VLUE"]
        cfg["strategies"]["qv_trend"]["proxies"] = qv_parsed

        # Crypto ETF trend
        cfg.setdefault("strategies", {}).setdefault("crypto_etf_trend", {})["enabled"] = bool(crypto_enabled)
        try:
            cparsed = [s.strip().upper() for s in crypto_universe.split(",") if s.strip()]
        except Exception:
            cparsed = ["IBIT", "FBTC", "BITO"]
        cfg["strategies"]["crypto_etf_trend"]["universe"] = cparsed
        cfg["strategies"]["crypto_etf_trend"]["lookback_months"] = int(crypto_lookback)
        
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

elif page == "Positions & Orders":
    st.header("📦 Positions & Orders")

    pos_data = get_api_data("/positions")
    ord_data = get_api_data("/orders")
    fill_data = get_api_data("/fills")

    # Positions
    st.subheader("Positions")
    if "error" in pos_data:
        st.error(f"Failed to load positions: {pos_data['error']}")
    else:
        live_positions = pd.DataFrame(pos_data.get("live_positions", []))
        cached_positions = pd.DataFrame(pos_data.get("cached_positions", []))

        tabs = st.tabs(["Live", "Cached Snapshot"])
        with tabs[0]:
            if not live_positions.empty:
                # Normalize types
                for col in ["qty", "market_value", "unrealized_pl", "unrealized_plpc"]:
                    if col in live_positions.columns:
                        live_positions[col] = pd.to_numeric(live_positions[col], errors="coerce")
                st.dataframe(live_positions)
                # Summary
                if "market_value" in live_positions.columns:
                    total_mv = float(live_positions["market_value"].fillna(0).sum())
                    st.metric("Total Market Value", format_currency(total_mv))
            else:
                st.info("No live positions reported by broker.")

        with tabs[1]:
            if not cached_positions.empty:
                st.dataframe(cached_positions)
            else:
                st.info("No cached positions in DB yet.")

    # Orders
    st.subheader("Recent Orders")
    if "error" in ord_data:
        st.error(f"Failed to load orders: {ord_data['error']}")
    else:
        recent_orders = pd.DataFrame(ord_data.get("recent_orders", []))
        legacy_orders = pd.DataFrame(ord_data.get("legacy_orders", []))

        cols = st.columns(2)
        with cols[0]:
            st.caption("Orders (Extended)")
            if not recent_orders.empty:
                st.dataframe(recent_orders)
                # Open orders count (best-effort based on status field)
                if "status" in recent_orders.columns:
                    open_mask = recent_orders["status"].str.upper().isin([
                        "NEW", "PARTIALLY_FILLED", "ACCEPTED", "PENDING", "SUBMITTED", "OPEN"
                    ])
                    st.metric("Open Orders", int(open_mask.sum()))
            else:
                st.info("No recent extended orders.")

        with cols[1]:
            st.caption("Orders (Legacy)")
            if not legacy_orders.empty:
                st.dataframe(legacy_orders)
            else:
                st.info("No legacy orders.")

    # Fills
    st.subheader("Recent Fills")
    if "error" in fill_data:
        st.error(f"Failed to load fills: {fill_data['error']}")
    else:
        fills_df = pd.DataFrame(fill_data.get("recent_fills", []))
        if not fills_df.empty:
            st.dataframe(fills_df)
        else:
            st.info("No recent fills.")

elif page == "Performance Metrics":
    st.header("📈 Performance & Signals")

    tab1, tab2, tab3 = st.tabs(["Backtest Reports", "Diagnostics", "Live Targets"])

    # Backtest Reports
    with tab1:
        meta_weights_path = reports_dir / "weights.csv"
        metrics_path = reports_dir / "metrics.csv"
        daily_path = reports_dir / "daily_returns.csv"

        cols = st.columns(2)
        with cols[0]:
            st.subheader("Strategy Blend Weights (Backtest)")
            try:
                if meta_weights_path.exists():
                    wdf = pd.read_csv(meta_weights_path, index_col=0, parse_dates=True)
                    areaf = px.area(wdf, title="Composite Strategy Weights")
                    areaf.update_layout(height=300, font_family="Inter")
                    st.plotly_chart(areaf, use_container_width=True)
                    # Top holdings latest
                    last = wdf.iloc[-1].sort_values(ascending=False).head(10)
                    st.caption("Top 10 Holdings (Last Backtest Day)")
                    barf = px.bar(last, title="Top Holdings", labels={"value":"Weight","index":"Symbol"})
                    barf.update_layout(height=300, font_family="Inter")
                    st.plotly_chart(barf, use_container_width=True)
                    # Download buttons
                    st.download_button("Download Weights CSV", data=wdf.to_csv().encode('utf-8'), file_name="weights.csv", mime="text/csv")
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
                    st.download_button("Download Metrics CSV", data=mdf.to_csv(index=False).encode('utf-8'), file_name="metrics.csv", mime="text/csv")
                else:
                    st.info("No metrics.csv found yet.")
            except Exception as e:
                st.error(f"Failed to load metrics: {e}")

    # Diagnostics
    with tab2:
        daily_path = reports_dir / "daily_returns.csv"
        weights_path = reports_dir / "weights.csv"
        try:
            if daily_path.exists():
                rdf = pd.read_csv(daily_path, index_col=0, parse_dates=True).rename(columns={"daily_return": "r"})
                r = rdf["r"]
                # Rolling Sharpe 63d
                roll_mu = r.rolling(63).mean()
                roll_sigma = r.rolling(63).std(ddof=0)
                roll_sharpe = (roll_mu / roll_sigma * (252 ** 0.5)).replace([np.inf, -np.inf], np.nan)
                fig_rs = px.line(roll_sharpe, title="Rolling Sharpe (63 trading days)")
                st.plotly_chart(fig_rs, use_container_width=True)
                # Histogram of daily returns
                fig_hist = px.histogram(r, nbins=50, title="Distribution of Daily Returns")
                st.plotly_chart(fig_hist, use_container_width=True)
                st.download_button("Download Daily Returns CSV", data=rdf.to_csv().encode('utf-8'), file_name="daily_returns.csv", mime="text/csv")
            else:
                st.info("No daily_returns.csv found yet.")
        except Exception as e:
            st.error(f"Failed diagnostics: {e}")
        try:
            if weights_path.exists():
                wdf = pd.read_csv(weights_path, index_col=0, parse_dates=True)
                turnover = wdf.diff().abs().sum(axis=1)
                fig_to = px.line(turnover, title="Portfolio Turnover (approximate)")
                st.plotly_chart(fig_to, use_container_width=True)
                st.metric("Average Daily Turnover", f"{turnover.mean():.3f}")
                st.download_button("Download Turnover CSV", data=turnover.to_csv().encode('utf-8'), file_name="turnover.csv", mime="text/csv")
            else:
                st.info("No weights.csv found yet.")
        except Exception as e:
            st.error(f"Failed turnover calc: {e}")

    # Live Targets
    with tab3:
        st.subheader("Live Targets & Signals (Latest)")
        with st.spinner("Loading live targets..."):
            targets_data = get_api_data("/targets")
        if "target_weights" in targets_data and targets_data["target_weights"]:
            lw_col, mw_col = st.columns(2)
            with lw_col:
                st.caption("Composite Target Weights")
                tdf = pd.DataFrame(list(targets_data["target_weights"].items()), columns=["Symbol", "Weight"]).set_index("Symbol")
                bar1 = px.bar(tdf, labels={"value":"Weight","index":"Symbol"})
                bar1.update_layout(height=280, font_family="Inter")
                st.plotly_chart(bar1, use_container_width=True)
            with mw_col:
                st.caption("Meta Weights by Strategy")
                mw = targets_data.get("meta_weights", {})
                if mw:
                    mw_df = pd.DataFrame(list(mw.items()), columns=["Strategy", "Weight"]).set_index("Strategy")
                    bar2 = px.bar(mw_df, labels={"value":"Weight","index":"Strategy"})
                    bar2.update_layout(height=280, font_family="Inter")
                    st.plotly_chart(bar2, use_container_width=True)
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

    # Quick maintenance actions
    st.subheader("🧹 Maintenance")
    colm1, colm2 = st.columns(2)
    with colm1:
        if st.button("Cancel All Open Orders", type="secondary"):
            with st.spinner("Canceling open orders..."):
                res = post_api("/cancel_open_orders")
            if "error" in res:
                st.error(f"Failed to cancel orders: {res['error']}")
            else:
                st.success(f"Canceled {res.get('count', 0)} open orders; DB synced {res.get('db_synced', 0)} rows")
    with colm2:
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest (this may take a moment)..."):
                res = post_api("/backtest/run", payload={})
            if "error" in res:
                st.error(f"Backtest failed: {res['error']}")
            else:
                st.success("Backtest complete")
                if isinstance(res, dict):
                    st.caption(f"Final equity: {res.get('final_equity')}")
                st.info("Navigate to Performance Metrics > Backtest Reports to view the new artifacts.")

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

elif page == "Funding Planner":
    st.header("📈 Funding Planner")
    st.caption("Project balance over time with monthly deposits and expected return/volatility.")

    colp, colr = st.columns(2)
    with colp:
        start_balance = st.number_input("Starting Balance ($)", min_value=0.0, value=1000.0, step=100.0)
        monthly_deposit = st.number_input("Monthly Deposit ($)", min_value=0.0, value=200.0, step=50.0)
        months = st.number_input("Months", min_value=1, value=60, step=1)
    with colr:
        exp_annual = st.number_input("Expected Annual Return (%)", min_value=-50.0, max_value=50.0, value=8.0, step=0.5)
        vol_annual = st.number_input("Annual Volatility (%)", min_value=1.0, max_value=50.0, value=12.0, step=0.5)
        sims = st.number_input("Monte Carlo Paths", min_value=100, max_value=5000, value=500, step=100)

    # Simple monthly MC with lognormal approximation
    rng = np.random.default_rng(42)
    mu_m = (exp_annual / 100.0) / 12.0
    sigma_m = (vol_annual / 100.0) / (12.0 ** 0.5)
    dates = pd.date_range(pd.Timestamp.today().normalize(), periods=int(months)+1, freq='MS')

    paths = np.zeros((int(months)+1, int(sims)))
    paths[0, :] = start_balance
    for t in range(1, int(months)+1):
        shock = rng.normal(mu_m, sigma_m, int(sims))
        paths[t, :] = (paths[t-1, :] + monthly_deposit) * (1.0 + shock)

    median_path = np.median(paths, axis=1)
    p10 = np.percentile(paths, 10, axis=1)
    p90 = np.percentile(paths, 90, axis=1)
    dfp = pd.DataFrame({"Median": median_path, "P10": p10, "P90": p90}, index=dates)
    figp = px.line(dfp, title="Projected Balance (Median with 10–90% band)")
    figp.add_traces(px.area(dfp[["P10", "P90"]]).data)
    st.plotly_chart(figp, use_container_width=True)

    st.subheader("Key Milestones")
    goal = st.number_input("Savings Goal ($)", min_value=0.0, value=50000.0, step=1000.0)
    hit_idx = np.argmax(median_path >= goal)
    if median_path[-1] >= goal and hit_idx > 0:
        hit_date = dates[hit_idx].date()
        st.success(f"Median projection hits ${goal:,.0f} by {hit_date}")
    else:
        st.info("Median projection does not hit the goal in the selected horizon.")

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