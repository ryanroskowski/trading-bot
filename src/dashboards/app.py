from __future__ import annotations

import pandas as pd
import streamlit as st
from loguru import logger

from src.config import load_config, save_config, project_root
from src.storage.db import get_conn


st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")

st.title("Trading Bot Dashboard")
cfg = load_config()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Mode", cfg.get("mode", "paper"))
with col2:
    st.metric("Profile", cfg.get("profile", "default"))
with col3:
    st.metric("Max Allocation", f"{int(cfg['risk']['max_total_allocation_pct']*100)}%")

st.subheader("Controls")
profile = st.selectbox("Profile", ["conservative", "default", "aggressive", "turbo"], index=["conservative","default","aggressive","turbo"].index(cfg.get("profile", "default")))
mode = st.selectbox("Mode", ["backtest", "paper", "live"], index=["backtest","paper","live"].index(cfg.get("mode", "paper")))
vm_enabled = st.checkbox("Enable VM-DM", value=cfg["strategies"]["vm_dual_momentum"]["enabled"])
ts_enabled = st.checkbox("Enable TSMOM-L", value=cfg["strategies"]["tsmom_macro_lite"]["enabled"])
qv_enabled = st.checkbox("Enable QV-Trend", value=cfg["strategies"]["qv_trend"]["enabled"])
on_enabled = st.checkbox("Enable Overnight Drift (research)", value=cfg["strategies"]["overnight_drift"]["enabled"]) 

if st.button("Apply"):
    cfg["profile"] = profile
    cfg["mode"] = mode
    cfg["strategies"]["vm_dual_momentum"]["enabled"] = vm_enabled
    cfg["strategies"]["tsmom_macro_lite"]["enabled"] = ts_enabled
    cfg["strategies"]["qv_trend"]["enabled"] = qv_enabled
    cfg["strategies"]["overnight_drift"]["enabled"] = on_enabled
    save_config(cfg)
    st.success("Configuration updated. The runner will pick up changes on next tick.")

st.divider()
st.subheader("Status")
refresh = st.slider("Auto-refresh seconds", 10, 60, 20)
st.caption("Auto-refreshing every N seconds")

@st.cache_data(ttl=30)
def load_equity():
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT ts, value FROM equity ORDER BY ts", conn, parse_dates=["ts"]) 
    return df

eq = load_equity()
if not eq.empty:
    st.line_chart(eq.set_index("ts")["value"], height=250)
else:
    st.info("No equity data yet. Start the live engine to populate.")


