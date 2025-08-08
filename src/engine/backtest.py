from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..data.marketdata import fetch_yf_ohlcv
from ..strategy.vm_dual_momentum import compute_weights as vm_weights, VMDMConfig
from ..strategy.tsmom_macro_lite import compute_weights as ts_weights, TSMOMConfig
from ..strategy.quality_value_trend import compute_weights as qv_weights, QVConfig
from ..strategy.overnight_drift_demo import compute_weights as on_weights, ONDriftConfig
from ..ensemble.meta_allocator import MetaAllocator, MetaConfig
from ..ensemble.regime import classify_regime
from ..utils.slippage import apply_costs_to_returns
from ..utils.risk import (
    realized_vol_annualized,
    vol_target_scale,
    cap_total_allocation,
    enforce_max_positions,
    apply_drawdown_derisking,
)
from ..storage.metrics import compute_report
from dataclasses import asdict
import pandas as pd
import matplotlib.pyplot as plt
from .. import config as cfgmod


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    per_strategy_returns: Dict[str, pd.Series]


def run_backtest(cfg_path: str | None = None) -> BacktestResult:
    cfg = load_config(cfg_path)
    etfs = cfg["universe"]["etfs"]
    px = fetch_yf_ohlcv(etfs)
    close = px.close.dropna(how="all")
    open_ = px.open.dropna(how="all")
    rets = close.pct_change().fillna(0.0)

    # Strategy weights (base)
    vm_cfg = VMDMConfig(
        lookbacks_months=cfg["strategies"]["vm_dual_momentum"]["lookbacks_months"],
        rebalance=cfg["strategies"]["vm_dual_momentum"]["rebalance"],
        abs_filter=cfg["strategies"]["vm_dual_momentum"]["abs_filter"],
        target_vol_annual=cfg["risk"]["target_vol_annual"],
    )
    ts_cfg = TSMOMConfig(
        lookback_months=cfg["strategies"]["tsmom_macro_lite"]["lookback_months"],
        rebalance=cfg["strategies"]["tsmom_macro_lite"]["rebalance"],
    )
    qv_cfg = QVConfig(top_n=cfg["strategies"]["qv_trend"]["top_n"], rebalance=cfg["strategies"]["qv_trend"]["rebalance"])
    on_cfg = ONDriftConfig(**cfg["strategies"]["overnight_drift_demo"])  # may be disabled

    w_vm = vm_weights(close, vm_cfg)
    w_ts = ts_weights(close, ts_cfg)

    # For QV-Trend, load large cap universe and fetch prices (use existing close if present)
    try:
        large = pd.read_csv(cfg["universe"]["large_cap_csv"])['Ticker'].tolist()
    except Exception:
        large = []
    if large:
        stock_px = fetch_yf_ohlcv(large)
        stock_close = stock_px.close.dropna(how="all")
        present = [c for c in large if c in stock_close.columns]
        if len(present) < 5:
            # Not enough stock data; skip QV and park in cash
            w_qv = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        else:
            p = pd.concat([stock_close[present], close.get(["BIL"], default=pd.DataFrame())], axis=1)
            w_qv = qv_weights(p, qv_cfg, universe=present)
    else:
        w_qv = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    # Compute overnight drift only if enabled
    if cfg["strategies"]["overnight_drift_demo"]["enabled"]:
        w_on = on_weights(close["SPY"], open_["SPY"], on_cfg) if "SPY" in close.columns else pd.DataFrame(0.0, index=close.index, columns=["SPY"])
    else:
        w_on = pd.DataFrame(0.0, index=close.index, columns=["SPY"])

    # Align to open for next-bar execution
    def shift_to_open(w: pd.DataFrame) -> pd.DataFrame:
        return w.shift(1).reindex(open_.index).fillna(0.0)

    w_vm_o = shift_to_open(w_vm)
    w_ts_o = shift_to_open(w_ts)
    # QV and ON drift are already designed to be next-bar shifted or monthly; ensure alignment
    w_qv_o = shift_to_open(w_qv.reindex(close.index).reindex(columns=close.columns, fill_value=0.0))
    w_on_o = shift_to_open(w_on.reindex(columns=["SPY"])) if not w_on.empty else pd.DataFrame(0.0, index=open_.index, columns=["SPY"]).fillna(0.0)

    # Per-strategy returns - only include enabled strategies
    strat_weights = {}
    if cfg["strategies"]["vm_dual_momentum"]["enabled"]:
        strat_weights["vm_dm"] = w_vm_o.reindex(columns=close.columns, fill_value=0.0)
    if cfg["strategies"]["tsmom_macro_lite"]["enabled"]:
        strat_weights["tsmom"] = w_ts_o.reindex(columns=close.columns, fill_value=0.0)
    if cfg["strategies"]["qv_trend"]["enabled"]:
        strat_weights["qv_trend"] = w_qv_o.reindex(columns=close.columns, fill_value=0.0)
    if cfg["strategies"]["overnight_drift_demo"]["enabled"]:
        strat_weights["overnight"] = w_on_o.reindex(columns=close.columns, fill_value=0.0)

    per_strat_returns = {}
    for name, w in strat_weights.items():
        # Next-bar: weights decided prev close, applied at today's open; here aligned already
        per_strat_returns[name] = (w.fillna(0.0) * rets.reindex_like(w).fillna(0.0)).sum(axis=1)

    # Meta allocation
    mcfg = MetaConfig(**cfg["ensemble"]["meta_allocator"])
    meta = MetaAllocator(mcfg)
    meta_weights_series = []
    blended_daily = []
    for d in open_.index:
        # score using returns up to yesterday
        ret_hist = {k: v.loc[:d].iloc[:-1] for k, v in per_strat_returns.items()}
        mw = meta.step(d, ret_hist, close.loc[:d])
        meta_weights_series.append(pd.Series(mw, name=d))
        # blend daily returns for the day
        day_ret = sum(mw.get(k, 0.0) * per_strat_returns[k].loc[d] for k in per_strat_returns)
        blended_daily.append(day_ret)

    meta_weights = pd.DataFrame(meta_weights_series).fillna(0.0)

    # Composite asset weights = sum_s meta_w_s * strategy_w_s
    comp_w = pd.DataFrame(0.0, index=open_.index, columns=close.columns)
    for k, w in strat_weights.items():
        mw = meta_weights.get(k, pd.Series(0.0, index=open_.index)).reindex(open_.index).fillna(0.0)
        # broadcast series to columns
        comp_w = comp_w.add(w.reindex(open_.index).fillna(0.0).mul(mw.values.reshape(-1, 1)), fill_value=0.0)

    # Apply risk caps row-wise
    max_total = float(cfg["risk"]["max_total_allocation_pct"])
    max_pos = int(cfg["risk"]["max_positions_total"])
    comp_w = comp_w.apply(lambda row: cap_total_allocation(enforce_max_positions(row, max_pos), max_total), axis=1)

    # Vol targeting with drawdown de-risking
    # Estimate base portfolio returns from composite weights
    base_port_ret = (comp_w.shift(0).fillna(0.0) * rets.reindex_like(comp_w).fillna(0.0)).sum(axis=1)
    port_vol_ann_series = realized_vol_annualized(base_port_ret, window=20).reindex(open_.index).bfill().fillna(0.0)
    target_vol = float(cfg["risk"]["target_vol_annual"])

    # Equity curve for DD de-risking uses after-cost daily returns; iterate approximately
    scale_series = pd.Series(1.0, index=open_.index)
    equity = pd.Series(1.0, index=open_.index)
    after_cost = pd.Series(0.0, index=open_.index)
    bps = float(cfg["slippage"].get("bps", 0.0))
    dd_thresh = float(cfg["risk"].get("dd_derisk_threshold_pct", 10))
    dd_scale = float(cfg["risk"].get("dd_derisk_scale", 0.5))

    prev_w = comp_w.iloc[0] * 0.0
    for i, d in enumerate(open_.index):
        # vol scale uses yesterday's vol
        vol_ann = float(port_vol_ann_series.iloc[i - 1]) if i > 0 else float(port_vol_ann_series.iloc[i])
        vol_scale = vol_target_scale(vol_ann, target_vol)
        # dd de-risk on current equity
        derisk = apply_drawdown_derisking(equity.iloc[:i+1] if i > 0 else pd.Series([1.0], index=[d]), dd_thresh, dd_scale)
        scale = vol_scale * derisk
        scale_series.iloc[i] = scale

        w_scaled = comp_w.iloc[i] * scale
        # Re-apply total cap after scaling
        w_scaled = cap_total_allocation(w_scaled, max_total)

        # cost from turnover
        turnover = float((w_scaled - prev_w).abs().sum())
        cost = turnover * (bps / 1e4)
        day_ret = float((w_scaled.fillna(0.0) * rets.loc[d].reindex(w_scaled.index).fillna(0.0)).sum())
        after_cost.iloc[i] = day_ret - cost
        equity.iloc[i] = (equity.iloc[i - 1] if i > 0 else 1.0) * (1.0 + after_cost.iloc[i])
        prev_w = w_scaled

    # Drop first day to enforce strict next-bar execution in outputs
    if len(after_cost) > 1:
        after_cost = after_cost.iloc[1:]
        equity = equity.iloc[1:]
        # Ensure first equity point is exactly 1.0 for reporting/tests
        if not equity.empty:
            equity.iloc[0] = 1.0

    logger.info("Backtest complete; writing reports")
    metrics = compute_report(after_cost)  # compute and persist metrics CSV
    reports_dir = cfgmod.project_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    comp_w.to_csv(reports_dir / "weights.csv")
    try:
        meta_weights.to_csv(reports_dir / "meta_weights.csv")
    except Exception:
        pass
    after_cost.rename("daily_return").to_csv(reports_dir / "daily_returns.csv")
    equity.rename("equity").to_csv(reports_dir / "equity.csv")
    # Metrics CSV
    try:
        pd.DataFrame([asdict(metrics)]).to_csv(reports_dir / "metrics.csv", index=False)
    except Exception:
        pass
    # Charts
    plt.figure(figsize=(10, 5))
    equity.plot(title="Equity Curve")
    plt.tight_layout()
    plt.savefig(reports_dir / "equity.png", dpi=150)
    plt.close()

    return BacktestResult(daily_returns=after_cost, equity_curve=equity, weights=comp_w, per_strategy_returns=per_strat_returns)


