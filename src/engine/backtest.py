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
from ..strategy.carry_macro import compute_weights as carry_weights, CarryConfig
from ..strategy.crypto_etf_trend import compute_weights as crypto_trend_weights, CryptoETFTrendConfig
from ..ensemble.meta_allocator import MetaAllocator, MetaConfig
from ..ensemble.regime import classify_regime
from ..utils.slippage import apply_costs_to_returns, apply_symbol_costs_to_returns
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
import json
import hashlib
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
    # Base returns for ETFs (close-to-close). We later consider open alignment for execution assumptions.
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
    # Optional carry sleeve using ETF proxies (e.g., bonds, commodities, FX)
    carry_cfg = CarryConfig(
        rebalance=cfg["strategies"].get("carry_macro", {}).get("rebalance", "monthly"),
        universe=[t for t in (cfg["strategies"].get("carry_macro", {}).get("universe", ["IEF", "TLT", "DBC", "UUP"])) if t in close.columns],
    )
    # Optional crypto ETF trend sleeve (e.g., IBIT, FBTC, BITO) if data available
    crypto_cfg = CryptoETFTrendConfig(
        rebalance=cfg["strategies"].get("crypto_etf_trend", {}).get("rebalance", "monthly"),
        lookback_months=int(cfg["strategies"].get("crypto_etf_trend", {}).get("lookback_months", 6)),
        universe=[t for t in (cfg["strategies"].get("crypto_etf_trend", {}).get("universe", []))],
        cash=cfg["strategies"].get("crypto_etf_trend", {}).get("cash", "BIL"),
        max_weight=float(cfg["strategies"].get("crypto_etf_trend", {}).get("max_weight", 0.10)),
    )

    w_vm = vm_weights(close, vm_cfg)
    w_ts = ts_weights(close, ts_cfg)

    # For QV-Trend, load large cap universe and fetch prices (use existing close if present)
    # Load large-cap universe for QV and fetch data; maintain a combined asset set (ETFs + stocks or ETF proxies)
    try:
        large = pd.read_csv(cfg["universe"]["large_cap_csv"])['Ticker'].tolist()
    except Exception:
        large = []
    stock_close = pd.DataFrame(index=close.index)
    stock_open = pd.DataFrame(index=open_.index)
    if large:
        try:
            stock_px = fetch_yf_ohlcv(large)
            stock_close = stock_px.close.dropna(how="all")
            stock_open = stock_px.open.dropna(how="all")
        except Exception:
            stock_close = pd.DataFrame(index=close.index)
            stock_open = pd.DataFrame(index=open_.index)
    # Optionally use ETF proxies for QV to avoid survivorship issues
    qv_use_proxies = bool(cfg["strategies"]["qv_trend"].get("use_etf_proxies", False))
    qv_proxies = [t for t in cfg["strategies"]["qv_trend"].get("proxies", []) or []]
    present = []
    present_proxies: List[str] = []
    proxy_close = pd.DataFrame(index=close.index)
    proxy_open = pd.DataFrame(index=open_.index)

    if qv_use_proxies and qv_proxies:
        try:
            proxy_px = fetch_yf_ohlcv(qv_proxies)
            proxy_close = proxy_px.close.dropna(how="all")
            proxy_open = proxy_px.open.dropna(how="all")
            present_proxies = [t for t in qv_proxies if t in proxy_close.columns]
            if present_proxies:
                prices_for_qv = pd.concat([proxy_close[present_proxies].reindex_like(close), close.get(["BIL"], default=pd.DataFrame())], axis=1)
                w_qv = qv_weights(prices_for_qv, qv_cfg, universe=present_proxies)
            else:
                w_qv = pd.DataFrame(0.0, index=close.index, columns=["BIL"]) if "BIL" in close.columns else pd.DataFrame(0.0, index=close.index, columns=close.columns)
        except Exception:
            w_qv = pd.DataFrame(0.0, index=close.index, columns=["BIL"]) if "BIL" in close.columns else pd.DataFrame(0.0, index=close.index, columns=close.columns)
    else:
        present = [c for c in large if c in stock_close.columns]
        if present and len(present) >= 5:
            prices_for_qv = pd.concat([stock_close[present].reindex_like(close, method=None, copy=False), close.get(["BIL"], default=pd.DataFrame())], axis=1)
            w_qv = qv_weights(prices_for_qv, qv_cfg, universe=present)
        else:
            # No viable stock data -> QV parks to cash within its logic; keep index aligned to ETF index
            w_qv = pd.DataFrame(0.0, index=close.index, columns=["BIL"]) if "BIL" in close.columns else pd.DataFrame(0.0, index=close.index, columns=close.columns)

    # Compute overnight drift only if enabled
    if cfg["strategies"]["overnight_drift_demo"]["enabled"]:
        w_on = on_weights(close["SPY"], open_["SPY"], on_cfg) if "SPY" in close.columns else pd.DataFrame(0.0, index=close.index, columns=["SPY"])
    else:
        w_on = pd.DataFrame(0.0, index=close.index, columns=["SPY"])

    # Compute carry weights if enabled and universe present
    if cfg["strategies"].get("carry_macro", {}).get("enabled", False) and len(carry_cfg.universe) > 0:
        w_carry = carry_weights(close, carry_cfg)
    else:
        w_carry = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    # Align to open for next-bar execution
    def shift_to_open(w: pd.DataFrame) -> pd.DataFrame:
        return w.shift(1).reindex(open_.index).fillna(0.0)

    w_vm_o = shift_to_open(w_vm)
    w_ts_o = shift_to_open(w_ts)
    # QV is computed on stocks (+ optional BIL). Do not drop its columns; align to open index only.
    w_qv_o = shift_to_open(w_qv.reindex(index=close.index))
    w_on_o = shift_to_open(w_on.reindex(columns=["SPY"])) if not w_on.empty else pd.DataFrame(0.0, index=open_.index, columns=["SPY"]).fillna(0.0)

    # Per-strategy returns - only include enabled strategies
    strat_weights = {}
    if cfg["strategies"]["vm_dual_momentum"]["enabled"]:
        strat_weights["vm_dm"] = w_vm_o.reindex(columns=close.columns, fill_value=0.0)
    if cfg["strategies"]["tsmom_macro_lite"]["enabled"]:
        strat_weights["tsmom"] = w_ts_o.reindex(columns=close.columns, fill_value=0.0)
    if cfg["strategies"]["qv_trend"]["enabled"]:
        # Keep QV's stock columns; we will expand the price/return panel accordingly
        strat_weights["qv_trend"] = w_qv_o
    if cfg["strategies"]["overnight_drift_demo"]["enabled"]:
        strat_weights["overnight"] = w_on_o.reindex(columns=close.columns, fill_value=0.0)
    if cfg["strategies"].get("carry_macro", {}).get("enabled", False):
        strat_weights["carry"] = shift_to_open(w_carry).reindex(columns=close.columns, fill_value=0.0)
    # Crypto ETF trend
    if cfg["strategies"].get("crypto_etf_trend", {}).get("enabled", False):
        # Fetch crypto ETF prices if they aren't already present
        crypto_syms = [s for s in crypto_cfg.universe if s not in close.columns]
        crypto_close = pd.DataFrame(index=close.index)
        if crypto_syms:
            try:
                px_crypto = fetch_yf_ohlcv(list(set(crypto_cfg.universe)))
                crypto_close = px_crypto.close.dropna(how="all")
            except Exception:
                crypto_close = pd.DataFrame(index=close.index)
        # Build price panel with whatever crypto ETF columns are present
        crypto_prices = pd.concat([
            close.reindex(columns=[c for c in crypto_cfg.universe if c in close.columns]),
            crypto_close.reindex(index=close.index, columns=[c for c in crypto_cfg.universe if c in crypto_close.columns])
        ], axis=1)
        crypto_prices = crypto_prices.loc[:, ~crypto_prices.columns.duplicated()]
        if not crypto_prices.empty:
            w_crypto = crypto_trend_weights(crypto_prices, crypto_cfg)
            strat_weights["crypto"] = shift_to_open(w_crypto).reindex(columns=crypto_prices.columns, fill_value=0.0)

    per_strat_returns = {}
    # Build a combined price/return panel covering ETFs and any stock columns used by strategies
    # Union of ETFs, any strategy columns, and detected present stock tickers
    all_symbols = sorted(set(close.columns) | set(present) | set(present_proxies) | set().union(*[set(w.columns) for w in strat_weights.values()]))
    # Build a combined close panel: ETFs from close, stocks from stock_close
    combined_close = pd.concat([
        close.reindex(columns=[c for c in all_symbols if c in close.columns]),
        stock_close.reindex(index=close.index, columns=[c for c in all_symbols if c in stock_close.columns]),
        proxy_close.reindex(index=close.index, columns=[c for c in all_symbols if c in proxy_close.columns])
    ], axis=1)
    combined_close = combined_close.loc[:, ~combined_close.columns.duplicated()].reindex(columns=all_symbols)
    combined_rets = combined_close.pct_change().fillna(0.0)
    # Combined open for return convention handling
    combined_open = pd.concat([
        open_.reindex(columns=[c for c in all_symbols if c in open_.columns]),
        stock_open.reindex(index=open_.index, columns=[c for c in all_symbols if c in stock_open.columns]),
        proxy_open.reindex(index=open_.index, columns=[c for c in all_symbols if c in proxy_open.columns])
    ], axis=1)
    combined_open = combined_open.loc[:, ~combined_open.columns.duplicated()].reindex(columns=all_symbols)

    for name, w in strat_weights.items():
        # Strategy-specific returns against the combined panel
        aligned_rets = combined_rets.reindex(columns=w.columns).reindex(index=w.index).fillna(0.0)
        per_strat_returns[name] = (w.fillna(0.0) * aligned_rets).sum(axis=1)

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
    # Ensure unique columns for composite to avoid reindex duplicate error
    comp_w_cols = pd.Index(combined_close.columns).unique()
    comp_w = pd.DataFrame(0.0, index=open_.index, columns=comp_w_cols)
    for k, w in strat_weights.items():
        mw = meta_weights.get(k, pd.Series(0.0, index=open_.index)).reindex(open_.index).fillna(0.0)
        # De-duplicate strategy columns then align to composite
        w_uni = w.loc[:, ~w.columns.duplicated()]
        w_uni = w_uni.reindex(open_.index).reindex(columns=comp_w.columns, fill_value=0.0).fillna(0.0)
        # Enforce an optional cap for crypto sleeve at combination time
        if k == "crypto":
            cap = float(cfg["strategies"].get("crypto_etf_trend", {}).get("max_weight", 0.10))
            w_uni = w_uni.clip(upper=cap)
        comp_w = comp_w.add(w_uni.mul(mw.values.reshape(-1, 1)), fill_value=0.0)

    # Apply risk caps row-wise
    max_total = float(cfg["risk"]["max_total_allocation_pct"])
    max_pos = int(cfg["risk"]["max_positions_total"])
    comp_w = comp_w.apply(lambda row: cap_total_allocation(enforce_max_positions(row, max_pos), max_total), axis=1)

    # Vol targeting with drawdown de-risking
    # Estimate base portfolio returns from composite weights
    # Respect return convention from config
    convention = (cfg.get("returns", {}) or {}).get("convention", "open_to_close")
    if convention == "open_to_open":
        # r_t = Open_{t+1} / Open_t - 1
        oo = combined_open.reindex_like(comp_w)
        daily_rets_for_pnl = (oo.shift(-1) / oo - 1.0)
    elif convention == "close_to_close":
        daily_rets_for_pnl = combined_rets.reindex_like(comp_w)
    else:  # open_to_close
        oc_close = combined_close.reindex_like(comp_w)
        oc_open = combined_open.reindex_like(comp_w)
        daily_rets_for_pnl = (oc_close / oc_open - 1.0)

    base_port_ret = (comp_w.shift(0).fillna(0.0) * daily_rets_for_pnl.fillna(0.0)).sum(axis=1)
    port_vol_ann_series = realized_vol_annualized(base_port_ret, window=20).reindex(open_.index).bfill().fillna(0.0)
    target_vol = float(cfg["risk"]["target_vol_annual"])

    # Equity curve for DD de-risking uses after-cost daily returns; iterate approximately
    scale_series = pd.Series(1.0, index=open_.index)
    equity = pd.Series(1.0, index=open_.index)
    after_cost = pd.Series(0.0, index=open_.index)
    # Cost model: allow per-asset bps overrides, else fallback to global ETF bps
    bps = float(cfg["slippage"].get("bps", 0.0))
    per_asset_bps = cfg["slippage"].get("per_asset_bps", {})
    bps_series = pd.Series({sym: float(per_asset_bps.get(sym, bps)) for sym in comp_w.columns})
    dd_thresh = float(cfg["risk"].get("dd_derisk_threshold_pct", 10))
    dd_scale = float(cfg["risk"].get("dd_derisk_scale", 0.5))

    if comp_w.empty:
        raise RuntimeError("Composite weights are empty; check data alignment")
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

        # Daily return using combined returns
        # Use the selected convention's aligned daily return
        day_ret = float((w_scaled.fillna(0.0) * daily_rets_for_pnl.loc[d].reindex(w_scaled.index).fillna(0.0)).sum())
        # Costs using per-symbol bps
        daily_cost = float(((w_scaled - prev_w).abs() * (bps_series.reindex(w_scaled.index).fillna(bps) / 1e4)).sum())
        after_cost.iloc[i] = day_ret - daily_cost
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

    # Repro metadata (commit hash if available, config sha256)
    try:
        try:
            import subprocess
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cfgmod.project_root())).decode().strip()
        except Exception:
            commit = None
        cfg_text = None
        try:
            with open(cfg_path or str(cfgmod.project_root() / "config.yaml"), "rb") as f:
                data = f.read()
                cfg_text = hashlib.sha256(data).hexdigest()
        except Exception:
            pass
        run_info = {
            "git_commit": commit,
            "config_sha256": cfg_text,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        }
        with open(reports_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)
    except Exception:
        pass
    # Charts
    plt.figure(figsize=(10, 5))
    equity.plot(title="Equity Curve")
    plt.tight_layout()
    plt.savefig(reports_dir / "equity.png", dpi=150)
    plt.close()

    return BacktestResult(daily_returns=after_cost, equity_curve=equity, weights=comp_w, per_strategy_returns=per_strat_returns)


