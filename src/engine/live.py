from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from src.config import load_config, project_root
from src.data.marketdata import fetch_yf_ohlcv
from src.execution.order_router import BrokerContext, reconcile_and_route
from src.utils.timeutils import is_market_open_now
from src.strategy.vm_dual_momentum import compute_weights as vm_weights, VMDMConfig
from src.strategy.tsmom_macro_lite import compute_weights as ts_weights, TSMOMConfig
from src.strategy.quality_value_trend import compute_weights as qv_weights, QVConfig
from src.ensemble.meta_allocator import MetaAllocator, MetaConfig
from src.utils.risk import (
    realized_vol_annualized,
    vol_target_scale,
    cap_total_allocation,
    enforce_max_positions,
    apply_drawdown_derisking,
)
from src.storage.db import init_db, insert_equity


@dataclass
class LiveContext:
    equity_usd: float
    last_config_mtime: Optional[float] = None


def run_live_loop() -> None:
    init_db()
    cfg = load_config()
    etfs = cfg["universe"]["etfs"]
    risk = cfg["risk"]
    interval = int(cfg["schedule"]["check_interval_seconds"]) or 60

    ctx = LiveContext(equity_usd=10000.0, last_config_mtime=(project_root() / "config.yaml").stat().st_mtime)
    mcfg = MetaConfig(**cfg["ensemble"]["meta_allocator"]) if cfg["ensemble"]["meta_allocator"]["enabled"] else None
    meta = MetaAllocator(mcfg) if mcfg else None

    while True:
        # Hot reload if config.yaml changed
        try:
            cfg_path = project_root() / "config.yaml"
            mtime = cfg_path.stat().st_mtime
            if ctx.last_config_mtime is None or mtime > ctx.last_config_mtime:
                cfg = load_config()
                etfs = cfg["universe"]["etfs"]
                risk = cfg["risk"]
                interval = int(cfg["schedule"]["check_interval_seconds"]) or 60
                if cfg["ensemble"]["meta_allocator"]["enabled"]:
                    mcfg = MetaConfig(**cfg["ensemble"]["meta_allocator"])
                    meta = MetaAllocator(mcfg)
                ctx.last_config_mtime = mtime
                logger.info("Detected config change; will apply on this cycle")
        except Exception as e:
            logger.warning(f"Config hot-reload check failed: {e}")

        if cfg["schedule"]["market_hours_only"] and not is_market_open_now(cfg.get("timezone", "America/New_York")):
            logger.info("Market closed; sleeping...")
            time.sleep(interval)
            continue

        # Refresh data
        px = fetch_yf_ohlcv(etfs, start="2020-01-01")
        close = px.close.dropna(how="all")
        open_ = px.open.dropna(how="all")
        rets = close.pct_change().fillna(0.0)

        # Strategy weights
        w_all = []
        if cfg["strategies"]["vm_dual_momentum"]["enabled"]:
            vm_cfg = VMDMConfig(
                lookbacks_months=cfg["strategies"]["vm_dual_momentum"]["lookbacks_months"],
                rebalance=cfg["strategies"]["vm_dual_momentum"]["rebalance"],
                abs_filter=cfg["strategies"]["vm_dual_momentum"]["abs_filter"],
                target_vol_annual=cfg["risk"]["target_vol_annual"],
            )
            w_all.append(("vm_dm", vm_weights(close, vm_cfg)))

        if cfg["strategies"]["tsmom_macro_lite"]["enabled"]:
            ts_cfg = TSMOMConfig(
                lookback_months=cfg["strategies"]["tsmom_macro_lite"]["lookback_months"],
                rebalance=cfg["strategies"]["tsmom_macro_lite"]["rebalance"],
            )
            w_all.append(("tsmom", ts_weights(close, ts_cfg)))

        if cfg["strategies"]["qv_trend"]["enabled"]:
            try:
                large = pd.read_csv(cfg["universe"]["large_cap_csv"])['Ticker'].tolist()
            except Exception:
                large = []
            if large:
                stock_px = fetch_yf_ohlcv(large)
                stock_close = stock_px.close.dropna(how="all")
                qv_cfg = QVConfig(top_n=cfg["strategies"]["qv_trend"]["top_n"], rebalance=cfg["strategies"]["qv_trend"]["rebalance"])
                w_all.append(("qv_trend", qv_weights(pd.concat([stock_close, close.get(["BIL"], default=pd.DataFrame())], axis=1), qv_cfg, universe=large)))

        # Align to open and blend via meta allocator
        strat_weights = {name: w.shift(1).reindex(open_.index).fillna(0.0).reindex(columns=close.columns, fill_value=0.0) for name, w in w_all}
        if meta:
            # compute per-strategy daily returns
            per_strat_returns = {name: (w * rets.reindex_like(w).fillna(0.0)).sum(axis=1) for name, w in strat_weights.items()}
            d = open_.index[-1]
            mw = meta.step(d, {k: v.iloc[:-1] for k, v in per_strat_returns.items()}, close)
            comp_w = pd.DataFrame(0.0, index=open_.index, columns=close.columns)
            for k, w in strat_weights.items():
                comp_w = comp_w.add(w.mul(float(mw.get(k, 0.0))), fill_value=0.0)
        else:
            # equal blend
            if strat_weights:
                comp_w = sum(w for _, w in strat_weights.items()) / max(1, len(strat_weights))
            else:
                comp_w = pd.DataFrame(0.0, index=open_.index, columns=close.columns)

        target = comp_w.iloc[-1].copy()
        # Risk overlays
        target = cap_total_allocation(target, float(risk["max_total_allocation_pct"]))
        target = enforce_max_positions(target, int(risk["max_positions_total"]))

        # Vol targeting scale
        base_port_ret = (comp_w.fillna(0.0) * rets.reindex_like(comp_w).fillna(0.0)).sum(axis=1)
        vol_ann = realized_vol_annualized(base_port_ret, window=20).iloc[-1]
        scale = vol_target_scale(float(vol_ann), float(risk["target_vol_annual"]))
        # DD derisking on simple equity memory stored in DB
        equity_proxy = (1 + base_port_ret.tail(90).fillna(0.0)).cumprod()
        derisk = apply_drawdown_derisking(equity_proxy, float(risk["dd_derisk_threshold_pct"]), float(risk["dd_derisk_scale"]))
        target = cap_total_allocation(target * scale * derisk, float(risk["max_total_allocation_pct"]))

        last_prices = close.iloc[-1]
        reconcile_and_route(
            target_weights=target,
            last_prices=last_prices,
            ctx=BrokerContext(equity_usd=ctx.equity_usd, allow_fractional=risk.get("allow_fractional", True)),
        )

        # Persist equity snapshot (placeholder; in real use, fetch from broker)
        try:
            insert_equity(pd.Timestamp.utcnow().isoformat(), float(ctx.equity_usd))
        except Exception as e:
            logger.warning(f"Equity insert failed: {e}")

        logger.info("Live loop tick complete; sleeping...")
        time.sleep(interval)


