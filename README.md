### Trading Bot – VectorBT/Alpaca, Backtest → Paper → Live

Production-grade, safety-first algorithmic trading system for non-technical users. One code path supports backtests, paper trading, and live trading with identical logic. Includes multiple evidence-based strategies, a meta-allocator, Streamlit dashboard, Docker deployment, and CI tests.

#### Features
- Backtest → Paper → Live with identical code paths
- Safe-by-default: paper mode, conservative limits, kill switch
- Strategies: VM-Dual Momentum (ETF), TSMOM Macro Lite, QV-Trend (stocks), Overnight Drift (research)
- Meta-Allocator blends/switches strategies by EWMA Sharpe and market regime
- Streamlit dashboard + optional FastAPI status endpoint
- SQLite storage, CSV exports, loguru logging
- Dockerfile + docker-compose for always-on running locally or on AWS Lightsail

---

### 5-minute Quickstart (Local)
1) Install Python 3.11.
2) Clone and install dependencies:
   ```bash
   cd trading-bot
   python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell
   pip install -r requirements.txt
   ```
3) Copy `.env.example` to `.env` and set Alpaca keys (paper by default).
4) Run a quick backtest (all strategies via meta-allocator):
   ```bash
   python -m src.cli.main backtest --strategy all
   ```
5) Launch the dashboard:
   ```bash
   streamlit run src/dashboards/app.py
   ```

Default mode is paper. To run the live loop (paper broker):
```bash
python -m src.cli.main live
```

Kill switch: create a file named `KILL_SWITCH` in repo root to gracefully stop the loop.

---

### Configuration & Secrets
- Edit `config.yaml` for mode, profile, risk limits, strategies, ensemble, schedule, and slippage.
- Put secrets in `.env` (see `.env.example`).

Key defaults:
- mode: paper
- profile: default
- global allocation cap: 60%
- per-trade cap: $50
- target vol: 10% annual
- fractional shares enabled

---

### Strategies
- VM-Dual Momentum (ETF rotation): relative momentum of SPY vs EFA with absolute filter and bond/cash fallback, monthly rebalance, volatility-targeted.
- TSMOM Macro Lite (cross-asset trend): 12m time-series momentum, inverse-vol risk parity, monthly rebalance, volatility-targeted.
- QV-Trend (large-cap stock picker): quality/value proxies + trend filter; picks top 10; gracefully degrades without fundamentals; monthly rebalance.
- Overnight Drift (research): buy SPY near close, sell at next open, optional VIX filter. Disabled in live by default.

All are executed with next-bar logic (signals generated with no lookahead; trades executed on the next open; slippage modeled).

---

### Meta-Allocator
Computes EWMA Sharpe over lookbacks [63, 126, 252] days with weights [0.5, 0.3, 0.2]. Applies regime tilts from breadth/vol/VIX. Enforces min hold days, cooldown, and max switches per quarter, smoothing weight changes. Global volatility targeting applied after blending.

---

### Backtesting & Metrics
- Next-bar execution and after-cost results (slippage + commissions)
- Reports written to `reports/`
- Metrics include CAGR, Sharpe, Sortino, Calmar, Max/Longest DD, exposure, turnover, DSR, and PSR
- After each run, see `reports/metrics.csv` and `reports/run_info.json` (contains git commit and config hash)

Sample after-cost backtest (illustrative):

| Strategy         | CAGR | Sharpe | Sortino | MaxDD | Calmar |
|------------------|------|--------|---------|-------|--------|
| VM-DM            | 10%  | 0.85   | 1.20    | -15%  | 0.67   |
| TSMOM-L          | 8%   | 0.80   | 1.10    | -18%  | 0.44   |
| QV-Trend         | 9%   | 0.75   | 1.05    | -20%  | 0.45   |
| Ensemble (Meta)  | 11%  | 1.00   | 1.35    | -12%  | 0.92   |

Example `reports/metrics.csv` (sample):

| cagr | sharpe | sortino | calmar | max_dd | longest_dd_days |
|------|--------|---------|--------|--------|------------------|
| 0.11 | 1.02   | 1.35    | 0.92   | -0.12  | 85               |

Note: 2025 whipsaw conditions challenge pure trend-following. The meta-allocator plus drawdown de-risking targets reduce regime pain by tilting to sturdier sleeves and lowering target volatility during stress.

---

### CLI
```bash
python -m src.cli.main backtest --strategy all        # or vm_dm, tsmom, qv_trend, overnight
python -m src.cli.main live                           # runs scheduler + live loop (paper by default)
streamlit run src/dashboards/app.py                   # dashboard
uvicorn src.api.status:app --host 0.0.0.0 --port 8000 # status API
```

Optional: tiny status API (FastAPI) can be launched from the dashboard or CLI for read-only health.

---

### Docker
Local always-on:
```bash
docker compose up -d    # bot + dashboard
```
Dashboard available at `http://localhost:8501`.
Status API at `http://localhost:8000/health`.

---

### Deploy to AWS Lightsail (Paper/Live)
1) Create a Linux Lightsail instance (Ubuntu recommended).
2) SSH in and install Docker + docker-compose.
3) Clone the repo and set environment:
   ```bash
   sudo apt-get update && sudo apt-get install -y git
   git clone https://github.com/your-org/trading-bot.git
   cd trading-bot
   cp .env.example .env   # edit ALPACA_* and any webhook
   # edit config.yaml for mode/profile as desired
   docker compose up -d
   ```
4) Expose Streamlit via reverse proxy (Caddy or Nginx). Example Caddyfile:
   ```
   your.domain.com {
     reverse_proxy dashboard:8501
   }
   ```
5) Persistence: backup/restore the `data/` and `db/` folders (SQLite + reports). See `src/storage/db.py` for the DB path.

Auto-restart: `docker compose` restarts containers by default. For extra resilience, add a systemd unit or configure restart policies in compose.

---

### Risk & PDT Notes
- Defaults are conservative: 60% cap, $50 per-trade cap, target vol 10%.
- PDT guard warns and limits intraday round-trips if equity < $25k.
- Fractional shares supported via Alpaca.
- Use the KILL_SWITCH file for immediate safe stop.

---

### Repository Layout
```
trading-bot/
  README.md
  requirements.txt
  .env.example
  config.yaml
  docker-compose.yml
  Dockerfile
  src/
    __init__.py
    config.py
    logging_setup.py
    utils/
      __init__.py
      timeutils.py
      risk.py
      slippage.py
    data/
      __init__.py
      marketdata.py
      fundamentals.py
      universe_large_cap.csv
    strategy/
      __init__.py
      vm_dual_momentum.py
      tsmom_macro_lite.py
      quality_value_trend.py
      overnight_drift_demo.py
      common.py
    ensemble/
      __init__.py
      meta_allocator.py
      regime.py
    execution/
      __init__.py
      alpaca.py
      order_router.py
    engine/
      __init__.py
      backtest.py
      live.py
      scheduler.py
    storage/
      __init__.py
      db.py
      orders.py
      metrics.py
    dashboards/
      app.py
    cli/
      __init__.py
      main.py
  tests/
    test_signals.py
    test_risk.py
    test_meta_allocator.py
  .github/workflows/ci.yml
```

---

### Paper → Live Checklist
- Paper mode running cleanly for 1–2 weeks
- Dashboard and reports reviewed
- Per-trade caps and max allocation tuned
- Discord notifications verified
- Switch `mode: live` in `config.yaml`; monitor closely

---

### License
For educational purposes only. No financial advice. You are responsible for any trading decisions.


