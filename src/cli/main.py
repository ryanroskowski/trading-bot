from __future__ import annotations

import argparse
from loguru import logger

from src.engine.backtest import run_backtest
from src.engine.scheduler import run_scheduler
from src.logging_setup import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(prog="trading-bot", description="Backtest / Paper / Live trading bot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_backtest = sub.add_parser("backtest", help="Run backtest")
    p_backtest.add_argument("--strategy", default="all", choices=["all", "vm_dm", "tsmom", "qv_trend", "overnight"], help="Strategy to backtest")

    p_live = sub.add_parser("live", help="Run live engine (paper by default)")

    args = parser.parse_args()

    if args.cmd == "backtest":
        logger.info(f"Running backtest for: {args.strategy}")
        _ = run_backtest()
    elif args.cmd == "live":
        logger.info("Starting live scheduler")
        run_scheduler()


if __name__ == "__main__":
    main()


