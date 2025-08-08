from __future__ import annotations

import os
import time
from loguru import logger

from ..config import load_config, project_root
from .live import run_live_loop


def run_scheduler() -> None:
    cfg = load_config()
    kill_switch = project_root() / cfg["risk"]["kill_switch_file"]
    logger.info("Starting scheduler (live loop)")
    while True:
        if kill_switch.exists():
            logger.warning("KILL_SWITCH detected; exiting scheduler.")
            break
        # For now, just run live loop directly (could schedule rebalances)
        run_live_loop()
        time.sleep(int(cfg["schedule"]["check_interval_seconds"]))


