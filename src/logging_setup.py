from __future__ import annotations

from pathlib import Path
from loguru import logger

from src.config import project_root


def setup_logging() -> None:
    logs_dir = project_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "bot.log"

    logger.remove()  # reset default handlers
    logger.add(lambda msg: print(msg, end=""))
    logger.add(
        str(log_file),
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        level="INFO",
    )


