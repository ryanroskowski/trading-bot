from __future__ import annotations

import os
import json
from typing import Optional

import httpx
from loguru import logger


def notify_discord(message: str, webhook: Optional[str] = None) -> None:
    url = webhook or os.getenv("DISCORD_WEBHOOK", "")
    if not url:
        return
    try:
        httpx.post(url, data={"content": message}, timeout=10.0)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(f"Discord notify failed: {e}")


