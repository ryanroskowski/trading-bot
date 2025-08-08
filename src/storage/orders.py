from __future__ import annotations

from datetime import datetime
from typing import Literal

from .db import get_conn


def record_order(ts: datetime, symbol: str, side: Literal["buy", "sell"], qty: float, status: str = "submitted") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO orders(ts, symbol, side, qty, status) VALUES (?, ?, ?, ?, ?)",
            (ts.isoformat(), symbol, side, qty, status),
        )


def record_fill(ts: datetime, order_id: int | None, symbol: str, qty: float, price: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO fills(ts, order_id, symbol, qty, price) VALUES (?, ?, ?, ?, ?)",
            (ts.isoformat(), order_id, symbol, qty, price),
        )


