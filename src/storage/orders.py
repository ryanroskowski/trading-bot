from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Dict, Any

from .db import get_conn


def record_order(ts: datetime, symbol: str, side: Literal["buy", "sell"], qty: float, status: str = "submitted") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO orders(ts, symbol, side, qty, status) VALUES (?, ?, ?, ?, ?)",
            (ts.isoformat(), symbol, side, qty, status),
        )


def save_order(
    client_order_id: str,
    symbol: str,
    qty: float,
    side: str,
    order_type: str = "market",
    status: str = "submitted",
    broker_order_id: Optional[str] = None,
    submitted_at: Optional[str] = None
) -> None:
    """Save order to database with client order ID for idempotency."""
    if submitted_at is None:
        submitted_at = datetime.now().isoformat()
        
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO orders_extended 
            (client_order_id, symbol, qty, side, order_type, status, broker_order_id, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (client_order_id, symbol, qty, side, order_type, status, broker_order_id, submitted_at))


def get_order_by_client_id(client_order_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve order by client order ID for idempotency checking."""
    with get_conn() as conn:
        cursor = conn.execute("""
            SELECT client_order_id, symbol, qty, side, order_type, status, broker_order_id, submitted_at
            FROM orders_extended 
            WHERE client_order_id = ?
        """, (client_order_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                'client_order_id': row[0],
                'symbol': row[1], 
                'qty': row[2],
                'side': row[3],
                'order_type': row[4],
                'status': row[5],
                'broker_order_id': row[6],
                'submitted_at': row[7]
            }
        return None


def record_fill(ts: datetime, order_id: int | None, symbol: str, qty: float, price: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO fills(ts, order_id, symbol, qty, price) VALUES (?, ?, ?, ?, ?)",
            (ts.isoformat(), order_id, symbol, qty, price),
        )


def save_fill(
    client_order_id: str,
    symbol: str,
    qty: float,
    price: float,
    fill_time: Optional[str] = None
) -> None:
    """Save fill data linked to client order ID."""
    if fill_time is None:
        fill_time = datetime.now().isoformat()
        
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO fills_extended 
            (client_order_id, symbol, qty, price, fill_time)
            VALUES (?, ?, ?, ?, ?)
        """, (client_order_id, symbol, qty, price, fill_time))


