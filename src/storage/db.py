from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Tuple
from datetime import datetime, timedelta

from ..config import project_root


DB_PATH = project_root() / "db" / "trading_bot.sqlite"


def get_conn() -> sqlite3.Connection:
    """Open a SQLite connection.

    - Default: read/write connection (WAL for durability).
    - Fallback: read-only connection when running in environments with a read-only
      DB mount (e.g., dashboard service uses :ro).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            # Use WAL for durability in production; prefer DELETE for ephemeral test DBs
            if "test" in DB_PATH.name.lower():
                conn.execute("PRAGMA journal_mode=DELETE;")
            else:
                conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        return conn
    except sqlite3.OperationalError:
        # Read-only fallback
        try:
            return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        except Exception:
            raise


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS orders_extended (
                client_order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL,
                broker_order_id TEXT,
                submitted_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                order_id INTEGER,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fills_extended (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                fill_time TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS equity (
                ts TEXT PRIMARY KEY,
                value REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS positions (
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                market_value REAL NOT NULL,
                unrealized_pl REAL NOT NULL,
                PRIMARY KEY (ts, symbol)
            );
            CREATE TABLE IF NOT EXISTS targets (
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (ts, symbol)
            );

            CREATE TABLE IF NOT EXISTS meta_targets (
                ts TEXT NOT NULL,
                strategy TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (ts, strategy)
            );

            CREATE TABLE IF NOT EXISTS strategy_targets (
                ts TEXT NOT NULL,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (ts, strategy, symbol)
            );

            CREATE TABLE IF NOT EXISTS runtime_status (
                ts TEXT PRIMARY KEY,
                pdt_trades_today INTEGER NOT NULL,
                circuit_breaker_active INTEGER NOT NULL
            );
            
            -- Helpful indexes
            CREATE INDEX IF NOT EXISTS idx_orders_extended_client_id ON orders_extended(client_order_id);
            CREATE INDEX IF NOT EXISTS ix_orders_status ON orders(status);
            CREATE INDEX IF NOT EXISTS ix_orders_ext_status ON orders_extended(status);
            CREATE INDEX IF NOT EXISTS idx_fills_extended_client_id ON fills_extended(client_order_id);
            CREATE INDEX IF NOT EXISTS ix_fills_extended_time ON fills_extended(fill_time);
            CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity(ts);
            CREATE INDEX IF NOT EXISTS idx_positions_ts ON positions(ts);
            CREATE INDEX IF NOT EXISTS idx_targets_ts ON targets(ts);
            CREATE INDEX IF NOT EXISTS idx_meta_targets_ts ON meta_targets(ts);
            CREATE INDEX IF NOT EXISTS idx_strategy_targets_ts ON strategy_targets(ts);
            CREATE INDEX IF NOT EXISTS idx_runtime_status_ts ON runtime_status(ts);
            """
        )


def insert_equity(ts: str, value: float) -> None:
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO equity(ts, value) VALUES (?, ?)", (ts, value))


def insert_targets(ts: str, weights) -> None:
    """Insert a snapshot of target weights.

    weights is expected to be a pandas Series-like mapping symbol -> weight.
    """
    try:
        items = list(weights.items())
    except AttributeError:
        # Fallback if weights is a dict-like already
        items = list(weights)
    if not items:
        return
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO targets(ts, symbol, weight) VALUES (?, ?, ?)",
            [(ts, sym, float(w)) for sym, w in items]
        )


def insert_meta_targets(ts: str, meta_weights) -> None:
    """Insert a snapshot of meta-allocator weights per strategy."""
    if not meta_weights:
        return
    if hasattr(meta_weights, 'items'):
        items = list(meta_weights.items())
    else:
        items = [(k, v) for k, v in meta_weights]
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO meta_targets(ts, strategy, weight) VALUES (?, ?, ?)",
            [(ts, str(name), float(w)) for name, w in items]
        )


def insert_strategy_targets(ts: str, strat_to_series) -> None:
    """Insert per-strategy per-symbol weights for the latest cycle.

    Expects a mapping: strategy -> pandas Series (symbol -> weight)
    """
    if not strat_to_series:
        return
    rows = []
    for strat_name, series in strat_to_series.items():
        try:
            for sym, w in series.items():
                rows.append((ts, str(strat_name), str(sym), float(w)))
        except Exception:
            continue
    if not rows:
        return
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO strategy_targets(ts, strategy, symbol, weight)
            VALUES (?, ?, ?, ?)
            """,
            rows
        )


def insert_runtime_status(ts: str, pdt_trades_today: int, circuit_breaker_active: bool) -> None:
    """Persist runtime status markers for dashboard/API visibility."""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runtime_status(ts, pdt_trades_today, circuit_breaker_active) VALUES (?, ?, ?)",
            (ts, int(pdt_trades_today), 1 if circuit_breaker_active else 0)
        )


# ---- Order and fill reconciliation helpers ----

def get_open_orders() -> list[dict]:
    """Return open/working orders with aggregated filled quantity.

    Status considered open: NEW, PARTIALLY_FILLED, ACCEPTED, PENDING, SUBMITTED
    """
    open_statuses = ("NEW", "PARTIALLY_FILLED", "ACCEPTED", "PENDING", "SUBMITTED")
    placeholders = ",".join(["?"] * len(open_statuses))
    sql = f"""
        SELECT oe.client_order_id, oe.symbol, oe.side, oe.qty, oe.status, oe.submitted_at,
               COALESCE(SUM(fe.qty), 0.0) AS filled_qty
        FROM orders_extended AS oe
        LEFT JOIN fills_extended AS fe ON fe.client_order_id = oe.client_order_id
        WHERE oe.status IN ({placeholders})
        GROUP BY oe.client_order_id, oe.symbol, oe.side, oe.qty, oe.status, oe.submitted_at
    """
    with get_conn() as conn:
        rows = conn.execute(sql, open_statuses).fetchall()
    results = []
    for row in rows:
        results.append({
            "client_order_id": row[0],
            "symbol": row[1],
            "side": row[2],
            "qty": float(row[3]),
            "status": row[4],
            "submitted_at": row[5],
            "filled_qty": float(row[6] or 0.0),
        })
    return results


def get_recent_fills(since_minutes: int = 60) -> list[dict]:
    """Return fills in the last `since_minutes`."""
    threshold = (datetime.utcnow() - timedelta(minutes=since_minutes)).isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT client_order_id, symbol, qty, price, fill_time
            FROM fills_extended
            WHERE fill_time >= ?
            ORDER BY fill_time ASC
            """,
            (threshold,)
        ).fetchall()
    return [
        {
            "client_order_id": r[0],
            "symbol": r[1],
            "qty": float(r[2]),
            "price": float(r[3]),
            "fill_time": r[4],
        }
        for r in rows
    ]


def mark_order_closed(client_order_id: str) -> None:
    """Mark order as FILLED when its absolute filled qty >= absolute order qty."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT qty FROM orders_extended WHERE client_order_id = ?",
            (client_order_id,)
        ).fetchone()
        if not row:
            return
        qty = abs(float(row[0]))
        filled = conn.execute(
            "SELECT COALESCE(SUM(ABS(qty)), 0.0) FROM fills_extended WHERE client_order_id = ?",
            (client_order_id,)
        ).fetchone()[0]
        if float(filled or 0.0) >= qty - 1e-9:
            conn.execute(
                "UPDATE orders_extended SET status = 'FILLED' WHERE client_order_id = ?",
                (client_order_id,)
            )


def mark_order_canceled(client_order_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE orders_extended SET status = 'CANCELED' WHERE client_order_id = ?",
            (client_order_id,)
        )

