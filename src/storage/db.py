from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Tuple

from ..config import project_root


DB_PATH = project_root() / "db" / "trading_bot.sqlite"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    # Use WAL for durability in production; prefer DELETE for ephemeral test DBs
    try:
        if "test" in DB_PATH.name.lower():
            conn.execute("PRAGMA journal_mode=DELETE;")
        else:
            conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    return conn


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
            CREATE INDEX IF NOT EXISTS idx_fills_extended_client_id ON fills_extended(client_order_id);
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

