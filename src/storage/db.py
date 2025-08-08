from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Tuple

from ..config import project_root


DB_PATH = project_root() / "db" / "trading_bot.sqlite"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
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
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                order_id INTEGER,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS equity (
                ts TEXT PRIMARY KEY,
                value REAL NOT NULL
            );
            """
        )


def insert_equity(ts: str, value: float) -> None:
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO equity(ts, value) VALUES (?, ?)", (ts, value))


