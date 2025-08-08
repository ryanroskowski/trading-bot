from __future__ import annotations

from fastapi import FastAPI
import pandas as pd

from ..config import load_config
from ..storage.db import get_conn


app = FastAPI(title="Trading Bot Status", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    cfg = load_config()
    mode = cfg.get("mode", "paper")
    profile = cfg.get("profile", "default")
    equity = None
    with get_conn() as conn:
        cur = conn.execute("SELECT ts, value FROM equity ORDER BY ts DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            equity = {"ts": row[0], "value": row[1]}
    return {"mode": mode, "profile": profile, "equity": equity}


@app.get("/equity")
def equity():
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT ts, value FROM equity ORDER BY ts", conn)
    return df.to_dict(orient="records")


@app.get("/positions")
def positions():
    # Placeholder: would read positions table if maintained; return empty list for now
    return []


@app.get("/targets")
def targets():
    # Placeholder: would return last computed target weights if persisted; return empty
    return {}


