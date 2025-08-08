from __future__ import annotations

from typing import Dict, List

import pandas as pd


def fetch_basic_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """Stub for fundamentals. Returns empty DataFrame when unavailable.
    The QV-Trend strategy will gracefully degrade to price-based proxies.
    """
    columns = ["ticker", "pe", "pb", "roe", "roa", "gross_margin"]
    return pd.DataFrame(columns=columns).astype({"ticker": str})


