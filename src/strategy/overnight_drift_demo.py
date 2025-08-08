from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class ONDriftConfig:
    enabled: bool
    vix_filter: bool
    vix_threshold: float


def compute_weights(spy_close: pd.Series, spy_open: pd.Series, cfg: ONDriftConfig) -> pd.DataFrame:
    # Research demo: long overnight only. Use signal at close, position at next open, exit at next open.
    idx = spy_close.index
    w = pd.Series(0.0, index=idx)
    # If enabled, set weight 1.0 at each day close (applied next open)
    if cfg.enabled:
        w.loc[:] = 1.0
    df = pd.DataFrame({"SPY": w}).shift(1).reindex(spy_open.index).fillna(0.0)
    return df


