from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load YAML config and apply profile overrides.

    Looks for `config.yaml` at project root by default.
    Also loads environment variables from `.env` if present.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = apply_profile_overrides(cfg)
    _ensure_dirs_exist()
    return cfg


def save_config(cfg: Dict[str, Any], path: Path | str | None = None) -> None:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def apply_profile_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    profile = (cfg.get("profile") or "default").lower()
    strategies = cfg.setdefault("strategies", {})
    risk = cfg.setdefault("risk", {})

    # Defaults if missing
    risk.setdefault("target_vol_annual", 0.10)
    risk.setdefault("use_leverage", False)
    risk.setdefault("max_gross_exposure", 1.0)

    if profile == "conservative":
        risk["target_vol_annual"] = 0.08
        # longer lookbacks, monthly rebalances
        strategies.setdefault("vm_dual_momentum", {}).update({
            "rebalance": "monthly",
            "lookbacks_months": [12],
        })
        strategies.setdefault("tsmom_macro_lite", {}).update({
            "rebalance": "monthly",
            "lookback_months": 12,
        })
    elif profile == "aggressive":
        # allow weekly rebalances and shorter lookbacks
        risk["target_vol_annual"] = 0.18
        strategies.setdefault("vm_dual_momentum", {}).update({
            "rebalance": "weekly",
            "lookbacks_months": [3, 6],
        })
        strategies.setdefault("tsmom_macro_lite", {}).update({
            "rebalance": "weekly",
            "lookback_months": 6,
        })
    elif profile == "turbo":
        # aggressive + leverage + leveraged ETFs
        risk["target_vol_annual"] = 0.20
        risk["use_leverage"] = True
        risk["max_gross_exposure"] = max(1.2, float(risk.get("max_gross_exposure", 1.2)))
        strategies.setdefault("vm_dual_momentum", {}).update({
            "rebalance": strategies.get("vm_dual_momentum", {}).get("rebalance", "weekly"),
            "lookbacks_months": strategies.get("vm_dual_momentum", {}).get("lookbacks_months", [3, 6]),
        })
        strategies.setdefault("tsmom_macro_lite", {}).update({
            "rebalance": strategies.get("tsmom_macro_lite", {}).get("rebalance", "weekly"),
            "lookback_months": strategies.get("tsmom_macro_lite", {}).get("lookback_months", 6),
        })
    else:
        # default profile: keep config.yaml values
        strategies.setdefault("vm_dual_momentum", {}).setdefault("rebalance", "monthly")
        strategies.setdefault("tsmom_macro_lite", {}).setdefault("rebalance", "monthly")

    return cfg


def _ensure_dirs_exist() -> None:
    for rel in ("data", "db", "logs", "reports"):
        (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)


def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def project_root() -> Path:
    return PROJECT_ROOT


