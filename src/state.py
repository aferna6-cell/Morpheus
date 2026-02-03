"""Bot run/state utilities.

State directory is used for any persistent bot state (portfolio, risk, cooldowns, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import load_json_state, save_json_state


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TradeMemory:
    """Minimal trade memory to prevent retrading/cooldowns."""

    last_trade_ts_by_market: Dict[str, str] = field(default_factory=dict)
    traded_markets: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "TradeMemory":
        raw = load_json_state(path) or {}
        return cls(
            last_trade_ts_by_market=dict(raw.get("last_trade_ts_by_market", {})),
            traded_markets=dict(raw.get("traded_markets", {})),
        )

    def save(self, path: Path) -> None:
        save_json_state(
            {
                "last_trade_ts_by_market": self.last_trade_ts_by_market,
                "traded_markets": self.traded_markets,
                "last_update": utc_now_iso(),
            },
            path,
        )


def parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None
