"""Manage the master-wallet watchlist for copy trading.

Provides load/save, per-wallet stats, and a small CLI:

    python3 -m src.engines.copy_watchlist --list
    python3 -m src.engines.copy_watchlist --add 0xABC...
    python3 -m src.engines.copy_watchlist --remove 0xABC...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger()

DEFAULT_WATCHLIST_PATH = "state/copy_watchlist.json"


@dataclass
class WalletStats:
    """Per-wallet tracking stats."""

    address: str
    alias: str = ""
    added_at: str = ""
    trades_copied: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    total_pnl: float = 0.0
    last_trade_seen: str = ""

    @property
    def win_rate(self) -> float:
        total = self.trades_won + self.trades_lost
        return self.trades_won / total if total > 0 else 0.0


@dataclass
class Watchlist:
    """Collection of master wallets and their stats."""

    wallets: Dict[str, WalletStats] = field(default_factory=dict)

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path = DEFAULT_WATCHLIST_PATH) -> "Watchlist":
        """Load watchlist from disk.

        Supports multiple historical formats:
        - New format (preferred): {"wallets": {"0x..": {WalletStats...}, ...}}
        - Legacy list format: {"wallets": [{"address": "0x..", "alias": "...", ...}, ...]}
        """
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text())
            wallets: Dict[str, WalletStats] = {}

            if isinstance(raw, dict):
                w = raw.get("wallets")
                # Preferred dict-of-dicts
                if isinstance(w, dict):
                    for addr, data in w.items():
                        wallets[str(addr).lower()] = WalletStats(**data)
                    return cls(wallets=wallets)

                # Legacy list-of-objects
                if isinstance(w, list):
                    for item in w:
                        if not isinstance(item, dict):
                            continue
                        addr = str(item.get("address") or "").lower()
                        if not addr:
                            continue
                        # Map legacy fields
                        wallets[addr] = WalletStats(
                            address=addr,
                            alias=str(item.get("alias") or ""),
                            added_at=str(item.get("added_at") or item.get("addedAt") or raw.get("updated_at") or ""),
                            trades_copied=int(item.get("trades_copied") or 0),
                            trades_won=int(item.get("trades_won") or item.get("wins") or 0),
                            trades_lost=int(item.get("trades_lost") or item.get("losses") or 0),
                            total_pnl=float(item.get("total_pnl") or 0.0),
                            last_trade_seen=str(item.get("last_trade_seen") or ""),
                        )
                    return cls(wallets=wallets)

            # Unknown format
            return cls()

        except Exception as exc:
            logger.warning("watchlist_load_error", error=str(exc), path=str(p))
            return cls()

    def save(self, path: str | Path = DEFAULT_WATCHLIST_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"wallets": {addr: asdict(ws) for addr, ws in self.wallets.items()}}
        p.write_text(json.dumps(data, indent=2, default=str))

    # -- mutations ---------------------------------------------------------

    def add(self, address: str, alias: str = "") -> bool:
        """Add a wallet. Returns True if newly added."""
        addr = address.lower()
        if addr in self.wallets:
            return False
        self.wallets[addr] = WalletStats(
            address=addr,
            alias=alias,
            added_at=datetime.now(timezone.utc).isoformat(),
        )
        return True

    def remove(self, address: str) -> bool:
        """Remove a wallet. Returns True if it existed."""
        addr = address.lower()
        if addr not in self.wallets:
            return False
        del self.wallets[addr]
        return True

    def get(self, address: str) -> Optional[WalletStats]:
        return self.wallets.get(address.lower())

    def addresses(self) -> List[str]:
        return list(self.wallets.keys())

    # -- stat helpers ------------------------------------------------------

    def record_copy(self, address: str) -> None:
        ws = self.get(address)
        if ws:
            ws.trades_copied += 1
            ws.last_trade_seen = datetime.now(timezone.utc).isoformat()

    def record_outcome(self, address: str, pnl: float) -> None:
        ws = self.get(address)
        if not ws:
            return
        ws.total_pnl += pnl
        if pnl >= 0:
            ws.trades_won += 1
        else:
            ws.trades_lost += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Copy-trading watchlist manager")
    parser.add_argument("--file", default=DEFAULT_WATCHLIST_PATH, help="Watchlist JSON path")
    parser.add_argument("--add", metavar="ADDRESS", help="Add a wallet address")
    parser.add_argument("--alias", default="", help="Alias for --add")
    parser.add_argument("--remove", metavar="ADDRESS", help="Remove a wallet address")
    parser.add_argument("--list", action="store_true", help="List all wallets")
    args = parser.parse_args()

    wl = Watchlist.load(args.file)

    if args.add:
        ok = wl.add(args.add, alias=args.alias)
        wl.save(args.file)
        if ok:
            print(f"✅ Added {args.add.lower()}")
        else:
            print(f"⚠️  Already in watchlist: {args.add.lower()}")

    if args.remove:
        ok = wl.remove(args.remove)
        wl.save(args.file)
        if ok:
            print(f"🗑️  Removed {args.remove.lower()}")
        else:
            print(f"⚠️  Not found: {args.remove.lower()}")

    if args.list or (not args.add and not args.remove):
        if not wl.wallets:
            print("(empty watchlist)")
        else:
            for addr, ws in wl.wallets.items():
                alias = f" ({ws.alias})" if ws.alias else ""
                print(
                    f"  {addr}{alias}  copied={ws.trades_copied}  "
                    f"win_rate={ws.win_rate:.0%}  pnl={ws.total_pnl:+.2f}"
                )


if __name__ == "__main__":
    _cli()
