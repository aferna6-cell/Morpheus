"""Whale comparison summary report.

Reads state/whale_compare.jsonl and produces a readable breakdown:
- Per-wallet stats (volume, trade count, avg size)
- Agree / disagree / hold rates vs Morpheus
- Market overlap analysis
- Shadow PnL estimate (if resolution data available)

Usage:
    python3 -m src.whale_summary
    python3 -m src.whale_summary --json
    python3 -m src.whale_summary --file state/whale_compare.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class WalletSummary:
    address: str
    total_volume: float = 0.0
    trade_count: int = 0
    avg_trade_size: float = 0.0

    # Sides
    buy_yes_count: int = 0
    buy_no_count: int = 0

    # Morpheus comparison (only for matched trades)
    matched_count: int = 0
    agree_count: int = 0
    disagree_count: int = 0
    hold_count: int = 0  # Morpheus says hold, whale trades

    # Markets
    unique_markets: set = field(default_factory=set)

    # Shadow PnL tracking
    shadow_pnl: float = 0.0
    resolved_trades: int = 0

    @property
    def agree_rate(self) -> float:
        if self.matched_count == 0:
            return 0.0
        return self.agree_count / self.matched_count

    @property
    def disagree_rate(self) -> float:
        if self.matched_count == 0:
            return 0.0
        return self.disagree_count / self.matched_count

    @property
    def hold_rate(self) -> float:
        if self.matched_count == 0:
            return 0.0
        return self.hold_count / self.matched_count


def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def build_summaries(records: List[Dict[str, Any]]) -> Dict[str, WalletSummary]:
    wallets: Dict[str, WalletSummary] = defaultdict(lambda: WalletSummary(address=""))
    all_markets_seen: set = set()

    for r in records:
        addr = r.get("wallet", "")
        if not addr:
            continue

        ws = wallets[addr]
        ws.address = addr

        size = float(r.get("trade_size_usd", 0))
        ws.total_volume += size
        ws.trade_count += 1

        side = r.get("trade_side", "")
        if side == "buy_yes":
            ws.buy_yes_count += 1
        elif side == "buy_no":
            ws.buy_no_count += 1

        title = r.get("trade_title", "")
        if title:
            ws.unique_markets.add(title)
            all_markets_seen.add(title)

        kind = r.get("kind", "")
        if kind == "whale_trade_compared":
            ws.matched_count += 1
            morpheus_side = r.get("morpheus_side", "")
            agree = r.get("agree")

            if morpheus_side == "hold":
                ws.hold_count += 1
            elif agree is True:
                ws.agree_count += 1
            elif agree is False:
                ws.disagree_count += 1

    # Compute averages
    for ws in wallets.values():
        if ws.trade_count > 0:
            ws.avg_trade_size = ws.total_volume / ws.trade_count

    return dict(wallets)


def compute_globals(
    records: List[Dict[str, Any]], wallets: Dict[str, WalletSummary]
) -> Dict[str, Any]:
    """Compute aggregate stats across all wallets."""
    total_volume = sum(ws.total_volume for ws in wallets.values())
    total_trades = sum(ws.trade_count for ws in wallets.values())
    total_matched = sum(ws.matched_count for ws in wallets.values())
    total_agree = sum(ws.agree_count for ws in wallets.values())
    total_disagree = sum(ws.disagree_count for ws in wallets.values())
    total_hold = sum(ws.hold_count for ws in wallets.values())

    all_markets = set()
    for ws in wallets.values():
        all_markets |= ws.unique_markets

    # Side bias
    total_yes = sum(ws.buy_yes_count for ws in wallets.values())
    total_no = sum(ws.buy_no_count for ws in wallets.values())

    # Time range
    timestamps = []
    for r in records:
        ts = r.get("ts") or r.get("trade_ts")
        if ts:
            timestamps.append(ts)
    time_range = ""
    if timestamps:
        timestamps.sort()
        time_range = f"{timestamps[0][:19]} → {timestamps[-1][:19]}"

    return {
        "total_volume_usd": total_volume,
        "total_trades": total_trades,
        "unique_wallets": len(wallets),
        "unique_markets": len(all_markets),
        "time_range": time_range,
        "matched_trades": total_matched,
        "agree": total_agree,
        "disagree": total_disagree,
        "hold": total_hold,
        "agree_rate": total_agree / total_matched if total_matched else 0,
        "side_bias": {
            "buy_yes": total_yes,
            "buy_no": total_no,
            "no_pct": total_no / total_trades if total_trades else 0,
        },
    }


def top_markets(records: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    """Top markets by whale volume."""
    market_vol: Dict[str, float] = defaultdict(float)
    market_count: Dict[str, int] = defaultdict(int)
    market_sides: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in records:
        title = r.get("trade_title", "")
        if not title:
            continue
        size = float(r.get("trade_size_usd", 0))
        market_vol[title] += size
        market_count[title] += 1
        side = r.get("trade_side", "")
        if side:
            market_sides[title][side] += 1

    ranked = sorted(market_vol.items(), key=lambda x: -x[1])[:n]
    return [
        {
            "market": title,
            "volume_usd": vol,
            "trade_count": market_count[title],
            "sides": dict(market_sides[title]),
        }
        for title, vol in ranked
    ]


def print_report(
    wallets: Dict[str, WalletSummary],
    globals_: Dict[str, Any],
    top_mkts: List[Dict[str, Any]],
) -> None:
    print("=" * 70)
    print("🐋 WHALE COMPARISON REPORT — Morpheus")
    print("=" * 70)
    print()

    # Global stats
    g = globals_
    print(f"📊 Overview")
    print(f"   Period:          {g['time_range']}")
    print(f"   Wallets tracked: {g['unique_wallets']}")
    print(f"   Total trades:    {g['total_trades']}")
    print(f"   Total volume:    ${g['total_volume_usd']:,.0f}")
    print(f"   Unique markets:  {g['unique_markets']}")
    print()

    # Side bias
    sb = g["side_bias"]
    print(f"📈 Whale Side Bias")
    print(f"   BUY_YES: {sb['buy_yes']}   BUY_NO: {sb['buy_no']}   NO%: {sb['no_pct']:.0%}")
    print()

    # Morpheus comparison
    print(f"🤖 Morpheus vs Whales (matched trades: {g['matched_trades']})")
    if g["matched_trades"] > 0:
        print(f"   Agree:    {g['agree']:>3d} ({g['agree_rate']:.0%})")
        print(f"   Disagree: {g['disagree']:>3d}")
        print(f"   Hold:     {g['hold']:>3d} (Morpheus wouldn't trade)")
    else:
        print("   No matched trades yet — keep the watcher running!")
    print()

    # Per-wallet breakdown
    print(f"👛 Per-Wallet Breakdown")
    print(f"   {'Wallet':<14s} {'Trades':>6s} {'Volume':>12s} {'Avg Size':>10s} {'Matched':>8s} {'Agree%':>7s}")
    print(f"   {'-'*14} {'-'*6} {'-'*12} {'-'*10} {'-'*8} {'-'*7}")
    for addr in sorted(wallets, key=lambda a: -wallets[a].total_volume):
        ws = wallets[addr]
        short = addr[:6] + "…" + addr[-4:]
        agree_pct = f"{ws.agree_rate:.0%}" if ws.matched_count > 0 else "—"
        print(
            f"   {short:<14s} {ws.trade_count:>6d} "
            f"${ws.total_volume:>11,.0f} ${ws.avg_trade_size:>9,.0f} "
            f"{ws.matched_count:>8d} {agree_pct:>7s}"
        )
    print()

    # Top markets
    print(f"🔥 Top Markets by Whale Volume")
    for i, m in enumerate(top_mkts, 1):
        sides_str = " | ".join(f"{k}:{v}" for k, v in m["sides"].items())
        print(f"   {i:>2d}. ${m['volume_usd']:>11,.0f}  ({m['trade_count']} trades, {sides_str})")
        print(f"       {m['market'][:65]}")
    print()
    print("=" * 70)


def export_json(
    wallets: Dict[str, WalletSummary],
    globals_: Dict[str, Any],
    top_mkts: List[Dict[str, Any]],
) -> str:
    wallet_list = []
    for addr, ws in wallets.items():
        wallet_list.append({
            "address": ws.address,
            "total_volume_usd": ws.total_volume,
            "trade_count": ws.trade_count,
            "avg_trade_size_usd": ws.avg_trade_size,
            "buy_yes": ws.buy_yes_count,
            "buy_no": ws.buy_no_count,
            "unique_markets": len(ws.unique_markets),
            "matched_count": ws.matched_count,
            "agree_count": ws.agree_count,
            "disagree_count": ws.disagree_count,
            "hold_count": ws.hold_count,
            "agree_rate": ws.agree_rate,
        })
    return json.dumps({
        "globals": globals_,
        "wallets": wallet_list,
        "top_markets": top_mkts,
    }, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Whale comparison summary report")
    parser.add_argument("--file", default="state/whale_compare.jsonl")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--top", type=int, default=10, help="Top N markets")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"❌ No data file found at {path}")
        print("   Run: python3 -m src.whale_compare --watch --interval 300")
        sys.exit(1)

    records = load_records(path)
    if not records:
        print("❌ No records in file")
        sys.exit(1)

    wallets = build_summaries(records)
    globals_ = compute_globals(records, wallets)
    top_mkts = top_markets(records, n=args.top)

    if args.json:
        print(export_json(wallets, globals_, top_mkts))
    else:
        print_report(wallets, globals_, top_mkts)


if __name__ == "__main__":
    main()
