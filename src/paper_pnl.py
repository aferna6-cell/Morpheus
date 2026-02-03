"""Paper PnL tracker — track hypothetical profit from dry-run decisions.

Reads decision logs from runs/<run_id>/decisions.jsonl (or state/paper_trades.jsonl)
and checks current market prices to compute unrealized PnL. For resolved markets,
computes realized PnL.

Features:
- Implied entry price from decision log
- Current mid-price fetch for unrealized PnL
- Resolution check for realized PnL
- Drawdown tracking (peak-to-trough)
- Per-market and aggregate stats
- JSONL output for charting

Usage:
    python3 -m src.paper_pnl                         # scan all run dirs + state
    python3 -m src.paper_pnl --live                   # fetch live prices & update
    python3 -m src.paper_pnl --file state/paper_trades.jsonl
    python3 -m src.paper_pnl --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

GAMMA_API = "https://gamma-api.polymarket.com"
STATE_FILE = "state/paper_pnl.json"


@dataclass
class PaperTrade:
    """A single paper (simulated) trade."""

    market_id: str
    question: str
    side: str  # buy_yes, buy_no
    entry_price: float  # implied from signal
    size_usd: float
    shares: float  # size_usd / entry_price
    timestamp: str
    signal_name: str = ""
    edge: float = 0.0
    confidence: float = 0.0
    conviction: str = ""

    # Updated later
    current_price: Optional[float] = None
    resolved: bool = False
    resolution: Optional[str] = None  # "yes" or "no"
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    status: str = "open"  # open, won, lost


@dataclass
class PaperPortfolio:
    """Aggregate paper portfolio tracker."""

    trades: List[PaperTrade] = field(default_factory=list)
    total_invested: float = 0.0
    total_pnl: float = 0.0
    peak_value: float = 0.0
    max_drawdown: float = 0.0
    last_updated: str = ""

    @property
    def total_value(self) -> float:
        return self.total_invested + self.total_pnl

    @property
    def roi_pct(self) -> float:
        if self.total_invested == 0:
            return 0.0
        return (self.total_pnl / self.total_invested) * 100

    def update_drawdown(self):
        val = self.total_value
        if val > self.peak_value:
            self.peak_value = val
        if self.peak_value > 0:
            dd = (self.peak_value - val) / self.peak_value
            if dd > self.max_drawdown:
                self.max_drawdown = dd

    def save(self, path: str | Path = STATE_FILE):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_invested": self.total_invested,
            "total_pnl": self.total_pnl,
            "peak_value": self.peak_value,
            "max_drawdown": self.max_drawdown,
            "trades": [asdict(t) for t in self.trades],
        }
        p.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path = STATE_FILE) -> "PaperPortfolio":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text())
            trades = [PaperTrade(**t) for t in raw.get("trades", [])]
            return cls(
                trades=trades,
                total_invested=raw.get("total_invested", 0),
                total_pnl=raw.get("total_pnl", 0),
                peak_value=raw.get("peak_value", 0),
                max_drawdown=raw.get("max_drawdown", 0),
                last_updated=raw.get("last_updated", ""),
            )
        except Exception:
            return cls()


def _parse_decisions(path: Path) -> List[Dict[str, Any]]:
    """Parse a decisions.jsonl file."""
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                # Only include actual trade decisions (not skips)
                if r.get("action") == "trade" or r.get("side") not in (None, "", "hold"):
                    records.append(r)
            except json.JSONDecodeError:
                continue
    return records


def scan_decision_files() -> List[Dict[str, Any]]:
    """Scan all runs/*/decisions.jsonl + state/decisions.jsonl for trade decisions."""
    all_decisions = []

    # Check runs directories
    runs_dir = Path("runs")
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            dec_file = run_dir / "decisions.jsonl"
            all_decisions.extend(_parse_decisions(dec_file))

    # Also check state directory
    state_dec = Path("state") / "decisions.jsonl"
    all_decisions.extend(_parse_decisions(state_dec))

    return all_decisions


def decisions_to_trades(decisions: List[Dict[str, Any]]) -> List[PaperTrade]:
    """Convert decision log entries to PaperTrade objects."""
    trades = []
    seen_markets = set()  # dedup

    for d in decisions:
        mid = d.get("market_id", "")
        if not mid or mid in seen_markets:
            continue
        seen_markets.add(mid)

        side = d.get("side", "")
        if side not in ("buy_yes", "buy_no"):
            continue

        # Entry price: use the market price at decision time
        # For buy_yes: entry = market mid price
        # For buy_no: entry = 1 - market mid price (we're buying NO shares)
        market_price = float(d.get("avg_price") or d.get("confidence") or 0.5)
        if side == "buy_yes":
            entry = market_price
        else:
            entry = 1.0 - market_price

        # Clamp to reasonable range
        entry = max(0.01, min(0.99, entry))

        size_usd = float(d.get("executed_usd") or d.get("sizing_usd") or 10.0)
        shares = size_usd / entry if entry > 0 else 0

        trade = PaperTrade(
            market_id=mid,
            question=d.get("question", "")[:120],
            side=side,
            entry_price=entry,
            size_usd=size_usd,
            shares=shares,
            timestamp=d.get("ts", ""),
            signal_name=d.get("signal", ""),
            edge=float(d.get("edge", 0)),
            confidence=float(d.get("confidence", 0)),
            conviction=str(d.get("conviction", "")),
        )
        trades.append(trade)

    return trades


async def fetch_market_prices(
    market_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Fetch current prices and resolution status for markets."""
    results = {}
    async with httpx.AsyncClient(timeout=20.0) as client:
        # Batch fetch — Gamma supports condition_id lookup
        for mid in market_ids:
            try:
                resp = await client.get(
                    f"{GAMMA_API}/markets",
                    params={"condition_id": mid, "limit": 1},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    markets = data if isinstance(data, list) else [data]
                    if markets:
                        m = markets[0]
                        price_yes = None
                        for p in ["outcomePrices", "outcome_prices"]:
                            prices = m.get(p)
                            if prices:
                                if isinstance(prices, str):
                                    prices = json.loads(prices)
                                if isinstance(prices, list) and len(prices) >= 1:
                                    price_yes = float(prices[0])
                                    break

                        resolved = m.get("closed", False) or m.get("resolved", False)
                        resolution = None
                        if resolved:
                            # Check resolution outcome
                            res_prices = m.get("outcomePrices") or m.get("outcome_prices")
                            if res_prices:
                                if isinstance(res_prices, str):
                                    res_prices = json.loads(res_prices)
                                if isinstance(res_prices, list) and len(res_prices) >= 1:
                                    # Resolved: price[0] = 1.0 means YES won
                                    if float(res_prices[0]) >= 0.99:
                                        resolution = "yes"
                                    elif float(res_prices[0]) <= 0.01:
                                        resolution = "no"

                        results[mid] = {
                            "price_yes": price_yes,
                            "resolved": resolved,
                            "resolution": resolution,
                            "question": m.get("question", ""),
                        }
                await asyncio.sleep(0.3)  # rate limit
            except Exception:
                continue
    return results


def compute_pnl(trade: PaperTrade, market_info: Optional[Dict[str, Any]]) -> None:
    """Update trade PnL based on current market info."""
    if market_info is None:
        return

    if market_info.get("resolved"):
        trade.resolved = True
        trade.resolution = market_info.get("resolution")

        if trade.resolution:
            # Realized PnL
            if trade.side == "buy_yes":
                payout = trade.shares if trade.resolution == "yes" else 0
            else:  # buy_no
                payout = trade.shares if trade.resolution == "no" else 0

            trade.pnl_usd = payout - trade.size_usd
            trade.pnl_pct = (trade.pnl_usd / trade.size_usd * 100) if trade.size_usd > 0 else 0
            trade.status = "won" if trade.pnl_usd >= 0 else "lost"
            trade.current_price = 1.0 if (
                (trade.side == "buy_yes" and trade.resolution == "yes") or
                (trade.side == "buy_no" and trade.resolution == "no")
            ) else 0.0
    else:
        # Unrealized PnL
        price_yes = market_info.get("price_yes")
        if price_yes is not None:
            if trade.side == "buy_yes":
                trade.current_price = price_yes
                current_value = trade.shares * price_yes
            else:
                trade.current_price = 1.0 - price_yes
                current_value = trade.shares * (1.0 - price_yes)

            trade.pnl_usd = current_value - trade.size_usd
            trade.pnl_pct = (trade.pnl_usd / trade.size_usd * 100) if trade.size_usd > 0 else 0
            trade.status = "open"


def recalc_portfolio(portfolio: PaperPortfolio) -> None:
    """Recalculate aggregate stats from individual trades."""
    portfolio.total_invested = sum(t.size_usd for t in portfolio.trades)
    portfolio.total_pnl = sum(t.pnl_usd for t in portfolio.trades)
    portfolio.update_drawdown()


def print_report(portfolio: PaperPortfolio) -> None:
    print("=" * 70)
    print("📈 PAPER PnL REPORT — Morpheus Dry Run")
    print("=" * 70)
    print()

    open_trades = [t for t in portfolio.trades if t.status == "open"]
    won_trades = [t for t in portfolio.trades if t.status == "won"]
    lost_trades = [t for t in portfolio.trades if t.status == "lost"]

    print(f"💰 Portfolio Summary")
    print(f"   Total invested:  ${portfolio.total_invested:,.2f}")
    print(f"   Total PnL:       ${portfolio.total_pnl:+,.2f}")
    print(f"   ROI:             {portfolio.roi_pct:+.1f}%")
    print(f"   Peak value:      ${portfolio.peak_value:,.2f}")
    print(f"   Max drawdown:    {portfolio.max_drawdown:.1%}")
    print()

    print(f"📊 Trade Breakdown")
    print(f"   Total:    {len(portfolio.trades)}")
    print(f"   Open:     {len(open_trades)}")
    print(f"   Won:      {len(won_trades)}")
    print(f"   Lost:     {len(lost_trades)}")
    if won_trades or lost_trades:
        resolved = len(won_trades) + len(lost_trades)
        print(f"   Win rate: {len(won_trades)/resolved:.0%} ({len(won_trades)}/{resolved})")
    print()

    # Side breakdown
    yes_trades = [t for t in portfolio.trades if t.side == "buy_yes"]
    no_trades = [t for t in portfolio.trades if t.side == "buy_no"]
    print(f"📐 Side Breakdown")
    print(f"   BUY_YES: {len(yes_trades)} trades, PnL ${sum(t.pnl_usd for t in yes_trades):+,.2f}")
    print(f"   BUY_NO:  {len(no_trades)} trades, PnL ${sum(t.pnl_usd for t in no_trades):+,.2f}")
    print()

    # Individual trades
    if portfolio.trades:
        print(f"📋 Trades (sorted by PnL)")
        print(f"   {'Side':8s} {'Entry':>6s} {'Now':>6s} {'PnL':>9s} {'%':>7s} {'Status':>6s}  Market")
        print(f"   {'-'*8} {'-'*6} {'-'*6} {'-'*9} {'-'*7} {'-'*6}  {'-'*30}")

        sorted_trades = sorted(portfolio.trades, key=lambda t: -t.pnl_usd)
        for t in sorted_trades:
            now_str = f"{t.current_price:.3f}" if t.current_price is not None else "  —  "
            emoji = "✅" if t.status == "won" else "❌" if t.status == "lost" else "⏳"
            print(
                f"   {t.side:8s} {t.entry_price:6.3f} {now_str:>6s} "
                f"${t.pnl_usd:>+8.2f} {t.pnl_pct:>+6.1f}% {emoji:>6s}  "
                f"{t.question[:45]}"
            )
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Paper PnL tracker for dry-run")
    parser.add_argument("--file", default=None, help="Specific paper_pnl.json to load")
    parser.add_argument("--live", action="store_true", help="Fetch live prices to update PnL")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--import-decisions", action="store_true",
                        help="Import trades from decision logs (runs/*/decisions.jsonl)")
    args = parser.parse_args()

    state_path = args.file or STATE_FILE
    portfolio = PaperPortfolio.load(state_path)

    # Import from decision logs if requested or if portfolio is empty
    if args.import_decisions or not portfolio.trades:
        decisions = scan_decision_files()
        if decisions:
            new_trades = decisions_to_trades(decisions)
            # Merge: only add trades for markets not already tracked
            existing_markets = {t.market_id for t in portfolio.trades}
            added = 0
            for t in new_trades:
                if t.market_id not in existing_markets:
                    portfolio.trades.append(t)
                    existing_markets.add(t.market_id)
                    added += 1
            if added:
                print(f"📥 Imported {added} new trades from decision logs")

    if not portfolio.trades:
        print("❌ No paper trades found.")
        print("   Run the bot with --dry-run first, then:")
        print("   python3 -m src.paper_pnl --import-decisions --live")
        sys.exit(1)

    # Fetch live prices if requested
    if args.live:
        print("🔄 Fetching live market prices...")
        market_ids = [t.market_id for t in portfolio.trades if not t.resolved]
        if market_ids:
            prices = asyncio.run(fetch_market_prices(market_ids))
            for t in portfolio.trades:
                if t.market_id in prices:
                    compute_pnl(t, prices[t.market_id])
            print(f"   Updated {len(prices)}/{len(market_ids)} markets")
        else:
            print("   All trades resolved — nothing to fetch")

    recalc_portfolio(portfolio)
    portfolio.save(state_path)

    if args.json:
        print(json.dumps({
            "total_invested": portfolio.total_invested,
            "total_pnl": portfolio.total_pnl,
            "roi_pct": portfolio.roi_pct,
            "peak_value": portfolio.peak_value,
            "max_drawdown": portfolio.max_drawdown,
            "trade_count": len(portfolio.trades),
            "open": len([t for t in portfolio.trades if t.status == "open"]),
            "won": len([t for t in portfolio.trades if t.status == "won"]),
            "lost": len([t for t in portfolio.trades if t.status == "lost"]),
            "trades": [asdict(t) for t in portfolio.trades],
        }, indent=2, default=str))
    else:
        print_report(portfolio)


if __name__ == "__main__":
    main()
