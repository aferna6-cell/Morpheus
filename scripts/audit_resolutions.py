#!/usr/bin/env python3
"""Audit resolved trades — breakdown by date, market type, signal source.

Usage:
    python scripts/audit_resolutions.py [--state-dir state] [--since 2026-02-11]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def detect_market_type(ticker: str, title: str = "") -> str:
    """Classify market type from ticker prefix."""
    t = ticker.upper()
    if any(t.startswith(p) for p in ["KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP"]):
        return "weather"
    if any(t.startswith(p) for p in ["KXBTC", "KXETH", "KXDOGE", "KXSOL", "KXSHIB", "KXXRP",
                                       "KXBNB", "KXLTC", "KXADA", "KXDOT", "KXAVAX", "KXLINK",
                                       "KXMAT", "KXUNI"]):
        return "crypto"
    if any(t.startswith(p) for p in ["KXNBAMENTION", "KXNCAAB", "KXNFLMENTION", "KXFOXNEWS",
                                       "KXMLBMENTION", "KXNHLMENTION", "KXMLSMENTION",
                                       "KXTRUMPMENTION", "KXWOMENTION", "KXWMENTION", "KXLLM"]):
        return "mention"
    if any(t.startswith(p) for p in ["KXSPY", "KXQQQ", "KXIWM", "KXDIA"]):
        return "stock_intraday"
    if any(t.startswith(p) for p in ["KXSUPERBOWLAD", "KXRT", "KXSPOTIFY", "KXSBAD",
                                       "KXTOPSONG", "KXTOPALBUM", "KXALBUMDEBUT",
                                       "KXFIRSTSUPERBOWL", "KXAAAGASW", "KXNBAALLSTAR"]):
        return "entertainment"
    if any(t.startswith(p) for p in ["KXNASCAR", "KXF1RACE", "KXINDYRACE"]):
        return "racing"
    if "JOBLESS" in t or "INITCLAIMS" in t:
        return "economic"
    return "other"


def load_resolutions(state_dir: str) -> list:
    path = Path(state_dir) / "resolutions.jsonl"
    if not path.exists():
        print(f"No resolutions file at {path}")
        sys.exit(1)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Audit resolved trades")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--since", default=None, help="Only show trades after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    records = load_resolutions(args.state_dir)

    if args.since:
        cutoff = datetime.fromisoformat(args.since)
        filtered = []
        for r in records:
            ts = r.get("resolved_at") or r.get("timestamp") or r.get("entry_time", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.replace(tzinfo=None) >= cutoff:
                        filtered.append(r)
                except (ValueError, TypeError):
                    pass
        records = filtered

    if not records:
        print("No records found.")
        return

    # By market type
    by_type = defaultdict(lambda: {"wins": 0, "losses": 0, "flat": 0, "pnl": 0.0, "count": 0})
    # By date
    by_date = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
    # By signal source
    by_source = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})

    total_pnl = 0.0
    total_wins = 0
    total_losses = 0
    total_flat = 0

    for r in records:
        ticker = r.get("ticker", r.get("market_id", ""))
        title = r.get("title", r.get("question", ""))
        pnl = r.get("pnl_usd", 0)
        source = r.get("signal_source", "llm")
        mtype = detect_market_type(ticker, title)

        # Date extraction
        ts = r.get("resolved_at") or r.get("timestamp") or r.get("entry_time", "")
        date_str = "unknown"
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        total_pnl += pnl
        if pnl > 0:
            total_wins += 1
            by_type[mtype]["wins"] += 1
            by_date[date_str]["wins"] += 1
            by_source[source]["wins"] += 1
        elif pnl < 0:
            total_losses += 1
            by_type[mtype]["losses"] += 1
            by_date[date_str]["losses"] += 1
            by_source[source]["losses"] += 1
        else:
            total_flat += 1
            by_type[mtype]["flat"] += 1

        by_type[mtype]["pnl"] += pnl
        by_type[mtype]["count"] += 1
        by_date[date_str]["pnl"] += pnl
        by_date[date_str]["count"] += 1
        by_source[source]["pnl"] += pnl
        by_source[source]["count"] += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESOLUTION AUDIT — {len(records)} trades")
    if args.since:
        print(f"(filtered: since {args.since})")
    print(f"{'='*60}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Record: {total_wins}W / {total_losses}L / {total_flat} flat")
    if total_wins + total_losses > 0:
        print(f"Win rate: {total_wins / (total_wins + total_losses):.1%}")
    print()

    print(f"{'--- BY MARKET TYPE ---':^60}")
    print(f"{'Type':<18} {'W':>4} {'L':>4} {'F':>4} {'P&L':>10} {'Win%':>6}")
    for mtype, d in sorted(by_type.items(), key=lambda x: x[1]["pnl"]):
        w, l = d["wins"], d["losses"]
        wr = f"{w/(w+l):.0%}" if w + l > 0 else "n/a"
        print(f"{mtype:<18} {w:>4} {l:>4} {d['flat']:>4} ${d['pnl']:>9.2f} {wr:>6}")
    print()

    print(f"{'--- BY DATE ---':^60}")
    print(f"{'Date':<14} {'W':>4} {'L':>4} {'P&L':>10} {'Win%':>6}")
    for date, d in sorted(by_date.items()):
        w, l = d["wins"], d["losses"]
        wr = f"{w/(w+l):.0%}" if w + l > 0 else "n/a"
        print(f"{date:<14} {w:>4} {l:>4} ${d['pnl']:>9.2f} {wr:>6}")
    print()

    print(f"{'--- BY SIGNAL SOURCE ---':^60}")
    print(f"{'Source':<18} {'W':>4} {'L':>4} {'P&L':>10} {'Win%':>6}")
    for source, d in sorted(by_source.items(), key=lambda x: x[1]["pnl"]):
        w, l = d["wins"], d["losses"]
        wr = f"{w/(w+l):.0%}" if w + l > 0 else "n/a"
        print(f"{source:<18} {w:>4} {l:>4} ${d['pnl']:>9.2f} {wr:>6}")
    print()


if __name__ == "__main__":
    main()
