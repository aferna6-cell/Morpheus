"""Whale comparison report — summarize whale_compare JSONL.

Reads `state/whale_compare.jsonl` (written by src.whale_compare) and outputs:
- Overall counts: matched / unmatched / compared
- Agree / disagree / hold rates (when Morpheus eval available)
- Top wallets by number of trades observed
- Top markets by whale activity

Usage:
  python3 -m src.whale_compare_report --file state/whale_compare.jsonl --hours 24
  python3 -m src.whale_compare_report --file state/whale_compare.jsonl --since "2026-02-01T00:00:00Z"

Notes:
- This is a log summarizer (no external calls).
- It tolerates mixed event kinds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


@dataclass
class Summary:
    total_events: int = 0
    compared_events: int = 0
    matched_only_events: int = 0
    unmatched_events: int = 0

    agree: int = 0
    disagree: int = 0
    hold: int = 0
    unavailable: int = 0


def summarize(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    s = Summary()
    wallets = Counter()
    markets = Counter()

    # Disagreement analysis
    # Keyed by market_id
    disag_by_market_count = Counter()
    disag_by_market_size = defaultdict(float)
    disag_by_market_examples: Dict[str, Dict[str, Any]] = {}

    for e in events:
        s.total_events += 1
        kind = e.get("kind")

        wallet = e.get("wallet")
        if wallet:
            wallets[str(wallet).lower()] += 1

        mkt = str(e.get("match_market_id") or e.get("market_id") or "")
        if mkt:
            markets[mkt] += 1

        if kind == "whale_trade_compared":
            s.compared_events += 1
            c = e.get("compare")
            if c == "agree":
                s.agree += 1
            elif c == "disagree":
                s.disagree += 1

                # Track disagreements by market
                size = float(e.get("trade_size_usd") or 0.0)
                disag_by_market_count[mkt] += 1
                disag_by_market_size[mkt] += size

                # Keep one representative example
                if mkt and mkt not in disag_by_market_examples:
                    disag_by_market_examples[mkt] = {
                        "market_id": mkt,
                        "market_question": e.get("match_market_question") or "",
                        "trade_title": e.get("trade_title") or "",
                        "whale_side": e.get("trade_side"),
                        "morpheus_side": e.get("morpheus_side"),
                        "morpheus_p_yes": e.get("morpheus_p_yes"),
                        "morpheus_edge": e.get("morpheus_edge"),
                        "example_wallet": e.get("wallet"),
                    }

            elif c == "hold":
                s.hold += 1
            else:
                s.unavailable += 1

        elif kind == "whale_trade_matched":
            s.matched_only_events += 1
            s.unavailable += 1
        elif kind == "whale_trade_unmatched":
            s.unmatched_events += 1

    # rates
    denom = max(1, s.compared_events)
    rates = {
        "agree_rate": round(s.agree / denom, 4),
        "disagree_rate": round(s.disagree / denom, 4),
        "hold_rate": round(s.hold / denom, 4),
    }

    # Build top disagreement list
    top_disagreements = []
    for market_id, cnt in disag_by_market_count.most_common(20):
        tot = disag_by_market_size.get(market_id, 0.0)
        avg = tot / max(1, cnt)
        ex = disag_by_market_examples.get(market_id, {"market_id": market_id})
        top_disagreements.append(
            {
                **ex,
                "disagree_count": cnt,
                "total_whale_size_usd": round(tot, 2),
                "avg_whale_size_usd": round(avg, 2),
            }
        )

    return {
        "counts": {
            "events": s.total_events,
            "compared": s.compared_events,
            "matched_only": s.matched_only_events,
            "unmatched": s.unmatched_events,
            "agree": s.agree,
            "disagree": s.disagree,
            "hold": s.hold,
            "unavailable": s.unavailable,
        },
        "rates": rates,
        "top_wallets": wallets.most_common(10),
        "top_markets": markets.most_common(10),
        "top_disagreements": top_disagreements,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize whale_compare.jsonl")
    ap.add_argument("--file", default="state/whale_compare.jsonl")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--since", default=None, help="ISO timestamp override")
    args = ap.parse_args()

    path = Path(args.file)
    since: Optional[datetime] = None
    if args.since:
        since = _parse_iso(args.since)
    else:
        since = _utc_now() - timedelta(hours=float(args.hours))

    events: List[Dict[str, Any]] = []
    for e in _iter_jsonl(path):
        ts = _parse_iso(str(e.get("ts") or ""))
        if since and ts and ts < since:
            continue
        events.append(e)

    out = summarize(events)
    out["file"] = str(path)
    out["since"] = since.isoformat() if since else None
    out["included_events"] = len(events)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
