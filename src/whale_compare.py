"""Whale comparison tool — track top traders and compare their bets vs Morpheus.

This is NOT copy-trading. It is observational: we fetch top wallets (Gamma leaderboard),
pull their recent activity, and for each significant trade we ask:

- Would Morpheus trade this market right now?
- If so, would it take the same side (BUY_YES/BUY_NO) or disagree / hold?

Outputs:
- JSONL log: state/whale_compare.jsonl

Usage:
  python3 -m src.whale_compare --once
  python3 -m src.whale_compare --watch --interval 300

Notes / caveats:
- Gamma "activity" objects are not perfectly standardized. We best-effort parse.
- Mapping whale trades to a specific Gamma market id is messy (conditionId vs id).
  We match by question/title similarity against the current market scan universe.
- Morpheus evaluation can be expensive (LLM calls). We evaluate only when we can
  confidently match a trade to a market, and we respect the LLMSignal cache.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import structlog

from .markets_scanner import MarketScanner
from .signals.llm_signal import LLMSignal
from .utils import append_jsonl, load_config, setup_logging

logger = structlog.get_logger(__name__)

DATA_API_BASE = "https://data-api.polymarket.com"
# data-api provides wallet activity; gamma endpoints are inconsistent / often 404.
ACTIVITY_URL = f"{DATA_API_BASE}/activity"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _similarity(a: str, b: str) -> float:
    a_n = _normalize(a)
    b_n = _normalize(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def _infer_side(trade_type: str, outcome_index: Any) -> Optional[str]:
    """Map Gamma activity fields to buy_yes / buy_no."""
    tt = (trade_type or "").lower()

    if outcome_index is not None:
        try:
            idx = int(outcome_index)
            if "sell" in tt:
                return "buy_no" if idx == 0 else "buy_yes"
            return "buy_yes" if idx == 0 else "buy_no"
        except (TypeError, ValueError):
            pass

    if "buy" in tt and "yes" in tt:
        return "buy_yes"
    if "buy" in tt and "no" in tt:
        return "buy_no"
    if "buy" in tt:
        return "buy_yes"
    if "sell" in tt:
        return "buy_no"

    return None


@dataclass
class WhaleTrade:
    wallet: str
    ts: str
    title: str
    trade_type: str
    outcome_index: Any
    side: Optional[str]
    size_usd: float
    raw: Dict[str, Any]


async def fetch_activity(client: httpx.AsyncClient, address: str) -> List[Dict[str, Any]]:
    """Fetch recent activity for a wallet.

    Uses data-api.polymarket.com which currently supports `activity?user=<wallet>`.
    """
    resp = await client.get(ACTIVITY_URL, params={"user": address})
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return data.get("data", data.get("activities", []))


def load_wallets_from_file(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return []

    # Format A: {"wallets": [{"address": "0x.."}, ...]}
    if isinstance(raw, dict) and isinstance(raw.get("wallets"), list):
        out = []
        for w in raw["wallets"]:
            if isinstance(w, dict) and w.get("address"):
                out.append(str(w["address"]).lower())
        return out

    # Format B: {"wallets": {"0x..": {...}, ...}}
    if isinstance(raw, dict) and isinstance(raw.get("wallets"), dict):
        return [str(a).lower() for a in raw["wallets"].keys()]

    # Format C: ["0x..", ...]
    if isinstance(raw, list):
        return [str(a).lower() for a in raw if isinstance(a, str) and a.startswith("0x")]

    return []


def parse_whale_trades(
    wallet: str,
    activity_items: List[Dict[str, Any]],
    *,
    min_trade_size_usd: float,
) -> List[WhaleTrade]:
    out: List[WhaleTrade] = []
    for item in activity_items:
        size = _safe_float(item.get("usdcSize") or item.get("amount") or 0)
        if size < min_trade_size_usd:
            continue

        title = str(item.get("title") or item.get("question") or item.get("marketTitle") or "")
        ttype = str(item.get("type") or item.get("side") or "")
        oidx = item.get("outcomeIndex")
        side = _infer_side(ttype, oidx)

        ts = str(item.get("timestamp") or item.get("createdAt") or item.get("time") or "")
        if ts and ts.isdigit():
            try:
                ts = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            except Exception:
                pass
        if not ts:
            ts = datetime.now(timezone.utc).isoformat()

        out.append(
            WhaleTrade(
                wallet=wallet,
                ts=ts,
                title=title,
                trade_type=ttype,
                outcome_index=oidx,
                side=side,
                size_usd=size,
                raw=item,
            )
        )
    return out


def match_trade_to_market(
    trade: WhaleTrade,
    markets: List[Any],
    *,
    min_sim: float = 0.78,
) -> Tuple[Optional[Any], float]:
    """Return (market, similarity) by best-match question/title."""
    if not trade.title:
        return None, 0.0

    best_m = None
    best_s = 0.0
    for m in markets:
        s = _similarity(trade.title, getattr(m, "question", ""))
        if s > best_s:
            best_s = s
            best_m = m

    if best_s < min_sim:
        return None, best_s
    return best_m, best_s


async def compare_once(
    *,
    config_path: str,
    top_n: int,
    min_trade_size_usd: float,
    out_path: str,
    all_markets: bool = False,
) -> Dict[str, Any]:
    config = load_config(config_path)

    # Optionally widen market universe for matching (useful for diagnostics).
    if all_markets:
        try:
            config.market_filters["focus_enabled"] = False
        except Exception:
            pass

    scanner = MarketScanner(config=config)

    # LLM evaluation requires OPENAI_API_KEY (or compatible env for AsyncOpenAI).
    # If missing, we still log whale trades and market matches, but skip Morpheus eval.
    import os

    llm: Optional[LLMSignal] = None
    llm_enabled = bool(os.getenv("OPENAI_API_KEY"))
    if llm_enabled:
        llm = LLMSignal(config=config)
    else:
        logger.warning("whale_compare_llm_disabled_no_api_key")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=20.0) as http:
        # Wallet discovery: use copy-trading watchlist file + config master_wallets.
        wallets: List[str] = []

        # Default watchlist file is state/copy_watchlist.json (custom format supported)
        wallets += load_wallets_from_file("state/copy_watchlist.json")

        # Also include any master_wallets from config.yaml
        masters = (getattr(config, "copy_trading", {}) or {}).get("master_wallets", [])
        wallets += [str(a).lower() for a in masters if a]

        # Dedupe / cap
        seen = set()
        uniq = []
        for a in wallets:
            if a and a.startswith("0x") and a not in seen:
                uniq.append(a)
                seen.add(a)
        wallets = uniq[:top_n]

        # Stub "leaderboard" entries
        leaderboard = [{"address": a, "pnl": 0.0, "volume": 0.0, "name": "watchlist"} for a in wallets]

        markets = await scanner.fetch_markets()
        # Don't enrich midpoints here — scanner uses Gamma prices already.

        logger.info("whale_compare_snapshot", top_wallets=len(leaderboard), markets=len(markets))

        seen_keys: set[str] = set()
        matched = 0
        evaluated = 0
        agreed = 0
        disagreed = 0
        held = 0
        skipped_no_match = 0
        skipped_no_side = 0

        for entry in leaderboard:
            addr = entry["address"]
            items = await fetch_activity(http, addr)
            trades = parse_whale_trades(addr, items, min_trade_size_usd=min_trade_size_usd)

            for t in trades:
                # Dedup key within this run
                key = f"{t.wallet}:{t.ts}:{t.size_usd}:{_normalize(t.title)[:80]}:{t.side}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                if not t.side:
                    skipped_no_side += 1
                    continue

                market, sim = match_trade_to_market(t, markets)
                if not market:
                    skipped_no_match += 1
                    append_jsonl(
                        out_file,
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "kind": "whale_trade_unmatched",
                            "wallet": t.wallet,
                            "wallet_pnl": entry.get("pnl"),
                            "wallet_volume": entry.get("volume"),
                            "trade_ts": t.ts,
                            "trade_side": t.side,
                            "trade_size_usd": round(t.size_usd, 4),
                            "trade_title": t.title,
                            "match_similarity": round(sim, 4),
                        },
                    )
                    continue

                matched += 1

                if llm_enabled and llm is not None:
                    # Evaluate Morpheus on this market (can trigger LLM call)
                    res = await llm.evaluate(market)
                    evaluated += 1

                    morpheus_side = res.recommended_side.value
                    compare = "hold"
                    if morpheus_side == "hold":
                        compare = "hold"
                        held += 1
                    else:
                        compare = "agree" if morpheus_side == t.side else "disagree"
                        if compare == "agree":
                            agreed += 1
                        else:
                            disagreed += 1

                    append_jsonl(
                        out_file,
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "kind": "whale_trade_compared",
                            "wallet": t.wallet,
                            "wallet_pnl": entry.get("pnl"),
                            "wallet_volume": entry.get("volume"),
                            "trade_ts": t.ts,
                            "trade_side": t.side,
                            "trade_size_usd": round(t.size_usd, 4),
                            "trade_title": t.title,
                            "match_market_id": getattr(market, "id", ""),
                            "match_market_question": getattr(market, "question", ""),
                            "match_similarity": round(sim, 4),
                            "morpheus_side": morpheus_side,
                            "morpheus_p_yes": round(res.estimated_prob, 4),
                            "morpheus_edge": round(res.edge, 4),
                            "morpheus_confidence": round(res.confidence, 4),
                            "compare": compare,
                        },
                    )
                else:
                    # Log match only
                    append_jsonl(
                        out_file,
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "kind": "whale_trade_matched",
                            "wallet": t.wallet,
                            "wallet_pnl": entry.get("pnl"),
                            "wallet_volume": entry.get("volume"),
                            "trade_ts": t.ts,
                            "trade_side": t.side,
                            "trade_size_usd": round(t.size_usd, 4),
                            "trade_title": t.title,
                            "match_market_id": getattr(market, "id", ""),
                            "match_market_question": getattr(market, "question", ""),
                            "match_similarity": round(sim, 4),
                            "morpheus_side": "unavailable",
                            "compare": "unavailable",
                        },
                    )

    await scanner.close()

    summary = {
        "top_wallets": top_n,
        "min_trade_size_usd": min_trade_size_usd,
        "matched": matched,
        "evaluated": evaluated,
        "agreed": agreed,
        "disagreed": disagreed,
        "held": held,
        "skipped_no_match": skipped_no_match,
        "skipped_no_side": skipped_no_side,
        "out": str(out_file),
    }
    return summary


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Compare Morpheus vs top whales")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--top", type=int, default=20, help="Top N wallets from leaderboard")
    parser.add_argument("--min-trade-size", type=float, default=250.0, help="Min whale trade size in USDC")
    parser.add_argument("--out", default="state/whale_compare.jsonl")
    parser.add_argument("--all-markets", action="store_true", help="Disable focus filter for market matching")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config, log_json=False)

    if not args.once and not args.watch:
        args.once = True

    while True:
        start = time.time()
        try:
            summary = await compare_once(
                config_path=args.config,
                top_n=args.top,
                min_trade_size_usd=args.min_trade_size,
                out_path=args.out,
                all_markets=bool(args.all_markets),
            )
            logger.info("whale_compare_done", **summary)
            print(json.dumps(summary, indent=2))

            # Auto-run whale summary report after each comparison cycle
            try:
                from .whale_summary import load_records, build_summaries, compute_globals, print_report, top_markets
                records = load_records(Path(args.out))
                wallets = build_summaries(records)
                globals_ = compute_globals(records, wallets)
                top_mkts = top_markets(records, n=10)
                print_report(wallets, globals_, top_mkts)
            except Exception as report_exc:
                logger.warning("whale_summary_report_error", error=str(report_exc))
        except Exception as exc:
            logger.error("whale_compare_error", error=str(exc))

        if args.once:
            break

        elapsed = time.time() - start
        await asyncio.sleep(max(1, args.interval - int(elapsed)))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
