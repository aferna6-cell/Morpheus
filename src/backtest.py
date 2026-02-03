"""Morpheus Backtester — test predictions against resolved markets.

Pulls closed markets from Gamma API, feeds them to the LLM signal
as if they're live, and compares predictions to actual outcomes.

Usage:
    python3 -m src.backtest [--markets 50] [--min-volume 10000] [--category politics]

Outputs:
    - Brier score (lower = better, 0.25 = random)
    - Calibration table (predicted vs actual by decile)
    - Win rate (directional accuracy)
    - Per-market results saved to state/backtest_results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import structlog

from .markets import Market, TokenInfo, parse_outcomes, parse_outcome_prices
from .signals.llm_signal import LLMSignal
from .utils import BotConfig, load_config, setup_logging

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Fetch resolved markets from Gamma
# ---------------------------------------------------------------------------

def _parse_market(raw: dict) -> Optional[Tuple[Market, bool]]:
    """Parse a raw Gamma market dict into a Market + resolved_yes bool.
    
    Returns None if the market isn't a clean binary YES/NO with clear resolution.
    """
    outcomes = raw.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except (json.JSONDecodeError, ValueError):
            return None
    if not outcomes or len(outcomes) != 2:
        return None
    
    o_lower = [str(o).lower() for o in outcomes]
    if "yes" not in o_lower or "no" not in o_lower:
        return None

    op = raw.get("outcomePrices", "[]")
    if isinstance(op, str):
        try:
            op = json.loads(op)
        except (json.JSONDecodeError, ValueError):
            return None
    
    try:
        prices = [float(p) for p in op]
    except (ValueError, TypeError):
        return None
    
    if 1.0 not in prices or 0.0 not in prices:
        return None

    yes_idx = o_lower.index("yes")
    resolved_yes = prices[yes_idx] == 1.0

    # Build Market object
    tokens: Dict[str, TokenInfo] = {}
    clob_ids = raw.get("clobTokenIds", [])
    if isinstance(clob_ids, str):
        try:
            clob_ids = json.loads(clob_ids)
        except (json.JSONDecodeError, ValueError):
            clob_ids = []

    for i, outcome_name in enumerate(outcomes):
        token_id = clob_ids[i] if i < len(clob_ids) else f"backtest_{raw['id']}_{i}"
        # Use a price that won't anchor the LLM — None forces it to evaluate from scratch
        tokens[str(outcome_name)] = TokenInfo(
            token_id=token_id,
            outcome=str(outcome_name),
            price=0.0,  # No price info — we want raw LLM prediction ability
            volume_24h=float(raw.get("volume24hr", 0) or 0),
        )

    end_date = None
    if raw.get("endDate"):
        try:
            end_str = raw["endDate"]
            if end_str.endswith("Z"):
                end_str = end_str[:-1] + "+00:00"
            end_date = datetime.fromisoformat(end_str)
        except (ValueError, TypeError):
            pass

    market = Market(
        id=str(raw["id"]),
        question=raw.get("question", ""),
        description=raw.get("description", "") or "",
        category=raw.get("category", "") or "",
        end_date=end_date,
        volume_24h=float(raw.get("volume24hr", 0) or 0),
        liquidity=float(raw.get("liquidity", 0) or 0),
        tokens=tokens,
    )

    return market, resolved_yes


async def fetch_resolved_markets(
    *,
    max_markets: int = 200,
    min_volume: float = 5000,
    category_filter: Optional[str] = None,
) -> List[Tuple[Market, bool]]:
    """Fetch resolved binary markets from Gamma API.
    
    Returns list of (Market, resolved_yes) tuples.
    """
    results: List[Tuple[Market, bool]] = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        offset = 0
        batch_size = 100
        
        while len(results) < max_markets:
            resp = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={
                    "closed": "true",
                    "limit": batch_size,
                    "offset": offset,
                    "order": "volume",
                    "ascending": "false",
                },
            )
            resp.raise_for_status()
            batch = resp.json()
            
            if not batch:
                break
            
            for raw in batch:
                vol = float(raw.get("volume", 0) or 0)
                if vol < min_volume:
                    continue
                
                if category_filter:
                    cat = (raw.get("category") or "").lower()
                    q = (raw.get("question") or "").lower()
                    desc = (raw.get("description") or "").lower()
                    text = f"{cat} {q} {desc}"
                    if category_filter.lower() not in text:
                        continue
                
                parsed = _parse_market(raw)
                if parsed:
                    results.append(parsed)
                    if len(results) >= max_markets:
                        break
            
            offset += batch_size
            if len(batch) < batch_size:
                break
            
            # Rate limit
            await asyncio.sleep(0.5)
    
    logger.info("resolved_markets_fetched", count=len(results))
    return results


# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

async def run_backtest(
    config: BotConfig,
    markets_with_outcomes: List[Tuple[Market, bool]],
    *,
    state_dir: str = "state",
    save_results: bool = True,
) -> Dict[str, Any]:
    """Run LLM signal on resolved markets and compute accuracy metrics."""

    signal = LLMSignal(config)
    
    results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    errors = 0
    
    total = len(markets_with_outcomes)
    
    for i, (market, resolved_yes) in enumerate(markets_with_outcomes):
        logger.info(
            "backtest_evaluating",
            progress=f"{i+1}/{total}",
            market_id=market.id,
            question=market.question[:80],
        )
        
        try:
            sig_result = await signal.backtest_evaluate(market)
        except Exception as e:
            logger.warning("backtest_eval_error", market_id=market.id, error=str(e))
            errors += 1
            continue
        
        # Check if market was skipped (type filter)
        if "Skipped market type" in sig_result.reasoning:
            skipped.append({
                "market_id": market.id,
                "question": market.question,
                "reason": sig_result.reasoning,
                "resolved_yes": resolved_yes,
            })
            print(
                f"  [{i+1}/{total}] ⏭️  SKIP "
                f"actual={'YES' if resolved_yes else 'NO'} "
                f"| {market.question[:65]}"
            )
            continue
        
        predicted_p_yes = sig_result.estimated_prob
        actual_outcome = 1.0 if resolved_yes else 0.0
        
        # Brier component
        brier_component = (predicted_p_yes - actual_outcome) ** 2
        
        # Directional accuracy
        side = sig_result.recommended_side.value
        if side == "hold":
            direction_correct = None
        elif side == "buy_yes":
            direction_correct = resolved_yes
        elif side == "buy_no":
            direction_correct = not resolved_yes
        else:
            direction_correct = None
        
        result = {
            "market_id": market.id,
            "question": market.question,
            "category": market.category,
            "volume": market.volume_24h,
            "predicted_p_yes": round(predicted_p_yes, 4),
            "actual_outcome": actual_outcome,
            "brier_component": round(brier_component, 6),
            "recommended_side": side,
            "edge": round(sig_result.edge, 4),
            "confidence": round(sig_result.confidence, 4),
            "conviction": getattr(sig_result, "conviction", None),
            "direction_correct": direction_correct,
            "resolved_yes": resolved_yes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Add conviction value if present
        conv = getattr(sig_result, "conviction", None)
        if conv and hasattr(conv, "value"):
            result["conviction"] = conv.value
        
        results.append(result)
        
        # Print live progress
        correct_str = "✅" if direction_correct else ("❌" if direction_correct is False else "➖")
        print(
            f"  [{i+1}/{total}] {correct_str} p_yes={predicted_p_yes:.2f} "
            f"actual={'YES' if resolved_yes else 'NO'} "
            f"side={side} edge={sig_result.edge:+.3f} "
            f"| {market.question[:65]}"
        )
        
        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)
    
    # -----------------------------------------------------------------------
    # Compute aggregate metrics
    # -----------------------------------------------------------------------
    
    if not results:
        print("\n⚠️  No results to analyze.")
        return {"error": "no results"}
    
    # Brier score
    brier_scores = [r["brier_component"] for r in results]
    brier_score = sum(brier_scores) / len(brier_scores)
    
    # Calibration buckets (deciles)
    buckets: Dict[str, Dict] = {}
    for decile in range(0, 10):
        lo = decile / 10.0
        hi = (decile + 1) / 10.0
        label = f"{int(lo*100):>2}-{int(hi*100)}%"
        in_bucket = [r for r in results if lo <= r["predicted_p_yes"] < hi]
        if in_bucket:
            mean_predicted = sum(r["predicted_p_yes"] for r in in_bucket) / len(in_bucket)
            mean_actual = sum(r["actual_outcome"] for r in in_bucket) / len(in_bucket)
            buckets[label] = {
                "count": len(in_bucket),
                "mean_predicted": round(mean_predicted, 3),
                "mean_actual": round(mean_actual, 3),
                "gap": round(abs(mean_predicted - mean_actual), 3),
            }
    
    # Win rate (directional)
    trades = [r for r in results if r["direction_correct"] is not None]
    wins = [r for r in trades if r["direction_correct"]]
    win_rate = len(wins) / len(trades) if trades else 0.0
    holds = len(results) - len(trades)
    
    # High-conviction win rate
    high_conv = [r for r in trades if r.get("conviction") == "high"]
    high_conv_wins = [r for r in high_conv if r["direction_correct"]]
    high_conv_wr = len(high_conv_wins) / len(high_conv) if high_conv else 0.0
    
    metrics = {
        "total_markets": len(results),
        "skipped_markets": len(skipped),
        "errors": errors,
        "brier_score": round(brier_score, 4),
        "brier_vs_random": round(0.25 - brier_score, 4),  # positive = better than random
        "win_rate": round(win_rate, 4),
        "wins": len(wins),
        "losses": len(trades) - len(wins),
        "holds": holds,
        "high_conviction_win_rate": round(high_conv_wr, 4),
        "high_conviction_trades": len(high_conv),
        "calibration": buckets,
    }
    
    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("  🔴 MORPHEUS — BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\n  Markets evaluated: {metrics['total_markets']}  |  Skipped: {metrics['skipped_markets']}  |  Errors: {metrics['errors']}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}  (random = 0.2500, perfect = 0.0000)")
    
    diff = metrics["brier_vs_random"]
    if diff > 0:
        print(f"  ✅ {diff:.4f} BETTER than random")
    elif diff < 0:
        print(f"  ❌ {abs(diff):.4f} WORSE than random")
    else:
        print(f"  ➖ Exactly random")
    
    print(f"\n  Win Rate: {metrics['win_rate']:.1%} ({metrics['wins']}W / {metrics['losses']}L / {metrics['holds']} holds)")
    if high_conv:
        print(f"  High-Conviction WR: {metrics['high_conviction_win_rate']:.1%} ({len(high_conv_wins)}W / {len(high_conv) - len(high_conv_wins)}L)")
    
    # Calibration table
    if buckets:
        print(f"\n  {'Bucket':<10} {'Count':>6} {'Predicted':>10} {'Actual':>8} {'Gap':>6}")
        print(f"  {'-'*42}")
        for label, b in buckets.items():
            gap_str = f"{b['gap']:.3f}"
            print(f"  {label:<10} {b['count']:>6} {b['mean_predicted']:>10.3f} {b['mean_actual']:>8.3f} {gap_str:>6}")
    
    # Worst predictions
    sorted_by_brier = sorted(results, key=lambda r: -r["brier_component"])
    print(f"\n  Worst 5 predictions (highest Brier):")
    for r in sorted_by_brier[:5]:
        actual = "YES" if r["resolved_yes"] else "NO"
        print(
            f"    Brier={r['brier_component']:.4f} p_yes={r['predicted_p_yes']:.2f} "
            f"actual={actual} | {r['question'][:60]}"
        )
    
    # Best predictions
    sorted_by_brier_asc = sorted(results, key=lambda r: r["brier_component"])
    print(f"\n  Best 5 predictions (lowest Brier):")
    for r in sorted_by_brier_asc[:5]:
        actual = "YES" if r["resolved_yes"] else "NO"
        print(
            f"    Brier={r['brier_component']:.4f} p_yes={r['predicted_p_yes']:.2f} "
            f"actual={actual} | {r['question'][:60]}"
        )
    
    print("\n" + "=" * 70)
    
    # Save results
    if save_results:
        state_path = Path(state_dir)
        state_path.mkdir(parents=True, exist_ok=True)
        
        results_file = state_path / "backtest_results.jsonl"
        with open(results_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        
        metrics_file = state_path / "backtest_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n  Results saved to {results_file}")
        print(f"  Metrics saved to {metrics_file}")
    
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Morpheus Backtester")
    parser.add_argument("--markets", type=int, default=50,
                        help="Number of resolved markets to test (default: 50)")
    parser.add_argument("--min-volume", type=float, default=10000,
                        help="Minimum market volume in USD (default: 10000)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by category keyword (e.g., 'politics', 'bitcoin')")
    parser.add_argument("--state-dir", type=str, default="state",
                        help="Directory for saving results")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config file path")
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging(config, log_json=False)
    
    print("=" * 70)
    print("  🔴 MORPHEUS — BACKTESTER")
    print("=" * 70)
    print(f"\n  Fetching up to {args.markets} resolved markets (min vol: ${args.min_volume:,.0f})...")
    if args.category:
        print(f"  Category filter: {args.category}")
    print()
    
    async def _run():
        markets = await fetch_resolved_markets(
            max_markets=args.markets,
            min_volume=args.min_volume,
            category_filter=args.category,
        )
        
        if not markets:
            print("❌ No resolved markets found matching criteria.")
            return
        
        print(f"  Found {len(markets)} markets. Starting evaluation...\n")
        
        await run_backtest(
            config,
            markets,
            state_dir=args.state_dir,
        )
    
    asyncio.run(_run())


if __name__ == "__main__":
    main()
