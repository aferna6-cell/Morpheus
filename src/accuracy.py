"""Accuracy dashboard — Brier score, calibration, win rate, edge decay, A/B.

Reads:
- state_dir/predictions.jsonl     (prediction log)
- state_dir/resolutions.jsonl     (resolution outcomes)
- state_dir/edge_decay.jsonl      (edge decay data)
- runs/*/decisions.jsonl           (legacy decisions table)

Usage:
    python3 -m src.accuracy [--state-dir state]
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
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


def _truncate(s: str, n: int = 60) -> str:
    return s[:n] + "…" if len(s) > n else s


# ---------------------------------------------------------------------------
# Brier Score & Calibration
# ---------------------------------------------------------------------------

def compute_brier_score(resolutions: List[Dict[str, Any]]) -> Optional[float]:
    """Compute Brier score: mean((predicted - actual)^2). Lower is better."""
    if not resolutions:
        return None
    total = sum(
        (float(r.get("predicted_p_yes", 0.5)) - float(r.get("actual_outcome", 0.5))) ** 2
        for r in resolutions
    )
    return total / len(resolutions)


def compute_calibration(
    resolutions: List[Dict[str, Any]], n_buckets: int = 10,
) -> List[Dict[str, Any]]:
    """Group predictions into decile buckets, compute predicted vs actual frequency."""
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for r in resolutions:
        p = float(r.get("predicted_p_yes", 0.5))
        bucket_idx = min(int(p * n_buckets), n_buckets - 1)
        buckets[bucket_idx].append(r)

    rows = []
    for i in range(n_buckets):
        items = buckets.get(i, [])
        lo = i / n_buckets
        hi = (i + 1) / n_buckets
        label = f"{lo*100:.0f}-{hi*100:.0f}%"

        if not items:
            rows.append({"bucket": label, "count": 0, "avg_predicted": None, "avg_actual": None})
            continue

        avg_pred = sum(float(r.get("predicted_p_yes", 0.5)) for r in items) / len(items)
        avg_actual = sum(float(r.get("actual_outcome", 0.5)) for r in items) / len(items)
        rows.append({
            "bucket": label,
            "count": len(items),
            "avg_predicted": round(avg_pred, 3),
            "avg_actual": round(avg_actual, 3),
        })

    return rows


def compute_win_rate(resolutions: List[Dict[str, Any]]) -> Optional[float]:
    """Win rate: % of trades where our edge was correct."""
    if not resolutions:
        return None

    wins = 0
    for r in resolutions:
        side = r.get("side", "").lower()
        actual = float(r.get("actual_outcome", 0.5))
        if "yes" in side and actual >= 0.5:
            wins += 1
        elif "no" in side and actual < 0.5:
            wins += 1

    return wins / len(resolutions)


# ---------------------------------------------------------------------------
# Edge Decay
# ---------------------------------------------------------------------------

def compute_edge_decay(state_dir: Path) -> Optional[Dict[str, Any]]:
    """Summarize edge erosion between decision and execution."""
    decay_records = _read_jsonl(state_dir / "edge_decay.jsonl")
    if not decay_records:
        return None

    edges_at_decision = []
    edges_at_execution = []
    time_deltas = []

    for r in decay_records:
        ed = float(r.get("edge_at_decision", 0))
        ee = float(r.get("edge_at_execution", 0))
        td = float(r.get("time_delta_seconds", 0))
        edges_at_decision.append(ed)
        edges_at_execution.append(ee)
        time_deltas.append(td)

    n = len(decay_records)
    avg_decision = sum(edges_at_decision) / n
    avg_execution = sum(edges_at_execution) / n
    avg_erosion = avg_decision - avg_execution
    avg_time = sum(time_deltas) / n

    return {
        "count": n,
        "avg_edge_at_decision": round(avg_decision, 4),
        "avg_edge_at_execution": round(avg_execution, 4),
        "avg_erosion": round(avg_erosion, 4),
        "avg_time_delta_seconds": round(avg_time, 1),
    }


# ---------------------------------------------------------------------------
# Legacy decisions table
# ---------------------------------------------------------------------------

def _load_decisions() -> List[Dict[str, Any]]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []

    rows: List[Dict[str, Any]] = []
    for decisions_file in sorted(runs_dir.glob("*/decisions.jsonl")):
        with open(decisions_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("side") and d.get("side") != "hold":
                    rows.append(d)
    return rows


def _print_decisions_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No trade decisions found in runs/*/decisions.jsonl")
        return

    header = (
        f"{'Market ID':<18} {'Question':<62} "
        f"{'Edge':>7} {'Net':>7} {'Side':<8} {'Conviction':<10}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        market_id = r.get("market_id", "?")[:16]
        question = _truncate(r.get("question", "?"), 60)
        edge = r.get("edge", 0.0)
        net_edge = r.get("net_edge", 0.0)
        side = r.get("side", "?")
        conviction = r.get("conviction", "?")
        print(
            f"{market_id:<18} {question:<62} "
            f"{edge:>+7.3f} {net_edge:>+7.3f} {side:<8} {conviction:<10}"
        )

    print(f"\nTotal decisions: {len(rows)}")


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def print_dashboard(state_dir: str = "state") -> None:
    """Print full accuracy dashboard."""
    state_path = Path(state_dir)

    predictions = _read_jsonl(state_path / "predictions.jsonl")
    resolutions = _read_jsonl(state_path / "resolutions.jsonl")

    total_predictions = len(predictions)
    resolved_count = len(resolutions)
    pending_count = total_predictions - resolved_count

    print("\n" + "=" * 70)
    print("  🔴 MORPHEUS — ACCURACY DASHBOARD")
    print("=" * 70)

    # Counts
    print(f"\n  Predictions: {total_predictions}  |  Resolved: {resolved_count}  |  Pending: {pending_count}")

    # Brier Score
    brier = compute_brier_score(resolutions)
    if brier is not None:
        quality = "🟢 Good" if brier < 0.15 else "🟡 Okay" if brier < 0.25 else "🔴 Poor"
        print(f"\n  Brier Score: {brier:.4f}  ({quality})")
        print(f"  (0.0 = perfect, 0.25 = random, 0.50 = always wrong)")
    else:
        print("\n  Brier Score: N/A (no resolutions yet)")

    # Win Rate
    win_rate = compute_win_rate(resolutions)
    if win_rate is not None:
        print(f"  Win Rate:    {win_rate:.1%}  ({resolved_count} resolved trades)")

    # Calibration
    if resolutions:
        print(f"\n  {'Bucket':<12} {'Count':>6} {'Predicted':>10} {'Actual':>10} {'Gap':>8}")
        print("  " + "-" * 48)
        cal = compute_calibration(resolutions)
        for row in cal:
            if row["count"] == 0:
                print(f"  {row['bucket']:<12} {row['count']:>6}")
            else:
                gap = row["avg_predicted"] - row["avg_actual"]
                print(
                    f"  {row['bucket']:<12} {row['count']:>6} "
                    f"{row['avg_predicted']:>10.3f} {row['avg_actual']:>10.3f} "
                    f"{gap:>+8.3f}"
                )

    # Edge Decay
    decay = compute_edge_decay(state_path)
    if decay:
        print(f"\n  ⏱️  EDGE DECAY ({decay['count']} trades)")
        print(f"  Avg edge at decision:  {decay['avg_edge_at_decision']:>+.4f}")
        print(f"  Avg edge at execution: {decay['avg_edge_at_execution']:>+.4f}")
        print(f"  Avg erosion:           {decay['avg_erosion']:>+.4f}")
        print(f"  Avg time delta:        {decay['avg_time_delta_seconds']:.1f}s")

    # A/B Feature Breakdown
    if resolutions:
        from .ab_tracker import brier_by_feature

        ab = brier_by_feature(resolutions)
        if ab:
            print(f"\n  🧪 A/B FEATURE BREAKDOWN")
            print(f"  {'Feature':<25} {'ON Brier':>9} {'ON n':>5} {'OFF Brier':>10} {'OFF n':>6} {'Δ':>8}")
            print("  " + "-" * 65)
            for feat, vals in ab.items():
                tb = f"{vals['true_brier']:.4f}" if vals["true_brier"] is not None else "N/A"
                fb = f"{vals['false_brier']:.4f}" if vals["false_brier"] is not None else "N/A"
                d = f"{vals['delta']:+.4f}" if vals["delta"] is not None else "N/A"
                print(
                    f"  {feat:<25} {tb:>9} {vals['true_n']:>5} "
                    f"{fb:>10} {vals['false_n']:>6} {d:>8}"
                )

    # Legacy decisions table
    decisions = _load_decisions()
    if decisions:
        print(f"\n{'—' * 70}")
        print("  📋 DECISIONS LOG (from runs/)")
        print(f"{'—' * 70}\n")
        _print_decisions_table(decisions)

    print("\n" + "=" * 70 + "\n")


def main() -> None:
    import argparse

    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Accuracy dashboard")
    parser.add_argument("--state-dir", default="state")
    args = parser.parse_args()

    print_dashboard(args.state_dir)


if __name__ == "__main__":
    main()
