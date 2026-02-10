"""Morpheus Dashboard — FastAPI backend serving JSON API + HTML UI.

Reads state files from the bot's state directory and exposes them
as JSON endpoints. Serves a single-page HTML dashboard at /.

Run: uvicorn src.dashboard:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

STATE_DIR = Path("/opt/morpheus/state")
DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"

app = FastAPI(title="Morpheus Dashboard", docs_url=None, redoc_url=None)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, return empty dict on any failure."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    """Read a JSONL file (all lines). If limit > 0, return last N."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    if limit > 0:
        records = records[-limit:]
    return records


def _bot_status() -> dict[str, Any]:
    """Check systemd service status for morpheus.service."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "morpheus.service"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        active = result.stdout.strip()
    except Exception:
        active = "unknown"

    uptime = None
    if active == "active":
        try:
            result = subprocess.run(
                ["systemctl", "show", "morpheus.service", "--property=ActiveEnterTimestamp"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            ts_line = result.stdout.strip()
            # Format: ActiveEnterTimestamp=Thu 2026-02-06 20:00:00 UTC
            if "=" in ts_line:
                ts_str = ts_line.split("=", 1)[1].strip()
                if ts_str:
                    # Parse systemd timestamp
                    try:
                        start = datetime.strptime(ts_str, "%a %Y-%m-%d %H:%M:%S %Z")
                        start = start.replace(tzinfo=timezone.utc)
                        delta = datetime.now(timezone.utc) - start
                        days = delta.days
                        hours, rem = divmod(delta.seconds, 3600)
                        minutes = rem // 60
                        uptime = f"{days}d {hours}h {minutes}m"
                    except ValueError:
                        pass
        except Exception:
            pass

    return {"status": active, "uptime": uptime}


def _compute_stats(trades: list[dict[str, Any]], days: int) -> dict[str, Any]:
    """Compute P&L stats from trade history for the last N days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for t in trades:
        ts_str = t.get("logged_at") or t.get("timestamp", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts >= cutoff:
                    filtered.append(t)
            except (ValueError, TypeError):
                filtered.append(t)
        else:
            filtered.append(t)

    orders = [t for t in filtered if t.get("event") == "order_placed"]
    exits = [t for t in filtered if t.get("event") in ("position_closed", "position_exit", "market_resolved")]

    total_cost = sum(float(t.get("cost_usd", 0)) for t in orders)
    total_pnl = sum(float(t.get("pnl_usd", t.get("pnl", 0))) for t in exits)

    wins = sum(1 for t in exits if float(t.get("pnl_usd", t.get("pnl", 0))) > 0)
    losses = sum(1 for t in exits if float(t.get("pnl_usd", t.get("pnl", 0))) <= 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    edges = [float(t.get("edge", 0)) for t in orders if t.get("edge")]
    avg_edge = sum(edges) / len(edges) if edges else 0.0

    by_strategy: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "pnl": 0.0, "cost": 0.0}
    )
    for t in orders:
        strat = t.get("strategy", t.get("account", "unknown"))
        by_strategy[strat]["count"] += 1
        by_strategy[strat]["cost"] += float(t.get("cost_usd", 0))
    for t in exits:
        strat = t.get("strategy", "unknown")
        by_strategy[strat]["pnl"] += float(t.get("pnl_usd", t.get("pnl", 0)))

    return {
        "period_days": days,
        "total_trades": len(orders),
        "total_exits": len(exits),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 3),
        "wins": wins,
        "losses": losses,
        "avg_edge": round(avg_edge, 4),
        "total_cost": round(total_cost, 2),
        "by_strategy": dict(by_strategy),
    }


# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------


@app.get("/api/summary")
def api_summary() -> dict[str, Any]:
    """P&L overview, balances, bot status."""
    bot = _bot_status()

    # Read perf summary if available, otherwise compute from trades
    perf = _read_json(STATE_DIR / "perf_summary.json")
    all_trades = _read_jsonl(STATE_DIR / "trade_history.jsonl")

    if not perf or "stats_1d" not in perf:
        perf = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "stats_1d": _compute_stats(all_trades, 1),
            "stats_7d": _compute_stats(all_trades, 7),
            "stats_30d": _compute_stats(all_trades, 30),
        }

    # Capital state
    capital = _read_json(STATE_DIR / "capital_state.json")

    return {
        "bot": bot,
        "performance": perf,
        "capital_state": capital,
        "total_trades_all_time": len(
            [t for t in all_trades if t.get("event") == "order_placed"]
        ),
    }


@app.get("/api/positions")
def api_positions() -> dict[str, Any]:
    """Open positions from state file."""
    positions = _read_json(STATE_DIR / "open_positions.json")

    # Convert dict-of-dicts to list, add computed fields
    pos_list = []
    now = datetime.now(timezone.utc)
    for _key, pos in positions.items():
        entry_time = pos.get("entry_time", "")
        age = ""
        if entry_time:
            try:
                et = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                delta = now - et
                hours = delta.total_seconds() / 3600
                if hours < 1:
                    age = f"{int(delta.total_seconds() / 60)}m"
                elif hours < 24:
                    age = f"{hours:.1f}h"
                else:
                    age = f"{delta.days}d {int(hours % 24)}h"
            except (ValueError, TypeError):
                pass
        pos["age"] = age
        pos_list.append(pos)

    return {"positions": pos_list, "count": len(pos_list)}


@app.get("/api/trades")
def api_trades(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    """Recent trade history."""
    trades = _read_jsonl(STATE_DIR / "trade_history.jsonl", limit=limit)
    # Reverse so newest first
    trades.reverse()
    return {"trades": trades, "count": len(trades)}


@app.get("/api/models")
def api_models() -> dict[str, Any]:
    """Model performance + ensemble weights."""
    predictions = _read_jsonl(STATE_DIR / "model_predictions.jsonl")
    resolutions = _read_jsonl(STATE_DIR / "resolutions.jsonl")

    # Build resolution lookup
    resolved: dict[str, float] = {}
    for r in resolutions:
        mid = r.get("market_id")
        outcome = r.get("actual_outcome")
        if mid is not None and outcome is not None:
            resolved[mid] = float(outcome)

    # Per-model stats
    model_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"predictions": 0, "p_yes_sum": 0.0, "brier_scores": []}
    )

    for pred in predictions:
        mid = pred.get("market_id")
        models = pred.get("models", {})
        for model_name, p_yes in models.items():
            stats = model_stats[model_name]
            stats["predictions"] += 1
            stats["p_yes_sum"] += float(p_yes)
            if mid in resolved:
                brier = (float(p_yes) - resolved[mid]) ** 2
                stats["brier_scores"].append(brier)

    # Compute summary per model
    model_summary = {}
    for model, stats in model_stats.items():
        n = stats["predictions"]
        avg_p_yes = stats["p_yes_sum"] / n if n > 0 else 0.0
        brier_list = stats["brier_scores"]
        avg_brier = sum(brier_list) / len(brier_list) if brier_list else None
        model_summary[model] = {
            "predictions": n,
            "avg_p_yes": round(avg_p_yes, 4),
            "resolved_count": len(brier_list),
            "avg_brier_score": round(avg_brier, 4) if avg_brier is not None else None,
        }

    # Agreement rate: how often models agree on direction
    agreement_count = 0
    total_pairs = 0
    for pred in predictions:
        models = pred.get("models", {})
        vals = list(models.values())
        if len(vals) >= 2:
            total_pairs += 1
            # All models agree on which side of 0.5
            sides = [v >= 0.5 for v in vals]
            if len(set(sides)) == 1:
                agreement_count += 1

    agreement_rate = (
        round(agreement_count / total_pairs, 3) if total_pairs > 0 else None
    )

    # Compute weights (same logic as model_tracker.py)
    weights: dict[str, float] = {}
    if model_summary:
        all_have_brier = all(
            (s.get("resolved_count", 0) or 0) >= 30 for s in model_summary.values()
        )
        if all_have_brier:
            inverse = {}
            for m, s in model_summary.items():
                b = s.get("avg_brier_score")
                if b is not None:
                    inverse[m] = 1.0 / (b + 0.001)
            total = sum(inverse.values())
            if total > 0:
                weights = {m: round(v / total, 4) for m, v in inverse.items()}
        else:
            n = len(model_summary)
            weights = {m: round(1.0 / n, 4) for m in model_summary}

    return {
        "models": model_summary,
        "agreement_rate": agreement_rate,
        "ensemble_weights": weights,
        "total_predictions": len(predictions),
        "total_resolutions": len(resolutions),
    }


@app.get("/api/costs")
def api_costs() -> dict[str, Any]:
    """LLM cost tracking."""
    costs = _read_json(STATE_DIR / "cost_tracker.json")
    if not costs:
        return {
            "daily_spend": 0.0,
            "monthly_spend": 0.0,
            "daily_budget": 10.0,
            "monthly_budget": 50.0,
            "call_count": 0,
            "spend_by_model": {},
        }

    return {
        "day": costs.get("day", ""),
        "month": costs.get("month", ""),
        "daily_spend": costs.get("daily_spend", 0.0),
        "monthly_spend": costs.get("total_spend", 0.0),
        "daily_budget": costs.get("daily_budget", 10.0),
        "monthly_budget": costs.get("monthly_budget", 50.0),
        "call_count": costs.get("call_count", 0),
        "spend_by_model": costs.get("spend_by_model", {}),
    }


@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> str:
    """Serve the HTML dashboard."""
    try:
        return DASHBOARD_HTML.read_text()
    except FileNotFoundError:
        return "<h1>Dashboard HTML not found</h1>"
