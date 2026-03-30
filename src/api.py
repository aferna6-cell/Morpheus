"""Morpheus REST API — monitoring and control endpoints.

Provides read-only status/monitoring endpoints plus admin controls.
Authentication via X-API-Key header (set MORPHEUS_API_KEY in .env).

Start with: python -m src.main --api
Or embed in main process: await api.serve()
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse

log = structlog.get_logger()

_server_start: float = 0.0
_bot_ref: Any = None  # Set by main.py when bot is running

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key() -> str | None:
    return os.environ.get("MORPHEUS_API_KEY")


def require_auth(api_key: str | None = Security(API_KEY_HEADER)):
    """Dependency: validate X-API-Key header."""
    expected = get_api_key()
    if not expected:
        return  # No key configured → open access (dev mode)
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def create_app(db=None, config=None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        db: Optional Database instance for persistent queries.
        config: Optional BotConfig for runtime configuration.
    """
    global _server_start
    _server_start = time.monotonic()

    app = FastAPI(
        title="Morpheus Trading Bot API",
        version="2.0.0",
        description="Kalshi prediction market trading bot — monitoring and control",
        docs_url="/docs",
        redoc_url=None,
    )

    # Security headers
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    # CORS (localhost only by default)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000", "http://localhost:3000", "http://127.0.0.1:8000"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    # -----------------------------------------------------------------------
    # Public endpoints (no auth)
    # -----------------------------------------------------------------------

    @app.get("/api/health")
    async def health():
        """Bot health check — no auth required."""
        uptime_s = time.monotonic() - _server_start
        db_ok = False
        db_size_kb = 0
        db_path = Path("state/morpheus.db")
        if db_path.exists():
            db_ok = True
            db_size_kb = db_path.stat().st_size // 1024

        stop_file = Path("state/STOP_TRADING")
        return {
            "status": "ok",
            "uptime_seconds": round(uptime_s),
            "db_ok": db_ok,
            "db_size_kb": db_size_kb,
            "trading_halted": stop_file.exists(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # -----------------------------------------------------------------------
    # Authenticated endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/status", dependencies=[Security(require_auth)])
    async def status():
        """Full bot status — trading state, exposure, daily P&L."""
        stop_file = Path("state/STOP_TRADING")
        result: dict[str, Any] = {
            "trading_active": not stop_file.exists(),
            "uptime_seconds": round(time.monotonic() - _server_start),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if db:
            try:
                positions = await db.get_positions()
                total_exposure = sum(
                    p.get("quantity", 0) * p.get("avg_price_cents", 0) / 100
                    for p in positions
                )
                daily = await db.get_daily_pnl(days=1)
                today_pnl = daily[0]["realized_pnl_cents"] / 100 if daily else 0.0

                result["open_positions"] = len(positions)
                result["total_exposure_usd"] = round(total_exposure, 2)
                result["today_pnl_usd"] = round(today_pnl, 2)
            except Exception as e:
                log.warning("api_status_db_error", error=str(e))
                result["db_error"] = str(e)

        return result

    @app.get("/api/positions", dependencies=[Security(require_auth)])
    async def positions():
        """Open positions with unrealized P&L."""
        if not db:
            return {"positions": [], "note": "Database not configured"}

        try:
            rows = await db.get_positions()
            return {"positions": rows, "count": len(rows)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/trades", dependencies=[Security(require_auth)])
    async def trades(days: int = 7, strategy: Optional[str] = None, limit: int = 100):
        """Recent trades. Filter by strategy or lookback days."""
        if not db:
            return {"trades": [], "note": "Database not configured"}

        try:
            async with db._conn() as conn:
                if strategy:
                    rows = await conn.execute_fetchall(
                        """
                        SELECT * FROM trades
                        WHERE timestamp >= datetime('now', ?)
                          AND strategy = ?
                        ORDER BY timestamp DESC LIMIT ?
                        """,
                        (f"-{days} days", strategy, limit),
                    )
                else:
                    rows = await conn.execute_fetchall(
                        """
                        SELECT * FROM trades
                        WHERE timestamp >= datetime('now', ?)
                        ORDER BY timestamp DESC LIMIT ?
                        """,
                        (f"-{days} days", limit),
                    )
            return {"trades": [dict(r) for r in rows], "count": len(rows)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/signals", dependencies=[Security(require_auth)])
    async def signals(hours: int = 1, approved_only: bool = False, limit: int = 200):
        """Recent strategy signals (approved and rejected)."""
        if not db:
            return {"signals": [], "note": "Database not configured"}

        try:
            async with db._conn() as conn:
                query = """
                    SELECT * FROM signals
                    WHERE timestamp >= datetime('now', ?)
                """
                params: list = [f"-{hours} hours"]
                if approved_only:
                    query += " AND approved = 1"
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                rows = await conn.execute_fetchall(query, params)
            return {"signals": [dict(r) for r in rows], "count": len(rows)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/pnl", dependencies=[Security(require_auth)])
    async def pnl(days: int = 30):
        """Daily P&L history."""
        if not db:
            return {"pnl": [], "note": "Database not configured"}

        try:
            rows = await db.get_daily_pnl(days=days)
            return {"pnl": rows, "days": days}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/strategies", dependencies=[Security(require_auth)])
    async def strategy_stats(days: int = 30):
        """Per-strategy performance statistics."""
        if not db:
            return {"strategies": {}, "note": "Database not configured"}

        try:
            perf = await db.get_strategy_perf(days=days)
            # Compute win rate per strategy
            result = {}
            for strategy, stats in perf.items():
                wins = stats.get("wins", 0)
                losses = stats.get("losses", 0)
                total = wins + losses
                result[strategy] = {
                    **stats,
                    "win_rate": round(wins / total, 3) if total > 0 else 0.0,
                    "total_trades": total,
                    "pnl_usd": round(stats.get("pnl_cents", 0) / 100, 2),
                }
            return {"strategies": result, "days": days}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/risk", dependencies=[Security(require_auth)])
    async def risk():
        """Current risk state — exposure, daily loss, circuit breaker status."""
        stop_file = Path("state/STOP_TRADING")
        result: dict[str, Any] = {
            "trading_halted": stop_file.exists(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if db:
            try:
                positions = await db.get_positions()
                total_exposure = sum(
                    p.get("quantity", 0) * p.get("avg_price_cents", 0) / 100
                    for p in positions
                )

                daily = await db.get_daily_pnl(days=1)
                today_pnl_cents = daily[0]["realized_pnl_cents"] if daily else 0

                result.update({
                    "open_positions": len(positions),
                    "total_exposure_usd": round(total_exposure, 2),
                    "today_pnl_usd": round(today_pnl_cents / 100, 2),
                })

                if config:
                    max_daily_loss = config.risk.get("max_daily_loss", 20.0)
                    max_total_exposure = config.strategy.get("max_total_exposure", 200.0)
                    result["max_daily_loss"] = max_daily_loss
                    result["max_total_exposure"] = max_total_exposure
                    result["daily_loss_used_pct"] = round(
                        abs(min(today_pnl_cents / 100, 0)) / max_daily_loss * 100, 1
                    ) if max_daily_loss > 0 else 0
                    result["exposure_used_pct"] = round(
                        total_exposure / max_total_exposure * 100, 1
                    ) if max_total_exposure > 0 else 0

            except Exception as e:
                log.warning("api_risk_db_error", error=str(e))
                result["db_error"] = str(e)

        return result

    @app.get("/api/clv", dependencies=[Security(require_auth)])
    async def clv(days: int = 30):
        """Closing Line Value by market type (edge quality metric)."""
        if not db:
            return {"clv": {}, "note": "Database not configured"}

        try:
            async with db._conn() as conn:
                rows = await conn.execute_fetchall(
                    """
                    SELECT market_type,
                           AVG(clv) as avg_clv,
                           COUNT(*) as n_samples,
                           MIN(clv) as min_clv,
                           MAX(clv) as max_clv
                    FROM clv_records
                    WHERE timestamp >= datetime('now', ?)
                    GROUP BY market_type
                    """,
                    (f"-{days} days",),
                )
            by_type = {r["market_type"]: dict(r) for r in rows}
            return {"clv_by_type": by_type, "days": days}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # -----------------------------------------------------------------------
    # Admin endpoints (require auth)
    # -----------------------------------------------------------------------

    @app.post("/api/admin/halt", dependencies=[Security(require_auth)])
    async def halt_trading():
        """Activate kill switch — halt all new trading."""
        stop_file = Path("state/STOP_TRADING")
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.touch()
        log.warning("kill_switch_activated", source="api")
        return {"status": "halted", "message": "Kill switch activated. Delete state/STOP_TRADING to resume."}

    @app.post("/api/admin/resume", dependencies=[Security(require_auth)])
    async def resume_trading():
        """Deactivate kill switch — resume trading."""
        stop_file = Path("state/STOP_TRADING")
        if stop_file.exists():
            stop_file.unlink()
            log.info("kill_switch_cleared", source="api")
            return {"status": "resumed", "message": "Kill switch cleared. Trading will resume on next cycle."}
        return {"status": "already_running", "message": "Bot was not halted."}

    @app.get("/api/config", dependencies=[Security(require_auth)])
    async def get_config():
        """Current configuration (secrets excluded)."""
        if not config:
            return {"config": None, "note": "Config not available"}

        # Return safe subset (no API keys)
        safe = {
            "strategy": config.strategy,
            "market_filters": config.market_filters,
            "risk": config.risk,
            "calibration": config.calibration,
            "bonding": config.bonding,
            "bracket_arb": config.bracket_arb,
            "survival": config.survival,
        }
        return {"config": safe}

    return app


async def serve(db=None, config=None, host: str = "0.0.0.0", port: int = 8000):
    """Start the API server (blocking)."""
    import uvicorn
    app = create_app(db=db, config=config)
    server_config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(server_config)
    log.info("api_server_starting", host=host, port=port)
    await server.serve()
