"""Copy-trading engine — monitor top Polymarket wallets and mirror their trades.

Polls data-api.polymarket.com for recent activity of each master wallet,
deduplicates trades we've already seen, and emits TradeSignal objects for
the orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
import structlog

from ..utils import BotConfig, append_jsonl, load_json_state, save_json_state
from .base import BaseEngine
from .copy_watchlist import Watchlist
from .signals import TradeSignal

logger = structlog.get_logger()

DATA_API_BASE = "https://data-api.polymarket.com"
ACTIVITY_URL = f"{DATA_API_BASE}/activity"

# Only copy trades from the last N minutes on each poll.
# Prevents the "202 signals on first run" flood from historical trades.
DEFAULT_LOOKBACK_MINUTES = 30


class CopyTradingEngine(BaseEngine):
    """Watches master wallets on Polymarket and generates copy-trade signals."""

    name: str = "copy_trading"

    def __init__(self, config: BotConfig):
        self.config = config
        self._ct: Dict[str, Any] = getattr(config, "copy_trading", None) or {}

        self.poll_interval: float = float(self._ct.get("poll_interval_seconds", 30))
        self.min_trade_size: float = float(self._ct.get("min_trade_size_usd", 50))
        self.max_copy_size: float = float(self._ct.get("max_copy_size_usd", 25))
        self.copy_ratio: float = float(self._ct.get("copy_ratio", 0.1))
        self.max_masters: int = int(self._ct.get("max_masters", 10))
        self.lookback_minutes: float = float(
            self._ct.get("lookback_minutes", DEFAULT_LOOKBACK_MINUTES)
        )

        watchlist_file = self._ct.get("watchlist_file", "state/copy_watchlist.json")
        self.watchlist_path = Path(watchlist_file)
        history_file = self._ct.get("trade_history_file", "state/copy_trades.jsonl")
        self.history_path = Path(history_file)

        # Runtime state
        self._watchlist: Optional[Watchlist] = None
        self._seen_trade_ids: Set[str] = set()
        self._seen_market_wallet: Set[str] = set()  # "wallet|market_id" dedup
        self._pending_signals: List[TradeSignal] = []
        self._http: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False

        # Simple rate-limit tracking
        self._last_request_ts: float = 0.0
        self._min_request_gap: float = 1.0  # seconds between API requests

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._watchlist = Watchlist.load(self.watchlist_path)

        # Merge any addresses from config.yaml master_wallets list
        for addr in self._ct.get("master_wallets", []):
            self._watchlist.add(addr, alias="config")

        # For wallets with no last_trade_seen, set it to now minus lookback
        # so the first poll only picks up truly recent trades.
        now_iso = datetime.now(timezone.utc).isoformat()
        for ws in self._watchlist.wallets.values():
            if not ws.last_trade_seen:
                ws.last_trade_seen = now_iso
                logger.info("copy_wallet_init_cutoff", wallet=ws.address[:10], cutoff=now_iso)

        self._watchlist.save(self.watchlist_path)

        self._http = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0))
        self._load_seen_trades()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

        logger.info(
            "copy_engine_started",
            wallets=len(self._watchlist.wallets),
            poll_interval=self.poll_interval,
            lookback_minutes=self.lookback_minutes,
        )

    async def stop(self) -> None:
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._http:
            await self._http.aclose()
        if self._watchlist:
            self._watchlist.save(self.watchlist_path)
        logger.info("copy_engine_stopped")

    async def get_signals(self) -> List[TradeSignal]:
        """Drain and return pending signals (called by orchestrator)."""
        signals = list(self._pending_signals)
        self._pending_signals.clear()
        return signals

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_all_wallets()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("copy_poll_error", error=str(exc))
            await asyncio.sleep(self.poll_interval)

    async def _poll_all_wallets(self) -> None:
        if not self._watchlist:
            return

        addresses = self._watchlist.addresses()[: self.max_masters]
        for addr in addresses:
            try:
                trades = await self._fetch_wallet_activity(addr)
                cutoff_iso = (self._watchlist.get(addr).last_trade_seen if self._watchlist.get(addr) else "")
                new_trades = self._filter_new_trades(trades, cutoff_iso=cutoff_iso, wallet_address=addr)
                for trade in new_trades:
                    self._process_trade(addr, trade)
            except Exception as exc:
                logger.warning("copy_wallet_error", address=addr, error=str(exc))

    # ------------------------------------------------------------------
    # Gamma API helpers
    # ------------------------------------------------------------------

    async def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self._min_request_gap:
            await asyncio.sleep(self._min_request_gap - elapsed)
        self._last_request_ts = time.monotonic()

    async def _fetch_wallet_activity(self, address: str) -> List[Dict[str, Any]]:
        """Fetch recent trades for a wallet via data-api activity endpoint."""
        await self._rate_limit()
        assert self._http is not None

        params = {"user": address}

        resp = await self._http.get(ACTIVITY_URL, params=params)
        # data-api returns 4xx when params are wrong; bubble up for now
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("data", data.get("activities", []))

    async def fetch_wallet_portfolio(self, address: str) -> Dict[str, Any]:
        """Fetch portfolio summary for a wallet (useful for sizing context)."""
        await self._rate_limit()
        assert self._http is not None

        url = f"{DATA_API_BASE}/portfolios"
        params = {"user": address}

        resp = await self._http.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Trade processing
    # ------------------------------------------------------------------

    def _trade_id(self, trade: Dict[str, Any]) -> str:
        """Derive a stable dedup key from a Gamma activity item."""
        # Prefer explicit id, fall back to hash of key fields
        tid = trade.get("id") or trade.get("transactionHash") or trade.get("hash")
        if tid:
            return str(tid)
        # Deterministic fallback
        parts = [
            str(trade.get("conditionId", "")),
            str(trade.get("outcomeIndex", "")),
            str(trade.get("timestamp", "")),
            str(trade.get("amount", "")),
        ]
        return "|".join(parts)

    def _filter_new_trades(
        self,
        trades: List[Dict[str, Any]],
        *,
        cutoff_iso: str = "",
        wallet_address: str = "",
    ) -> List[Dict[str, Any]]:
        """Return only trades we haven't processed yet.

        Filters applied (in order):
        1. Per-wallet cutoff timestamp — skip anything at/before last_trade_seen
        2. Lookback window — skip trades older than lookback_minutes
        3. Dedup by trade ID
        4. Dedup by wallet+market (only latest trade per market per wallet)
        """
        cutoff_ts: Optional[int] = None
        if cutoff_iso:
            try:
                cutoff_dt = datetime.fromisoformat(cutoff_iso.replace("Z", "+00:00"))
                cutoff_ts = int(cutoff_dt.timestamp())
            except Exception:
                cutoff_ts = None

        # Hard lookback window: ignore anything older than N minutes
        lookback_cutoff = int(
            (datetime.now(timezone.utc) - timedelta(minutes=self.lookback_minutes)).timestamp()
        )

        new: List[Dict[str, Any]] = []
        for t in trades:
            # Parse trade timestamp
            try:
                tts = int(t.get("timestamp") or 0)
            except Exception:
                tts = 0

            # Skip anything before per-wallet cutoff
            if cutoff_ts is not None and tts and tts <= cutoff_ts:
                continue

            # Skip anything outside lookback window
            if tts and tts < lookback_cutoff:
                continue

            tid = self._trade_id(t)
            if tid in self._seen_trade_ids:
                continue

            # Per-wallet per-market dedup: only copy latest signal per market per wallet
            market_id = str(t.get("conditionId") or t.get("marketSlug") or "")
            if wallet_address and market_id:
                dedup_key = f"{wallet_address}|{market_id}"
                if dedup_key in self._seen_market_wallet:
                    continue
                # Mark as seen immediately so later trades in this batch are skipped
                self._seen_market_wallet.add(dedup_key)

            new.append(t)
        return new

    def _process_trade(self, wallet_address: str, trade: Dict[str, Any]) -> None:
        """Convert a Gamma activity item into a TradeSignal and queue it."""
        tid = self._trade_id(trade)
        self._seen_trade_ids.add(tid)

        # Track wallet+market for dedup
        market_id_raw = str(trade.get("conditionId") or trade.get("marketSlug") or "")
        if market_id_raw:
            self._seen_market_wallet.add(f"{wallet_address}|{market_id_raw}")

        # Parse trade details
        trade_size_usd = _safe_float(trade.get("usdcSize") or trade.get("amount") or 0)
        if trade_size_usd < self.min_trade_size:
            logger.debug("copy_trade_too_small", wallet=wallet_address, size=trade_size_usd)
            return

        # Determine market and side
        market_id = str(trade.get("conditionId") or trade.get("marketSlug") or "")
        token_id = str(trade.get("asset") or trade.get("assetId") or trade.get("tokenId") or "")
        outcome_index = trade.get("outcomeIndex")
        trade_type = str(trade.get("type") or trade.get("side") or "").lower()

        # Validate token_id: CLOB token IDs are large decimal integers.
        # data-api sometimes returns condition_id (0x hex) or empty strings.
        # These will fail dispatch, so skip them.
        if not token_id or token_id.startswith("0x") or len(token_id) < 10:
            logger.debug(
                "copy_trade_invalid_token_id",
                wallet=wallet_address,
                token_id=token_id[:20] if token_id else "",
                market_id=market_id[:20],
            )
            return

        # Map to our side convention
        side = self._infer_side(trade_type, outcome_index)
        if not side:
            logger.debug("copy_trade_unknown_side", trade=trade)
            return

        price = _safe_float(trade.get("price") or 0)
        question = str(trade.get("title") or trade.get("question") or "")

        # Calculate copy size
        copy_size = min(trade_size_usd * self.copy_ratio, self.max_copy_size)

        # Build wallet stats context
        ws = self._watchlist.get(wallet_address) if self._watchlist else None
        wallet_meta = {
            "master_wallet": wallet_address,
            "master_alias": ws.alias if ws else "",
            "master_win_rate": ws.win_rate if ws else 0.0,
            "master_trade_size_usd": trade_size_usd,
            "copy_size_usd": copy_size,
            "original_price": price,
            "gamma_trade_id": tid,
        }

        wallet_meta["question"] = question

        signal = TradeSignal(
            engine=self.name,
            market_id=market_id,
            token_id=token_id,
            side=side,
            confidence=min(0.5 + (ws.win_rate * 0.4 if ws else 0.0), 0.9),
            edge=0.0,  # Copy trading doesn't estimate edge directly
            urgency="high",  # Copy trades should execute quickly
            metadata=wallet_meta,
        )

        self._pending_signals.append(signal)
        self._log_trade(wallet_address, trade, signal)

        # Update watchlist stats
        if self._watchlist:
            # Persist latest seen trade timestamp to prevent replaying history
            ws2 = self._watchlist.get(wallet_address)
            try:
                tts = int(trade.get("timestamp") or 0)
            except Exception:
                tts = 0
            if ws2 and tts:
                trade_iso = datetime.fromtimestamp(tts, tz=timezone.utc).isoformat()
                # Keep max(last_trade_seen, trade_iso)
                if not ws2.last_trade_seen or trade_iso > ws2.last_trade_seen:
                    ws2.last_trade_seen = trade_iso
            self._watchlist.record_copy(wallet_address)
            self._watchlist.save(self.watchlist_path)

        logger.info(
            "copy_signal_generated",
            wallet=wallet_address[:10] + "...",
            market=market_id[:20],
            side=side,
            master_size=trade_size_usd,
            copy_size=copy_size,
        )

    def _infer_side(self, trade_type: str, outcome_index: Any) -> Optional[str]:
        """Map Gamma activity fields to buy_yes / buy_no."""
        # Gamma uses outcomeIndex: 0 = Yes, 1 = No (usually)
        if outcome_index is not None:
            try:
                idx = int(outcome_index)
                if "sell" in trade_type:
                    # Selling Yes ≈ buying No, and vice versa
                    return "buy_no" if idx == 0 else "buy_yes"
                return "buy_yes" if idx == 0 else "buy_no"
            except (ValueError, TypeError):
                pass

        if "buy" in trade_type and "yes" in trade_type:
            return "buy_yes"
        if "buy" in trade_type and "no" in trade_type:
            return "buy_no"
        if "buy" in trade_type:
            return "buy_yes"
        if "sell" in trade_type:
            return "buy_no"

        return None

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _log_trade(self, wallet: str, raw_trade: Dict, signal: TradeSignal) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wallet": wallet,
            "market_id": signal.market_id,
            "side": signal.side,
            "master_size_usd": signal.metadata.get("master_trade_size_usd"),
            "copy_size_usd": signal.metadata.get("copy_size_usd"),
            "price": signal.metadata.get("original_price"),
            "question": signal.metadata.get("question"),
            "gamma_trade_id": signal.metadata.get("gamma_trade_id"),
        }
        append_jsonl(self.history_path, record)

    def _load_seen_trades(self) -> None:
        """Load previously seen trade IDs from the history JSONL to avoid re-signaling."""
        if not self.history_path.exists():
            return
        try:
            with open(self.history_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        tid = rec.get("gamma_trade_id")
                        if tid:
                            self._seen_trade_ids.add(str(tid))
                        # Also load wallet+market dedup keys
                        wallet = rec.get("wallet", "")
                        mid = rec.get("market_id", "")
                        if wallet and mid:
                            self._seen_market_wallet.add(f"{wallet}|{mid}")
                    except json.JSONDecodeError:
                        continue
            logger.info(
                "copy_seen_trades_loaded",
                trade_ids=len(self._seen_trade_ids),
                market_wallet_pairs=len(self._seen_market_wallet),
            )
        except Exception as exc:
            logger.warning("copy_seen_trades_load_error", error=str(exc))

    # Note: leaderboard discovery disabled for now; Gamma /leaderboard has been 404.
    # We rely on a curated wallet list in config + watchlist.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default
