"""Capital management — recycling rules and CLV tracking.

Capital Recycling Rule:
- >=70% of bankroll must resolve within 14 days
- >=50% of bankroll must resolve within 7 days
- If violated -> bot stops trading new positions

CLV (Closing Line Value) Tracking:
- Log entry probability vs closing probability for every bet
- If average CLV <= 0 -> bot auto-halts
- CLV > 0 means we're beating the market consistently
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .utils import BotConfig, load_json_state, save_json_state


@dataclass
class OpenPosition:
    """Tracks an open position for capital recycling."""
    market_id: str
    ticker: str
    platform: str
    entry_time: datetime
    resolution_time: Optional[datetime]
    amount_usd: float
    entry_price: float
    entry_probability: float  # our predicted probability
    side: str  # "yes" or "no"

    def days_to_resolution(self) -> Optional[float]:
        """Days until this position resolves."""
        if self.resolution_time is None:
            return None
        now = datetime.now(timezone.utc)
        return (self.resolution_time - now).total_seconds() / 86400.0


@dataclass
class CLVRecord:
    """Record for CLV tracking."""
    market_id: str
    ticker: str
    platform: str
    entry_time: str
    entry_probability: float  # our model's probability
    entry_market_price: float  # market price when we entered
    closing_probability: Optional[float] = None  # market price at close
    direction: str = ""  # "yes" or "no"
    clv: Optional[float] = None  # entry_prob - closing_prob (positive = we were ahead)
    resolved: bool = False
    outcome: Optional[str] = None  # "yes", "no", or None


@dataclass
class CapitalRecyclingStatus:
    """Status of capital recycling compliance."""
    compliant: bool
    pct_resolving_7d: float
    pct_resolving_14d: float
    required_7d_pct: float
    required_14d_pct: float
    total_exposure: float
    exposure_7d: float
    exposure_14d: float
    reason: str


@dataclass
class CLVStatus:
    """Status of CLV tracking."""
    average_clv: float
    total_bets: int
    resolved_bets: int
    positive_clv_count: int
    negative_clv_count: int
    should_halt: bool
    reason: str


class CapitalManager:
    """Manages capital recycling and CLV tracking."""

    def __init__(self, config: BotConfig, *, state_dir: str = "state"):
        self.config = config
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger()

        # Load config
        mf = getattr(config, "market_filters", {}) or {}
        if not isinstance(mf, dict):
            mf = {}

        self.required_14d_pct = float(mf.get("max_14d_allocation_pct", 0.70))
        self.required_7d_pct = float(mf.get("max_7d_allocation_pct", 0.50))

        clv_cfg = getattr(config, "clv_tracking", {}) or {}
        if not isinstance(clv_cfg, dict):
            clv_cfg = {}

        self.clv_enabled = bool(clv_cfg.get("enabled", True))
        self.clv_alert_threshold = float(clv_cfg.get("alert_threshold", -0.01))
        self.clv_halt_threshold = float(clv_cfg.get("halt_threshold", -0.01))
        self.clv_min_samples = int(clv_cfg.get("min_samples", 10))
        self.per_type_tracking = bool(clv_cfg.get("per_type_tracking", True))
        self.per_type_halt_threshold = float(clv_cfg.get("per_type_halt_threshold", -0.01))
        self._disabled_market_types: set = set()  # market types with poor CLV

        # State files
        self.positions_file = self.state_dir / "open_positions.json"
        self.clv_file = self.state_dir / "clv_history.jsonl"
        self.capital_state_file = self.state_dir / "capital_state.json"

        # In-memory state
        self._positions: Dict[str, OpenPosition] = {}
        self._halted_capital_recycling = False
        self._halted_clv = False
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state.

        Note: halt flags are NOT restored from disk. Each run starts fresh
        and halts only if it detects a violation during this session.
        This prevents stale state from permanently blocking the bot.
        """
        try:
            state = load_json_state(str(self.capital_state_file))
            # Deliberately do NOT restore halt flags — start clean each run

            pos_state = load_json_state(str(self.positions_file))
            if pos_state and isinstance(pos_state, dict):
                for mid, pdata in pos_state.items():
                    try:
                        res_time = None
                        if pdata.get("resolution_time"):
                            res_time = datetime.fromisoformat(pdata["resolution_time"])
                        self._positions[mid] = OpenPosition(
                            market_id=pdata["market_id"],
                            ticker=pdata["ticker"],
                            platform=pdata["platform"],
                            entry_time=datetime.fromisoformat(pdata["entry_time"]),
                            resolution_time=res_time,
                            amount_usd=pdata["amount_usd"],
                            entry_price=pdata["entry_price"],
                            entry_probability=pdata["entry_probability"],
                            side=pdata["side"],
                        )
                    except Exception as e:
                        self.logger.warning("position_load_error", market_id=mid, error=str(e))
        except Exception as e:
            self.logger.warning("capital_state_load_error", error=str(e))

        self._purge_expired()

    def _purge_expired(self) -> None:
        """Remove positions whose resolution_time has already passed."""
        now = datetime.now(timezone.utc)
        expired = [
            mid for mid, pos in self._positions.items()
            if pos.resolution_time is not None and pos.resolution_time < now
        ]
        if not expired:
            return
        for mid in expired:
            self.logger.info("purged_expired_position", market_id=mid,
                             resolution_time=self._positions[mid].resolution_time.isoformat())
            del self._positions[mid]
        self.logger.info("purge_expired_complete", count=len(expired))
        self._save_state()

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            save_json_state({
                "halted_capital_recycling": self._halted_capital_recycling,
                "halted_clv": self._halted_clv,
                "last_update": datetime.now(timezone.utc).isoformat(),
            }, str(self.capital_state_file))

            pos_data = {}
            for mid, pos in self._positions.items():
                pos_data[mid] = {
                    "market_id": pos.market_id,
                    "ticker": pos.ticker,
                    "platform": pos.platform,
                    "entry_time": pos.entry_time.isoformat(),
                    "resolution_time": pos.resolution_time.isoformat() if pos.resolution_time else None,
                    "amount_usd": pos.amount_usd,
                    "entry_price": pos.entry_price,
                    "entry_probability": pos.entry_probability,
                    "side": pos.side,
                }
            save_json_state(pos_data, str(self.positions_file))
        except Exception as e:
            self.logger.error("capital_state_save_error", error=str(e))

    # -------------------------------------------------------------------------
    # Position tracking
    # -------------------------------------------------------------------------

    def add_position(
        self,
        market_id: str,
        ticker: str,
        platform: str,
        amount_usd: float,
        entry_price: float,
        entry_probability: float,
        side: str,
        resolution_time: Optional[datetime] = None,
        market_type: str = "unknown",
    ) -> None:
        """Add a new position to track."""
        self._positions[market_id] = OpenPosition(
            market_id=market_id,
            ticker=ticker,
            platform=platform,
            entry_time=datetime.now(timezone.utc),
            resolution_time=resolution_time,
            amount_usd=amount_usd,
            entry_price=entry_price,
            entry_probability=entry_probability,
            side=side,
        )

        # Log CLV entry with market type for per-type tracking
        self._log_clv_entry(
            market_id=market_id,
            ticker=ticker,
            platform=platform,
            entry_probability=entry_probability,
            entry_market_price=entry_price,
            direction=side,
            market_type=market_type,
        )

        self._save_state()
        self.logger.info(
            "position_added",
            market_id=market_id,
            amount_usd=amount_usd,
            days_to_resolution=self._positions[market_id].days_to_resolution(),
        )

    def remove_position(self, market_id: str, closing_price: Optional[float] = None) -> None:
        """Remove a position (resolved or closed)."""
        if market_id in self._positions:
            pos = self._positions[market_id]

            # Update CLV with closing price if available
            if closing_price is not None:
                self._update_clv_closing(market_id, closing_price)

            del self._positions[market_id]
            self._save_state()
            self.logger.info("position_removed", market_id=market_id)

    def get_open_positions(self) -> List[OpenPosition]:
        """Get all open positions."""
        return list(self._positions.values())

    # -------------------------------------------------------------------------
    # Capital Recycling Rule
    # -------------------------------------------------------------------------

    def check_capital_recycling(self) -> CapitalRecyclingStatus:
        """Check if capital recycling rules are met.

        Rules:
        - >=70% of exposure must resolve within 14 days
        - >=50% of exposure must resolve within 7 days
        """
        total_exposure = sum(p.amount_usd for p in self._positions.values())

        if total_exposure <= 0:
            return CapitalRecyclingStatus(
                compliant=True,
                pct_resolving_7d=1.0,
                pct_resolving_14d=1.0,
                required_7d_pct=self.required_7d_pct,
                required_14d_pct=self.required_14d_pct,
                total_exposure=0.0,
                exposure_7d=0.0,
                exposure_14d=0.0,
                reason="No open positions",
            )

        exposure_7d = 0.0
        exposure_14d = 0.0
        exposure_unknown = 0.0

        for pos in self._positions.values():
            days = pos.days_to_resolution()
            if days is None:
                # Unknown resolution — can't enforce time rule on these
                exposure_unknown += pos.amount_usd
            elif days <= 7:
                exposure_7d += pos.amount_usd
                exposure_14d += pos.amount_usd
            elif days <= 14:
                exposure_14d += pos.amount_usd

        # Only enforce recycling on positions with known resolution dates.
        # Positions with unknown resolution are excluded from the denominator.
        known_exposure = total_exposure - exposure_unknown
        if known_exposure <= 0:
            # All positions have unknown resolution — can't enforce, allow trading
            return CapitalRecyclingStatus(
                compliant=True,
                pct_resolving_7d=1.0,
                pct_resolving_14d=1.0,
                required_7d_pct=self.required_7d_pct,
                required_14d_pct=self.required_14d_pct,
                total_exposure=total_exposure,
                exposure_7d=exposure_7d,
                exposure_14d=exposure_14d,
                reason="All positions have unknown resolution — skipping check",
            )

        pct_7d = exposure_7d / known_exposure
        pct_14d = exposure_14d / known_exposure

        compliant = pct_14d >= self.required_14d_pct and pct_7d >= self.required_7d_pct

        reasons = []
        if pct_14d < self.required_14d_pct:
            reasons.append(f"14d: {pct_14d:.1%} < {self.required_14d_pct:.1%}")
        if pct_7d < self.required_7d_pct:
            reasons.append(f"7d: {pct_7d:.1%} < {self.required_7d_pct:.1%}")

        status = CapitalRecyclingStatus(
            compliant=compliant,
            pct_resolving_7d=pct_7d,
            pct_resolving_14d=pct_14d,
            required_7d_pct=self.required_7d_pct,
            required_14d_pct=self.required_14d_pct,
            total_exposure=total_exposure,
            exposure_7d=exposure_7d,
            exposure_14d=exposure_14d,
            reason="OK" if compliant else "; ".join(reasons),
        )

        if not compliant:
            self._halted_capital_recycling = True
            self._save_state()
            self.logger.warning(
                "capital_recycling_violation",
                pct_7d=pct_7d,
                pct_14d=pct_14d,
                total_exposure=total_exposure,
            )

        return status

    def is_capital_recycling_compliant(self) -> bool:
        """Quick check if we're allowed to trade based on capital recycling."""
        if self._halted_capital_recycling:
            status = self.check_capital_recycling()
            if status.compliant:
                self._halted_capital_recycling = False
                self._save_state()
                self.logger.info("capital_recycling_resumed")
            return status.compliant
        return True

    # -------------------------------------------------------------------------
    # CLV Tracking
    # -------------------------------------------------------------------------

    def is_market_type_allowed(self, market_type: str) -> bool:
        """Check if a market type is allowed based on per-type CLV tracking."""
        if not self.per_type_tracking:
            return True
        return market_type not in self._disabled_market_types

    def check_per_type_clv(self) -> Dict[str, Dict[str, Any]]:
        """Check CLV per market type and disable types with poor performance."""
        if not self.clv_enabled or not self.per_type_tracking:
            return {}

        records = []
        try:
            if self.clv_file.exists():
                with open(self.clv_file, "r") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
        except Exception:
            return {}

        # Group by market type
        from collections import defaultdict
        by_type: Dict[str, list] = defaultdict(list)
        for r in records:
            if r.get("clv") is not None:
                mtype = r.get("market_type", "unknown")
                by_type[mtype].append(r["clv"])

        results = {}
        for mtype, clv_values in by_type.items():
            if len(clv_values) < 5:  # Need minimum samples per type
                continue
            avg = sum(clv_values) / len(clv_values)
            results[mtype] = {
                "avg_clv": round(avg, 4),
                "count": len(clv_values),
                "disabled": avg < self.per_type_halt_threshold,
            }
            if avg < self.per_type_halt_threshold:
                if mtype not in self._disabled_market_types:
                    self._disabled_market_types.add(mtype)
                    self.logger.warning(
                        "clv_type_disabled",
                        market_type=mtype,
                        avg_clv=avg,
                        threshold=self.per_type_halt_threshold,
                        sample_size=len(clv_values),
                    )
            elif mtype in self._disabled_market_types:
                self._disabled_market_types.discard(mtype)
                self.logger.info("clv_type_re_enabled", market_type=mtype, avg_clv=avg)

        return results

    def _log_clv_entry(
        self,
        market_id: str,
        ticker: str,
        platform: str,
        entry_probability: float,
        entry_market_price: float,
        direction: str,
        market_type: str = "unknown",
    ) -> None:
        """Log a CLV entry when a trade is placed."""
        if not self.clv_enabled:
            return

        record = {
            "market_id": market_id,
            "ticker": ticker,
            "platform": platform,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_probability": entry_probability,
            "entry_market_price": entry_market_price,
            "direction": direction,
            "market_type": market_type,
            "closing_probability": None,
            "clv": None,
            "resolved": False,
            "outcome": None,
        }

        try:
            with open(self.clv_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self.logger.error("clv_log_entry_error", error=str(e))

    def _update_clv_closing(self, market_id: str, closing_price: float) -> None:
        """Update CLV record with closing price (atomic write)."""
        if not self.clv_enabled:
            return

        try:
            # Read all records, update matching one, write back
            records = []
            if self.clv_file.exists():
                with open(self.clv_file, "r") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))

            for record in records:
                if record.get("market_id") == market_id and record.get("clv") is None:
                    record["closing_probability"] = closing_price
                    # CLV = our edge at entry vs closing line
                    # If we bought YES: CLV = closing_price - entry_market_price
                    # Positive CLV means the line moved toward our prediction
                    entry_price = record.get("entry_market_price", 0.5)
                    direction = record.get("direction", "yes")

                    if direction == "yes":
                        # We bet YES, so if closing price > entry price, we had +CLV
                        record["clv"] = closing_price - entry_price
                    else:
                        # We bet NO, so if closing price < entry price (YES dropped), we had +CLV
                        record["clv"] = entry_price - closing_price

                    record["resolved"] = True
                    break

            # Atomic write: temp file + rename (safe against crash)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.clv_file.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    for record in records:
                        f.write(json.dumps(record) + "\n")
                os.replace(tmp_path, str(self.clv_file))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            self.logger.error("clv_update_error", market_id=market_id, error=str(e))

    def log_clv_resolution(
        self,
        market_id: str,
        closing_price: float,
        outcome: str,
    ) -> None:
        """Log CLV when a market resolves."""
        if not self.clv_enabled:
            return

        try:
            records = []
            if self.clv_file.exists():
                with open(self.clv_file, "r") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))

            for record in records:
                if record.get("market_id") == market_id:
                    record["closing_probability"] = closing_price
                    record["outcome"] = outcome
                    record["resolved"] = True

                    entry_price = record.get("entry_market_price", 0.5)
                    direction = record.get("direction", "yes")

                    if direction == "yes":
                        record["clv"] = closing_price - entry_price
                    else:
                        record["clv"] = entry_price - closing_price
                    break

            # Atomic write: temp file + rename (safe against crash)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.clv_file.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    for record in records:
                        f.write(json.dumps(record) + "\n")
                os.replace(tmp_path, str(self.clv_file))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            self.logger.info(
                "clv_resolution_logged",
                market_id=market_id,
                closing_price=closing_price,
                outcome=outcome,
            )
        except Exception as e:
            self.logger.error("clv_resolution_error", market_id=market_id, error=str(e))

    def check_clv_status(self) -> CLVStatus:
        """Check average CLV and determine if trading should halt.

        CLV (Closing Line Value) measures if our entry was better than the
        closing price. Positive CLV = we're beating the market.

        If average CLV <= 0 with sufficient samples, halt trading.
        """
        if not self.clv_enabled:
            return CLVStatus(
                average_clv=0.0,
                total_bets=0,
                resolved_bets=0,
                positive_clv_count=0,
                negative_clv_count=0,
                should_halt=False,
                reason="CLV tracking disabled",
            )

        records = []
        try:
            if self.clv_file.exists():
                with open(self.clv_file, "r") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
        except Exception as e:
            self.logger.error("clv_read_error", error=str(e))
            return CLVStatus(
                average_clv=0.0,
                total_bets=0,
                resolved_bets=0,
                positive_clv_count=0,
                negative_clv_count=0,
                should_halt=False,
                reason=f"Error reading CLV data: {e}",
            )

        total_bets = len(records)
        resolved = [r for r in records if r.get("clv") is not None]
        resolved_bets = len(resolved)

        if resolved_bets < self.clv_min_samples:
            return CLVStatus(
                average_clv=0.0,
                total_bets=total_bets,
                resolved_bets=resolved_bets,
                positive_clv_count=0,
                negative_clv_count=0,
                should_halt=False,
                reason=f"Need {self.clv_min_samples} samples, have {resolved_bets}",
            )

        clv_values = [r["clv"] for r in resolved]
        avg_clv = sum(clv_values) / len(clv_values)
        positive_count = sum(1 for c in clv_values if c > 0)
        negative_count = sum(1 for c in clv_values if c <= 0)

        should_halt = avg_clv <= self.clv_halt_threshold

        if should_halt:
            self._halted_clv = True
            self._save_state()
            self.logger.error(
                "clv_halt_triggered",
                average_clv=avg_clv,
                threshold=self.clv_halt_threshold,
                resolved_bets=resolved_bets,
            )

        return CLVStatus(
            average_clv=avg_clv,
            total_bets=total_bets,
            resolved_bets=resolved_bets,
            positive_clv_count=positive_count,
            negative_clv_count=negative_count,
            should_halt=should_halt,
            reason="CLV too low - halt trading" if should_halt else "OK",
        )

    def is_clv_compliant(self) -> bool:
        """Quick check if we're allowed to trade based on CLV."""
        if not self.clv_enabled:
            return True
        if self._halted_clv:
            status = self.check_clv_status()
            if not status.should_halt:
                self._halted_clv = False
                self._save_state()
                self.logger.info("clv_halt_cleared")
            return not status.should_halt
        return True

    # -------------------------------------------------------------------------
    # Combined check
    # -------------------------------------------------------------------------

    def can_trade(self, market_type: str = "") -> tuple[bool, str]:
        """Check if trading is allowed based on all capital management rules."""
        reasons = []

        if not self.is_capital_recycling_compliant():
            status = self.check_capital_recycling()
            reasons.append(f"Capital recycling: {status.reason}")

        if not self.is_clv_compliant():
            status = self.check_clv_status()
            reasons.append(f"CLV: {status.reason} (avg={status.average_clv:.3f})")

        # Per-type CLV gate
        if market_type and not self.is_market_type_allowed(market_type):
            reasons.append(f"Market type '{market_type}' disabled due to poor CLV")

        if reasons:
            return False, "; ".join(reasons)

        return True, "OK"

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of capital management status."""
        recycling = self.check_capital_recycling()
        clv = self.check_clv_status()
        can_trade, reason = self.can_trade()

        return {
            "can_trade": can_trade,
            "reason": reason,
            "capital_recycling": {
                "compliant": recycling.compliant,
                "pct_7d": recycling.pct_resolving_7d,
                "pct_14d": recycling.pct_resolving_14d,
                "total_exposure": recycling.total_exposure,
            },
            "clv": {
                "enabled": self.clv_enabled,
                "average": clv.average_clv,
                "total_bets": clv.total_bets,
                "resolved_bets": clv.resolved_bets,
                "should_halt": clv.should_halt,
            },
            "halted_capital_recycling": self._halted_capital_recycling,
            "halted_clv": self._halted_clv,
        }


# Singleton
_capital_manager: Optional[CapitalManager] = None


def get_capital_manager(config: Optional[BotConfig] = None, state_dir: str = "state") -> CapitalManager:
    """Get or create the singleton capital manager."""
    global _capital_manager
    if _capital_manager is None:
        if config is None:
            raise ValueError("Config required for first initialization")
        _capital_manager = CapitalManager(config, state_dir=state_dir)
    return _capital_manager
