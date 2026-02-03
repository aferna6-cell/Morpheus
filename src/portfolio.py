"""Portfolio management for the Polymarket trading bot."""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

from .client import PolymarketClient
from .markets import Market
from .utils import BotConfig, format_usd, load_json_state, save_json_state


class PositionStatus(Enum):
    """Position status."""

    OPEN = "open"
    CLOSED = "closed"
    RESOLVED = "resolved"


@dataclass
class Position:
    """Represents a trading position."""

    market_id: str
    token_id: str
    side: str  # "buy_yes" or "buy_no"
    entry_price: float
    entry_amount: float
    entry_time: datetime
    current_price: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN

    # Performance metrics
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Exit information
    exit_price: Optional[float] = None
    exit_amount: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""

    # Market resolution
    resolved_outcome: Optional[str] = None
    resolution_time: Optional[datetime] = None

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate current unrealized P&L."""
        if self.status != PositionStatus.OPEN:
            return 0.0

        self.current_price = current_price

        # For YES positions: profit when price goes up
        # For NO positions: profit when price goes down
        if "yes" in self.side.lower():
            pnl = (current_price - self.entry_price) * self.entry_amount
        else:  # NO position
            pnl = (self.entry_price - current_price) * self.entry_amount

        self.unrealized_pnl = pnl
        return pnl

    def close_position(
        self, exit_price: float, exit_amount: Optional[float] = None, reason: str = ""
    ) -> float:
        """Close the position and calculate realized P&L."""
        if self.status != PositionStatus.OPEN:
            return 0.0

        exit_amount = exit_amount or self.entry_amount

        # Calculate realized P&L
        if "yes" in self.side.lower():
            realized_pnl = (exit_price - self.entry_price) * exit_amount
        else:  # NO position
            realized_pnl = (self.entry_price - exit_price) * exit_amount

        # Update position
        self.status = PositionStatus.CLOSED
        self.exit_price = exit_price
        self.exit_amount = exit_amount
        self.exit_time = datetime.now(timezone.utc)
        self.exit_reason = reason
        self.realized_pnl = realized_pnl
        self.unrealized_pnl = 0.0

        return realized_pnl

    def resolve_position(self, outcome: str, resolution_price: float = 1.0) -> float:
        """Resolve position based on market outcome."""
        if self.status == PositionStatus.RESOLVED:
            return self.realized_pnl

        # Determine if position won or lost
        position_won = False
        if "yes" in self.side.lower() and outcome.lower() in ["yes", "true", "1"]:
            position_won = True
        elif "no" in self.side.lower() and outcome.lower() in ["no", "false", "0"]:
            position_won = True

        # Calculate final P&L
        if position_won:
            # Position pays out at resolution_price (usually 1.0)
            final_pnl = (resolution_price - self.entry_price) * self.entry_amount
        else:
            # Position loses, pays out 0
            final_pnl = (0.0 - self.entry_price) * self.entry_amount

        # Update position
        self.status = PositionStatus.RESOLVED
        self.resolved_outcome = outcome
        self.resolution_time = datetime.now(timezone.utc)
        self.realized_pnl = final_pnl
        self.unrealized_pnl = 0.0

        return final_pnl

    @property
    def age_hours(self) -> float:
        """Get position age in hours."""
        now = datetime.now(timezone.utc)
        return (now - self.entry_time).total_seconds() / 3600

    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        total_pnl = self.realized_pnl + self.unrealized_pnl
        return total_pnl > 0


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""

    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    open_positions: int
    closed_positions: int
    resolved_positions: int

    # Performance ratios
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0


class Portfolio:
    """Manages trading portfolio and positions."""

    def __init__(
        self, config: BotConfig, client: PolymarketClient, *, state_dir: str = "."
    ):
        """Initialize portfolio manager."""
        self.config = config
        self.client = client
        self.state_dir = state_dir
        self.logger = structlog.get_logger()

        # Portfolio state
        from pathlib import Path

        self.positions: Dict[str, Position] = {}  # position_id -> Position
        self.state_file = str(Path(self.state_dir) / "portfolio_state.json")

        # Performance tracking
        self.starting_balance = config.dev.get("paper_trading_balance", 10000.0)
        self.daily_pnl_history: List[Dict[str, Any]] = []

        # Load previous state
        self._load_state()

    async def add_position(
        self,
        market: Market,
        token_id: str,
        side: str,
        entry_price: float,
        amount: float,
    ) -> str:
        """Add a new position to the portfolio."""
        position_id = (
            f"{market.id}_{token_id}_{side}_{datetime.now(timezone.utc).timestamp()}"
        )

        position = Position(
            market_id=market.id,
            token_id=token_id,
            side=side,
            entry_price=entry_price,
            entry_amount=amount,
            entry_time=datetime.now(timezone.utc),
        )

        self.positions[position_id] = position

        self.logger.info(
            "Position added",
            position_id=position_id,
            market_id=market.id,
            side=side,
            entry_price=entry_price,
            amount=amount,
        )

        # Save state
        await self._save_state()

        return position_id

    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_amount: Optional[float] = None,
        reason: str = "",
    ) -> float:
        """Close a specific position."""
        if position_id not in self.positions:
            self.logger.warning(
                "Attempted to close non-existent position", position_id=position_id
            )
            return 0.0

        position = self.positions[position_id]
        realized_pnl = position.close_position(exit_price, exit_amount, reason)

        self.logger.info(
            "Position closed",
            position_id=position_id,
            realized_pnl=realized_pnl,
            reason=reason,
        )

        # Save state
        await self._save_state()

        return realized_pnl

    async def update_position_prices(self) -> None:
        """Update current prices for all open positions."""
        open_positions = [
            p for p in self.positions.values() if p.status == PositionStatus.OPEN
        ]

        if not open_positions:
            return

        self.logger.debug("Updating position prices", count=len(open_positions))

        # Get current prices for all tokens
        for position in open_positions:
            try:
                current_price = await self.client.get_midpoint(position.token_id)
                if current_price is not None:
                    position.calculate_unrealized_pnl(current_price)
                else:
                    self.logger.warning(
                        "Failed to get price update", token_id=position.token_id
                    )
            except Exception as e:
                self.logger.error(
                    "Error updating position price",
                    position_id=position.token_id,
                    error=str(e),
                )

        # Save updated state
        await self._save_state()

    async def check_position_exits(self, risk_manager) -> List[str]:
        """Check if any positions should be closed based on risk rules."""
        positions_to_close = []

        for position_id, position in self.positions.items():
            if position.status != PositionStatus.OPEN:
                continue

            if position.current_price is None:
                continue

            # Check if position should be closed
            should_close, reason = risk_manager.should_close_position(
                entry_price=position.entry_price,
                current_price=position.current_price,
                entry_time=position.entry_time,
                position_amount=position.entry_amount,
            )

            if should_close:
                positions_to_close.append(position_id)
                self.logger.info(
                    "Position flagged for closure",
                    position_id=position_id,
                    reason=reason,
                    current_price=position.current_price,
                    entry_price=position.entry_price,
                )

        return positions_to_close

    async def resolve_markets(self, resolved_markets: Dict[str, str]) -> float:
        """Resolve positions for markets that have closed."""
        total_realized_pnl = 0.0

        for market_id, outcome in resolved_markets.items():
            # Find positions for this market
            market_positions = [
                (pid, pos)
                for pid, pos in self.positions.items()
                if pos.market_id == market_id and pos.status == PositionStatus.OPEN
            ]

            for position_id, position in market_positions:
                realized_pnl = position.resolve_position(outcome)
                total_realized_pnl += realized_pnl

                self.logger.info(
                    "Position resolved",
                    position_id=position_id,
                    market_id=market_id,
                    outcome=outcome,
                    realized_pnl=realized_pnl,
                )

        if total_realized_pnl != 0:
            await self._save_state()

        return total_realized_pnl

    def get_metrics(self) -> PortfolioMetrics:
        """Calculate and return portfolio metrics."""
        # Calculate P&L
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_pnl = unrealized_pnl + realized_pnl

        # Count positions by status
        open_positions = len(
            [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
        )
        closed_positions = len(
            [p for p in self.positions.values() if p.status == PositionStatus.CLOSED]
        )
        resolved_positions = len(
            [p for p in self.positions.values() if p.status == PositionStatus.RESOLVED]
        )

        # Calculate performance metrics
        closed_and_resolved = [
            p
            for p in self.positions.values()
            if p.status in [PositionStatus.CLOSED, PositionStatus.RESOLVED]
        ]

        win_rate = 0.0
        average_win = 0.0
        average_loss = 0.0
        profit_factor = 0.0

        if closed_and_resolved:
            winners = [p for p in closed_and_resolved if p.realized_pnl > 0]
            losers = [p for p in closed_and_resolved if p.realized_pnl < 0]

            win_rate = len(winners) / len(closed_and_resolved)

            if winners:
                average_win = sum(p.realized_pnl for p in winners) / len(winners)

            if losers:
                average_loss = abs(sum(p.realized_pnl for p in losers) / len(losers))

            if average_loss > 0:
                profit_factor = average_win / average_loss

        return PortfolioMetrics(
            total_value=self.starting_balance + total_pnl,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_pnl=total_pnl,
            open_positions=open_positions,
            closed_positions=closed_positions,
            resolved_positions=resolved_positions,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
        )

    def get_current_exposure(self) -> Dict[str, float]:
        """Get current position exposure by market."""
        exposure = {}

        for position in self.positions.values():
            if position.status == PositionStatus.OPEN:
                market_id = position.market_id
                amount = position.entry_amount * position.entry_price  # USD exposure

                if market_id in exposure:
                    exposure[market_id] += amount
                else:
                    exposure[market_id] = amount

        return exposure

    def get_position_summary(self) -> Dict[str, Any]:
        """Get a summary of all positions."""
        summary = {
            "open_positions": [],
            "recent_closed": [],
            "performance": asdict(self.get_metrics()),
        }

        # Open positions
        for position_id, position in self.positions.items():
            if position.status == PositionStatus.OPEN:
                summary["open_positions"].append(
                    {
                        "position_id": position_id,
                        "market_id": position.market_id,
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "amount": position.entry_amount,
                        "age_hours": position.age_hours,
                        "current_price": position.current_price,
                        "unrealized_pnl": position.unrealized_pnl,
                    }
                )

        # Recent closed positions (last 10)
        closed_positions = [
            (pid, pos)
            for pid, pos in self.positions.items()
            if pos.status in [PositionStatus.CLOSED, PositionStatus.RESOLVED]
        ]
        closed_positions.sort(
            key=lambda x: x[1].exit_time or x[1].resolution_time, reverse=True
        )

        for position_id, position in closed_positions[:10]:
            summary["recent_closed"].append(
                {
                    "position_id": position_id,
                    "market_id": position.market_id,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "exit_price": (
                        position.exit_price or 1.0 if position.resolved_outcome else 0.0
                    ),
                    "realized_pnl": position.realized_pnl,
                    "exit_reason": position.exit_reason
                    or f"Resolved: {position.resolved_outcome}",
                }
            )

        return summary

    async def _save_state(self) -> None:
        """Save portfolio state to file."""
        try:
            # Convert positions to serializable format
            positions_data = {}
            for position_id, position in self.positions.items():
                pos_data = asdict(position)
                # Convert datetime objects to ISO strings
                if pos_data["entry_time"]:
                    pos_data["entry_time"] = position.entry_time.isoformat()
                if pos_data["exit_time"]:
                    pos_data["exit_time"] = position.exit_time.isoformat()
                if pos_data["resolution_time"]:
                    pos_data["resolution_time"] = position.resolution_time.isoformat()

                # Ensure enums serialize as their underlying value
                pos_data["status"] = (
                    position.status.value if isinstance(position.status, PositionStatus) else str(position.status)
                )

                positions_data[position_id] = pos_data

            state = {
                "positions": positions_data,
                "starting_balance": self.starting_balance,
                "last_update": datetime.now(timezone.utc).isoformat(),
            }

            save_json_state(state, self.state_file)

        except Exception as e:
            self.logger.error("Failed to save portfolio state", error=str(e))

    def _load_state(self) -> None:
        """Load portfolio state from file."""
        try:
            state = load_json_state(self.state_file)
            if not state:
                return

            self.starting_balance = state.get("starting_balance", self.starting_balance)

            # Load positions
            positions_data = state.get("positions", {})
            for position_id, pos_data in positions_data.items():
                # Convert ISO strings back to datetime objects
                if pos_data["entry_time"]:
                    pos_data["entry_time"] = datetime.fromisoformat(
                        pos_data["entry_time"]
                    )
                if pos_data["exit_time"]:
                    pos_data["exit_time"] = datetime.fromisoformat(
                        pos_data["exit_time"]
                    )
                if pos_data["resolution_time"]:
                    pos_data["resolution_time"] = datetime.fromisoformat(
                        pos_data["resolution_time"]
                    )

                # Convert status back to enum (be robust to legacy serialization)
                raw_status = pos_data.get("status")
                if isinstance(raw_status, dict) and "value" in raw_status:
                    raw_status = raw_status["value"]
                if isinstance(raw_status, str):
                    # Legacy bug: enum got stringified as "PositionStatus.OPEN"
                    if raw_status.startswith("PositionStatus."):
                        raw_status = raw_status.split(".", 1)[1]
                    # Accept member names (OPEN/CLOSED/RESOLVED)
                    if raw_status in PositionStatus.__members__:
                        pos_data["status"] = PositionStatus[raw_status]
                    else:
                        pos_data["status"] = PositionStatus(raw_status.lower())
                else:
                    pos_data["status"] = PositionStatus.OPEN

                self.positions[position_id] = Position(**pos_data)

            self.logger.info("Portfolio state loaded", positions=len(self.positions))

        except Exception as e:
            self.logger.warning("Failed to load portfolio state", error=str(e))
