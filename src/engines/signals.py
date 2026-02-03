"""Unified trade signal format used by all engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass
class TradeSignal:
    """Signal emitted by an engine recommending a trade.

    All engines (spike, copy, arb, llm) produce signals in this format so the
    orchestrator can rank, de-duplicate, and route them consistently.
    """

    engine: str          # "spike", "copy", "arb", "llm"
    market_id: str
    token_id: str
    side: str            # "buy_yes", "buy_no", "sell"
    confidence: float    # 0-1
    edge: float          # estimated edge (positive = favourable)
    urgency: str         # "immediate", "normal", "low"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_buy(self) -> bool:
        return self.side in {"buy_yes", "buy_no"}

    @property
    def is_sell(self) -> bool:
        return self.side == "sell"

    def __repr__(self) -> str:
        return (
            f"TradeSignal(engine={self.engine!r}, side={self.side!r}, "
            f"confidence={self.confidence:.2f}, edge={self.edge:+.4f}, "
            f"urgency={self.urgency!r}, market={self.market_id[:12]}…)"
        )
