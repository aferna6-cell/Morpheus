"""Base signal interface for trading signals."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..markets import Market


class TradingSide(Enum):
    """Trading side recommendation."""

    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    HOLD = "hold"


@dataclass
class SignalResult:
    """Result from signal evaluation."""

    # Core signal outputs
    estimated_prob: float  # Estimated true probability (0-1)
    confidence: float  # Confidence in the estimate (0-1)
    edge: float  # Estimated edge (estimated_prob - market_price)
    recommended_side: TradingSide  # Trading recommendation
    reasoning: str  # Human-readable explanation

    # Additional metadata
    signal_name: str = ""
    market_price: Optional[float] = None  # Market price used for edge calculation
    timestamp: Optional[str] = None  # When signal was generated

    @property
    def has_edge(self) -> bool:
        """Check if signal indicates positive expected value."""
        return abs(self.edge) > 0.02  # At least 2% edge

    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (has edge and confidence)."""
        return self.has_edge and self.confidence > 0.5

    @property
    def risk_adjusted_edge(self) -> float:
        """Edge adjusted by confidence."""
        return self.edge * self.confidence

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"{self.signal_name}: {self.recommended_side.value} "
            f"(edge: {self.edge:.3f}, conf: {self.confidence:.3f})"
        )


class Signal(ABC):
    """Abstract base class for trading signals."""

    def __init__(self, name: str):
        """Initialize signal with a name."""
        self.name = name

    @abstractmethod
    async def evaluate(self, market: Market) -> SignalResult:
        """Evaluate a market and return trading signal.

        Args:
            market: Market to evaluate

        Returns:
            SignalResult with probability estimate and recommendation
        """
        pass

    def _calculate_edge(self, estimated_prob: float, market_price: float) -> float:
        """Calculate edge given estimated probability and market price.

        Edge represents expected profit as fraction of bet size.
        Positive edge means estimated probability is higher than market price.
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        # For YES token: edge = estimated_prob - market_price
        # This is the expected return per dollar bet
        return estimated_prob - market_price

    def _determine_side(self, edge: float, min_edge: float = 0.02) -> TradingSide:
        """Determine trading side based on edge."""
        if edge > min_edge:
            return TradingSide.BUY_YES
        elif edge < -min_edge:
            return TradingSide.BUY_NO
        else:
            return TradingSide.HOLD

    def _validate_probability(self, prob: float) -> float:
        """Validate and clamp probability to valid range."""
        return max(0.001, min(0.999, prob))  # Avoid exactly 0 or 1

    def _validate_confidence(self, conf: float) -> float:
        """Validate and clamp confidence to valid range."""
        return max(0.0, min(1.0, conf))
