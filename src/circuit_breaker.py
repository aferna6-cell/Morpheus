"""Per-strategy circuit breaker for Morpheus V2.

Implements the CLOSED → OPEN → HALF_OPEN state machine described in
ARCHITECTURE.md Section 7.

State transitions:
  CLOSED  → OPEN:      5 consecutive failures (configurable)
  OPEN    → HALF_OPEN: 5-minute cooldown elapsed
  HALF_OPEN → CLOSED:  Single success
  HALF_OPEN → OPEN:    Any failure

All state is in-memory and resets on process restart, which is intentional —
a clean restart should allow strategies to recover.

Usage:
    breaker = CircuitBreaker()
    if breaker.is_open("index_fast_path"):
        return None   # skip this strategy

    try:
        result = await strategy.analyze(market)
        breaker.record_success("index_fast_path")
        return result
    except Exception as exc:
        breaker.record_failure("index_fast_path", exc)
        return None
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FAILURE_THRESHOLD: int = 5       # Trips CLOSED → OPEN after this many consecutive failures
DEFAULT_COOLDOWN_SECONDS: float = 300.0  # Time in OPEN state before moving to HALF_OPEN


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class BreakerState(str, Enum):
    CLOSED = "CLOSED"        # Normal operation
    OPEN = "OPEN"            # Tripped — strategy is skipped
    HALF_OPEN = "HALF_OPEN"  # Recovery probe — allow one attempt


@dataclass
class _StrategyState:
    """Internal per-strategy bookkeeping."""
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: Optional[datetime] = field(default=None)
    last_success_time: Optional[datetime] = field(default=None)
    last_state_change: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_error: Optional[str] = field(default=None)
    opened_at: Optional[datetime] = field(default=None)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Per-strategy isolation using the circuit-breaker pattern.

    All strategies share a single CircuitBreaker instance. State is tracked
    per strategy name (string key).

    Args:
        failure_threshold: Number of consecutive failures before tripping.
        cooldown_seconds: Seconds to wait in OPEN state before probing.
    """

    def __init__(
        self,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._strategies: dict[str, _StrategyState] = {}
        self._lock = asyncio.Lock()
        self.logger = structlog.get_logger()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_open(self, strategy: str) -> bool:
        """Return True if the strategy should be skipped right now.

        This advances OPEN → HALF_OPEN if the cooldown has elapsed.
        Calling this does NOT consume the HALF_OPEN probe slot — that
        happens only when record_success / record_failure is called.
        """
        st = self._get_or_create(strategy)
        if st.state == BreakerState.CLOSED:
            return False

        if st.state == BreakerState.OPEN:
            if self._cooldown_elapsed(st):
                self._transition(strategy, st, BreakerState.HALF_OPEN)
                self.logger.info(
                    "circuit_breaker_half_open",
                    strategy=strategy,
                    consecutive_failures=st.consecutive_failures,
                )
                return False  # Allow probe attempt
            return True  # Still in cooldown

        # HALF_OPEN: allow exactly one probe
        return False

    def record_success(self, strategy: str) -> None:
        """Record a successful strategy execution.

        HALF_OPEN → CLOSED on success.
        CLOSED: resets consecutive failure counter.
        """
        st = self._get_or_create(strategy)
        st.total_successes += 1
        st.last_success_time = datetime.now(timezone.utc)

        if st.state == BreakerState.HALF_OPEN:
            self.logger.info(
                "circuit_breaker_closed",
                strategy=strategy,
                total_failures=st.total_failures,
                total_successes=st.total_successes,
            )
            self._transition(strategy, st, BreakerState.CLOSED)
            st.consecutive_failures = 0
        elif st.state == BreakerState.CLOSED:
            st.consecutive_failures = 0

    def record_failure(self, strategy: str, error: Optional[Exception] = None) -> None:
        """Record a strategy failure.

        CLOSED:    increment counter; trip to OPEN if threshold reached.
        HALF_OPEN: immediately back to OPEN.
        OPEN:      no-op (already open).
        """
        st = self._get_or_create(strategy)
        st.consecutive_failures += 1
        st.total_failures += 1
        st.last_failure_time = datetime.now(timezone.utc)
        st.last_error = str(error) if error is not None else "unknown error"

        if st.state == BreakerState.CLOSED:
            if st.consecutive_failures >= self._failure_threshold:
                st.opened_at = datetime.now(timezone.utc)
                self._transition(strategy, st, BreakerState.OPEN)
                self.logger.warning(
                    "circuit_breaker_opened",
                    strategy=strategy,
                    consecutive_failures=st.consecutive_failures,
                    threshold=self._failure_threshold,
                    last_error=st.last_error,
                    cooldown_seconds=self._cooldown_seconds,
                )
            else:
                self.logger.debug(
                    "circuit_breaker_failure_recorded",
                    strategy=strategy,
                    consecutive_failures=st.consecutive_failures,
                    threshold=self._failure_threshold,
                    last_error=st.last_error,
                )

        elif st.state == BreakerState.HALF_OPEN:
            st.opened_at = datetime.now(timezone.utc)
            self._transition(strategy, st, BreakerState.OPEN)
            self.logger.warning(
                "circuit_breaker_reopened",
                strategy=strategy,
                last_error=st.last_error,
                cooldown_seconds=self._cooldown_seconds,
            )

    def get_state(self, strategy: str) -> BreakerState:
        """Return the current state for a strategy."""
        return self._get_or_create(strategy).state

    def get_consecutive_failures(self, strategy: str) -> int:
        """Return the current consecutive failure count for a strategy."""
        return self._get_or_create(strategy).consecutive_failures

    def reset(self, strategy: str) -> None:
        """Force a strategy circuit breaker back to CLOSED (manual override)."""
        st = self._get_or_create(strategy)
        self._transition(strategy, st, BreakerState.CLOSED)
        st.consecutive_failures = 0
        self.logger.info("circuit_breaker_reset", strategy=strategy)

    def all_states(self) -> dict[str, dict[str, object]]:
        """Return a snapshot of all circuit breaker states (for REST API / logging)."""
        snapshot: dict[str, dict[str, object]] = {}
        for name, st in self._strategies.items():
            snapshot[name] = {
                "state": st.state.value,
                "consecutive_failures": st.consecutive_failures,
                "total_failures": st.total_failures,
                "total_successes": st.total_successes,
                "last_failure": st.last_failure_time.isoformat() if st.last_failure_time else None,
                "last_success": st.last_success_time.isoformat() if st.last_success_time else None,
                "last_error": st.last_error,
                "opened_at": st.opened_at.isoformat() if st.opened_at else None,
            }
        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, strategy: str) -> _StrategyState:
        if strategy not in self._strategies:
            self._strategies[strategy] = _StrategyState()
        return self._strategies[strategy]

    def _cooldown_elapsed(self, st: _StrategyState) -> bool:
        if st.opened_at is None:
            return True
        elapsed = (datetime.now(timezone.utc) - st.opened_at).total_seconds()
        return elapsed >= self._cooldown_seconds

    def _transition(
        self, strategy: str, st: _StrategyState, new_state: BreakerState
    ) -> None:
        old_state = st.state
        st.state = new_state
        st.last_state_change = datetime.now(timezone.utc)
        self.logger.debug(
            "circuit_breaker_state_change",
            strategy=strategy,
            old_state=old_state.value,
            new_state=new_state.value,
        )
