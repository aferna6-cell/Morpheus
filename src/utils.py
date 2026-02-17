"""Utilities for the Polymarket trading bot."""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
import yaml
from pydantic import BaseModel, Field


class BotConfig(BaseModel):
    """Configuration model for the trading bot."""

    # API Configuration
    polymarket: Dict[str, Any] = Field(default_factory=dict)

    # Strategy Configuration
    strategy: Dict[str, Any] = Field(default_factory=dict)

    # Market Filtering
    market_filters: Dict[str, Any] = Field(default_factory=dict)

    # LLM Configuration
    llm: Dict[str, Any] = Field(default_factory=dict)

    # News Configuration
    news: Dict[str, Any] = Field(default_factory=dict)

    # Risk Management
    risk: Dict[str, Any] = Field(default_factory=dict)

    # Timing Configuration
    timing: Dict[str, Any] = Field(default_factory=dict)

    # Logging Configuration
    logging: Dict[str, Any] = Field(default_factory=dict)

    # Execution Configuration
    execution: Dict[str, Any] = Field(default_factory=dict)

    # Development Configuration
    dev: Dict[str, Any] = Field(default_factory=dict)

    # Kalshi Integration
    kalshi: Dict[str, Any] = Field(default_factory=dict)

    # Bregman Arbitrage (single-market + event-group mispricing)
    bregman_arb: Dict[str, Any] = Field(default_factory=dict)

    # Cross-Platform Arbitrage
    arbitrage: Dict[str, Any] = Field(default_factory=dict)

    # Copy Trading Configuration
    copy_trading: Dict[str, Any] = Field(default_factory=dict)

    # Sports Filters
    sports_filters: Dict[str, Any] = Field(default_factory=dict)

    # CLV Tracking
    clv_tracking: Dict[str, Any] = Field(default_factory=dict)

    # WebSocket Configuration
    websocket: Dict[str, Any] = Field(default_factory=dict)

    # Spike Detection Configuration
    spike_detection: Dict[str, Any] = Field(default_factory=dict)

    # Kalshi Flow Engine (follow large trades)
    kalshi_flow: Dict[str, Any] = Field(default_factory=dict)

    # Kalshi Price Monitor (spike detection)
    kalshi_monitor: Dict[str, Any] = Field(default_factory=dict)

    # Orchestrator Configuration
    orchestrator: Dict[str, Any] = Field(default_factory=dict)

    # Market Making
    market_making: Dict[str, Any] = Field(default_factory=dict)

    # Contrarian Strategy
    contrarian: Dict[str, Any] = Field(default_factory=dict)

    # Survival Mode
    survival: Dict[str, Any] = Field(default_factory=dict)

    # Alerts (Telegram)
    alerts: Dict[str, Any] = Field(default_factory=dict)

    # 15-minute crypto engine
    crypto_engine: Dict[str, Any] = Field(default_factory=dict)

    # Bracket arbitrage scanner
    bracket_arb: Dict[str, Any] = Field(default_factory=dict)

    # Theta decay engine (time-decay seller)
    theta_engine: Dict[str, Any] = Field(default_factory=dict)

    # Economic data release sniping
    econ_sniping: Dict[str, Any] = Field(default_factory=dict)

    # Wave 23: calibration knobs (Platt alphas, shrink, per-type)
    calibration: Dict[str, Any] = Field(default_factory=dict)

    # Bonding engine (near-certain outcome harvesting)
    bonding: Dict[str, Any] = Field(default_factory=dict)

    # Longshot seller (favorite-longshot bias exploitation)
    longshot_seller: Dict[str, Any] = Field(default_factory=dict)

    # Cross-platform arbitrage (Polymarket → Kalshi price signals)
    cross_arb: Dict[str, Any] = Field(default_factory=dict)


def setup_logging(
    config: BotConfig,
    *,
    log_json: Optional[bool] = None,
    log_level: Optional[str] = None,
) -> structlog.stdlib.BoundLogger:
    """Set up structured logging.

    Args:
        log_json: If set, forces JSON (True) or human console (False) output.
        log_level: Optional override (e.g. "DEBUG").
    """

    log_config = config.logging

    fmt = log_config.get("format", "json")
    if log_json is True:
        fmt = "json"
    elif log_json is False:
        fmt = "human"

    level_name = (log_level or log_config.get("level", "INFO")).upper()

    renderer = (
        structlog.dev.ConsoleRenderer(colors=True)
        if fmt == "human"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # structlog emits via stdlib logging; configure the root handler.
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level_name, logging.INFO),
        format="%(message)s",
    )

    return structlog.get_logger()


def load_config(config_path: str = "config.yaml") -> BotConfig:
    """Load configuration from YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found")

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    return BotConfig(**config_data)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json_state(data: Dict[str, Any], file_path: str | Path) -> None:
    """Save state data to JSON file."""
    p = Path(file_path)
    if p.parent:
        ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json_state(file_path: str | Path) -> Optional[Dict[str, Any]]:
    """Load state data from JSON file."""
    p = Path(file_path)
    try:
        with open(p, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def append_jsonl(path: str | Path, event: Dict[str, Any]) -> None:
    p = Path(path)
    if p.parent:
        ensure_dir(p.parent)
    with open(p, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def format_usd(amount: float) -> str:
    """Format amount as USD string."""
    return f"${amount:,.2f}"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


async def exponential_backoff(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True
) -> None:
    """Exponential backoff with optional jitter."""
    import random

    delay = min(base_delay * (2**attempt), max_delay)

    if jitter:
        delay *= 0.5 + random.random() * 0.5  # Add ±50% jitter

    await asyncio.sleep(delay)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_calls: int, time_window: float):
        """Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    async def acquire(self) -> None:
        """Acquire permission to make a call."""
        now = utc_now().timestamp()

        # Remove old calls outside the time window
        self.calls = [
            call_time for call_time in self.calls if now - call_time < self.time_window
        ]

        # If we're at the limit, wait
        if len(self.calls) >= self.max_calls:
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        # Record this call
        self.calls.append(now)


def validate_ethereum_address(address: str) -> bool:
    """Validate Ethereum address format."""
    if not address.startswith("0x"):
        return False

    if len(address) != 42:
        return False

    try:
        int(address[2:], 16)
        return True
    except ValueError:
        return False


def validate_private_key(private_key: str) -> bool:
    """Validate Ethereum private key format."""
    if private_key.startswith("0x"):
        private_key = private_key[2:]

    if len(private_key) != 64:
        return False

    try:
        int(private_key, 16)
        return True
    except ValueError:
        return False


def calculate_kelly_fraction(edge: float, odds: float, fraction: float = 1.0) -> float:
    """Calculate Kelly criterion position size.

    Args:
        edge: Expected edge (probability - market_price)
        odds: Odds against (1 / market_price - 1)
        fraction: Fraction of Kelly to use (default 1.0 = full Kelly)

    Returns:
        Position size as fraction of bankroll
    """
    if odds <= 0:
        return 0.0

    kelly = edge * (1.0 + odds) / odds
    return max(0.0, kelly * fraction)
