"""Base class for all trading engines."""

from __future__ import annotations

import abc
from typing import List

from .signals import TradeSignal


class BaseEngine(abc.ABC):
    """Abstract engine interface.

    Every engine runs as a long-lived async task that monitors some data source
    (WebSocket feed, copy-trader stream, etc.) and emits :class:`TradeSignal`
    objects for the orchestrator to consume.
    """

    name: str = "base"

    @abc.abstractmethod
    async def start(self) -> None:
        """Start the engine (connect feeds, spawn background tasks, etc.)."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the engine."""

    @abc.abstractmethod
    async def get_signals(self) -> List[TradeSignal]:
        """Drain and return any pending trade signals."""
