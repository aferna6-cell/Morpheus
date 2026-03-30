"""Trading engines — modular signal generators for Morpheus."""

from .base import BaseEngine
from .kalshi_llm_engine import KalshiLLMEngine
from .kalshi_orderbook_engine import KalshiOrderbookEngine
from .kalshi_rules_engine import KalshiRulesEngine
from .signals import TradeSignal

__all__ = [
    "BaseEngine",
    "KalshiLLMEngine",
    "KalshiOrderbookEngine",
    "KalshiRulesEngine",
    "TradeSignal",
]
