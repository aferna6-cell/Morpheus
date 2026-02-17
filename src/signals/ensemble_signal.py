"""Ensemble LLM probability estimation — Claude Sonnet + GPT-4o in parallel.

Replaces the broken single-model approach (mushy-middle hack, abstention zone,
conviction gating) with a clean parallel ensemble that averages two independent
probability estimates.

Key differences from llm_signal.py:
- No mushy-middle calibration hack
- No abstention zone (25-75% BUY_YES block removed)
- No conviction gating (all positive-edge trades allowed)
- Parallel calls to Claude + GPT-4o (not sequential consensus)
- Lower min_edge: 3% instead of 5%
- Mild shrinkage toward 0.5 only
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog
from openai import AsyncOpenAI

try:
    from anthropic import AsyncAnthropic

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

from pathlib import Path

from ..cost_tracker import CostTracker
from ..markets import Market
from ..model_tracker import compute_model_weights, log_model_predictions, trimmed_mean, weighted_average
from ..news import NewsAggregator
from ..structured_data import get_structured_anchor, compute_weather_probability, compute_jobless_claims_probability, compute_stock_index_probability, get_recent_forecast_changes
from ..utils import BotConfig, RateLimiter
from .base import Signal, SignalResult, TradingSide


# ---------------------------------------------------------------------------
# Base rate database (reuse from llm_signal)
# ---------------------------------------------------------------------------

_BASE_RATES: List[Dict[str, Any]] = []


def _load_base_rates() -> List[Dict[str, Any]]:
    global _BASE_RATES
    if _BASE_RATES:
        return _BASE_RATES
    base_rates_path = Path(__file__).parent.parent / "base_rates.json"
    if base_rates_path.exists():
        try:
            with open(base_rates_path) as f:
                data = json.load(f)
            _BASE_RATES = data.get("patterns", [])
        except Exception:
            _BASE_RATES = []
    return _BASE_RATES


def lookup_base_rate(question: str, description: str = "") -> Optional[Dict[str, Any]]:
    rates = _load_base_rates()
    text = f"{question} {description}".lower()
    best_match = None
    best_score = 0
    for entry in rates:
        keywords = entry.get("keywords", [])
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_match = entry
    return best_match if best_score > 0 else None


# ---------------------------------------------------------------------------
# Market type detection — shared with llm_signal.py
# ---------------------------------------------------------------------------

# Import from llm_signal to keep detection logic in one place
from .llm_signal import (
    detect_market_type as _detect_market_type_llm,
    MARKET_TYPE_CALIBRATION as _MARKET_TYPE_CALIBRATION_LLM,
    MarketTypeCalibration,
)


def detect_market_type(question: str) -> str:
    """Classify market for skip/edge decisions. Delegates to llm_signal."""
    return _detect_market_type_llm(question)


# Markets to skip entirely — all types where LLM has no informational edge
_SKIP_TYPES = {
    "sports", "coin_flip", "announcer_mention", "word_mention",
    "crypto_range", "exact_phrase", "price_range", "entertainment",
}

# Min edge by market type — pull from full calibration profiles
_MIN_EDGE_BY_TYPE = {
    k: v.min_edge for k, v in _MARKET_TYPE_CALIBRATION_LLM.items()
}
# Ensure defaults
_MIN_EDGE_BY_TYPE.setdefault("normal", 0.05)
_MIN_EDGE_BY_TYPE.setdefault("politics", 0.05)
_MIN_EDGE_BY_TYPE.setdefault("economics", 0.05)


# ---------------------------------------------------------------------------
# Log-odds utilities — Wave 23: edge calculation in log-odds space
# ---------------------------------------------------------------------------

import math as _math


def prob_to_logodds(p: float) -> float:
    """Convert probability to log-odds. Clamps to avoid ±inf."""
    p = max(0.001, min(0.999, p))
    return _math.log(p / (1.0 - p))


def logodds_to_prob(lo: float) -> float:
    """Convert log-odds back to probability."""
    return 1.0 / (1.0 + _math.exp(-lo))


def edge_in_logodds(p_model: float, p_market: float) -> float:
    """Compute edge in log-odds space.

    In log-odds space, a fixed spread naturally has larger impact at extreme
    probabilities. A 0.2 logodds edge at p=0.50 is ~5c, but at p=0.95 is ~1.8c.
    This better captures informational advantage at extremes.
    """
    return prob_to_logodds(p_model) - prob_to_logodds(p_market)


# ---------------------------------------------------------------------------
# Calibration — symmetric shrinkage + asymmetric YES dampening
# ---------------------------------------------------------------------------

def calibrate_probability(
    p: float,
    *,
    shrink_strength: float = 0.03,
    platt_alpha: float = 0.68,
    floor: float = 0.05,
    ceiling: float = 0.95,
) -> float:
    """Apply Platt scaling then minimal shrinkage for calibration.

    Wave 25: fixed double-calibration bug — Platt FIRST (handles systematic
    overconfidence), then minimal shrinkage for per-type adjustment only.
    Previously shrinkage was applied first, compounding with Platt to crush
    edge by ~6pp instead of ~4pp.

    Alpha=0.68 fitted to our data: 58.9% NO WR implies systematic YES
    overestimation of ~15-20%. Maps: 80%→72%, 60%→57%, 90%→82%.
    Symmetric — works on both YES and NO extremes.
    """
    import math

    # Step 1: Platt scaling FIRST (handles systematic overconfidence)
    # sigmoid(alpha * logit(p)) — alpha < 1.0 = compression toward 0.5
    if 0 < platt_alpha < 1.0 and 0.001 < p < 0.999:
        logit_p = math.log(p / (1.0 - p))
        p = 1.0 / (1.0 + math.exp(-platt_alpha * logit_p))

    # Step 2: Minimal shrinkage for per-type adjustment only
    p = p * (1.0 - shrink_strength) + 0.5 * shrink_strength

    p = max(floor, min(ceiling, p))
    return round(p, 4)


# ---------------------------------------------------------------------------
# LLM response validation
# ---------------------------------------------------------------------------

def _validate_llm_response(data: Dict[str, Any]) -> Optional[str]:
    if not isinstance(data, dict):
        return "response is not a dict"
    p_yes = data.get("p_yes")
    if p_yes is None:
        return "missing p_yes"
    if not isinstance(p_yes, (int, float)):
        return f"p_yes is not a number: {type(p_yes)}"
    if not (0.0 <= float(p_yes) <= 1.0):
        return f"p_yes out of range: {p_yes}"
    return None


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------

def _round_price(price: float, step: float = 0.02) -> float:
    return round(round(price / step) * step, 4)


def _cache_key(market_id: str, question: str, price: float) -> str:
    return f"{market_id}:{question}:{_round_price(price)}"


class _ResultCache:
    def __init__(self, ttl_seconds: float = 1800.0, enabled: bool = True):
        self.ttl = ttl_seconds
        self.enabled = enabled
        self._store: Dict[str, Tuple[float, SignalResult]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[SignalResult]:
        if not self.enabled:
            return None
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        ts, result = entry
        if time.monotonic() - ts > self.ttl:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return result

    def put(self, key: str, result: SignalResult) -> None:
        if not self.enabled:
            return
        self._store[key] = (time.monotonic(), result)

    def evict_expired(self) -> int:
        now = time.monotonic()
        expired = [k for k, (ts, _) in self._store.items() if now - ts > self.ttl]
        for k in expired:
            del self._store[k]
        return len(expired)

    def reset_stats(self) -> Tuple[int, int]:
        h, m = self.hits, self.misses
        self.hits = self.misses = 0
        return h, m


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Ensemble Signal
# ---------------------------------------------------------------------------

class EnsembleSignal(Signal):
    """Parallel Claude + GPT-4o ensemble for probability estimation."""

    def __init__(self, config: BotConfig):
        super().__init__("Ensemble")
        self.config = config
        self.logger = structlog.get_logger()

        # LLM configuration
        llm_config = config.llm
        self.openai_model = llm_config.get("model", "gpt-4o")
        self.anthropic_model = llm_config.get("ensemble_anthropic_model", "claude-sonnet-4-20250514")
        self.temperature = llm_config.get("temperature", 0.1)
        self.max_tokens = llm_config.get("max_tokens", 1000)
        self.timeout = llm_config.get("timeout_seconds", 30)

        # Ensemble mode: "both", "openai_only", "anthropic_only", "full"
        # "full" = Wave 23 five-model trimmed mean ensemble
        self.ensemble_mode = llm_config.get("ensemble_mode", "openai_only")

        # Ensemble model config (for "full" mode)
        self.ensemble_models = llm_config.get("ensemble_models", [])

        # Screening config
        self.screening_enabled = llm_config.get("screening_enabled", True)
        self.screening_model = llm_config.get("screening_model", "gpt-4o-mini")

        # Clients — core (always initialized)
        self.openai_client = AsyncOpenAI(timeout=self.timeout)
        self._anthropic_client = None
        if _HAS_ANTHROPIC:
            try:
                self._anthropic_client = AsyncAnthropic(timeout=self.timeout)
            except Exception:
                self.logger.warning("anthropic_client_init_failed")

        # Clients — cheap ensemble models (Wave 23)
        # Only initialized if API key exists in environment
        self._mistral_client: Optional[AsyncOpenAI] = None
        self._deepseek_client: Optional[AsyncOpenAI] = None
        self._gemini_client: Optional[AsyncOpenAI] = None

        if os.environ.get("MISTRAL_API_KEY"):
            try:
                self._mistral_client = AsyncOpenAI(
                    api_key=os.environ["MISTRAL_API_KEY"],
                    base_url="https://api.mistral.ai/v1",
                    timeout=self.timeout,
                )
                self.logger.info("mistral_client_initialized")
            except Exception:
                self.logger.warning("mistral_client_init_failed")

        if os.environ.get("DEEPSEEK_API_KEY"):
            try:
                self._deepseek_client = AsyncOpenAI(
                    api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com",
                    timeout=self.timeout,
                )
                self.logger.info("deepseek_client_initialized")
            except Exception:
                self.logger.warning("deepseek_client_init_failed")

        if os.environ.get("GEMINI_API_KEY"):
            try:
                self._gemini_client = AsyncOpenAI(
                    api_key=os.environ["GEMINI_API_KEY"],
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    timeout=self.timeout,
                )
                self.logger.info("gemini_client_initialized")
            except Exception:
                self.logger.warning("gemini_client_init_failed")

        # News aggregator
        self.news_aggregator = NewsAggregator(config)

        # Rate limiting
        self.rate_limiter = RateLimiter(50, 60.0)

        # Edge parameters
        strategy = config.strategy
        self.fee_pct = float(strategy.get("fee_pct", 0.0))
        self.slippage_pct = float(strategy.get("slippage_pct", 0.005))

        # Market quality
        self.min_market_liquidity = float(
            config.market_filters.get("min_liquidity_aggressive", 200.0)
        )

        # Cost tracking
        self.cost_tracker: Optional[CostTracker] = None
        self._monthly_budget = float(llm_config.get("monthly_budget_usd", 50.0))
        self._daily_budget = float(llm_config.get("daily_budget_usd", 3.0))

        # Cache
        cache_ttl = float(llm_config.get("cache_ttl_minutes", 30)) * 60.0
        cache_enabled = llm_config.get("cache_enabled", True)
        self._cache = _ResultCache(ttl_seconds=cache_ttl, enabled=cache_enabled)

        # Calibration — Wave 23: read from config.calibration (was hardcoded)
        cal_config = getattr(config, "calibration", None) or {}
        self.calibration_shrink = float(
            cal_config.get("default_shrink", llm_config.get("calibration_shrink_strength", 0.03))
        )
        self._platt_alpha_default = float(cal_config.get("default_platt_alpha", 0.68))
        self._platt_alpha_weather = float(cal_config.get("weather_platt_alpha", 0.83))
        self._platt_alpha_weather_extreme = float(cal_config.get("weather_extreme_platt_alpha", 0.95))
        self._platt_alpha_weather_strong = float(cal_config.get("weather_strong_platt_alpha", 0.90))
        self._platt_alpha_index = float(cal_config.get("index_platt_alpha", 0.85))
        self._index_shrink = float(cal_config.get("index_shrink", 0.08))
        self._edge_cap = float(cal_config.get("edge_cap", 0.30))
        self._per_type_cal = cal_config.get("per_type", {})

        # Model weights (loaded periodically from resolved predictions)
        self._model_weights: Dict[str, float] = {}
        self._weights_loaded_at: float = 0.0
        self._weights_refresh_interval: float = 3600.0  # refresh hourly

        # Stats
        self._screened_total = 0
        self._screened_passed = 0
        self._cache_served = 0

    def set_cost_tracker(self, tracker: CostTracker) -> None:
        self.cost_tracker = tracker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(self, market: Market) -> SignalResult:
        """Evaluate market using parallel ensemble of Claude + GPT-4o."""
        try:
            self.logger.info(
                "ensemble_evaluate_start",
                market_id=market.id,
                question=market.question[:120],
            )

            # Market type detection first (fast-paths don't need LLM budget)
            mtype = detect_market_type(market.question)
            if mtype in _SKIP_TYPES:
                return self._hold(market, f"Skipped market type '{mtype}'")

            # Weather time-of-day filter: skip markets closing soon.
            # Threshold markets (T-prefix): 4h cutoff — thresholds closing in
            # 4-8h are our best edge (temps partially observed, NOAA most accurate).
            # Bracket markets (B-prefix): 8h cutoff — brackets are riskier near
            # close because edge effects matter more.
            if mtype == "weather" and market.time_to_close_hours is not None:
                is_threshold = "-T" in market.id and "-B" not in market.id
                weather_cutoff = 4.0 if is_threshold else 8.0
                if market.time_to_close_hours < weather_cutoff:
                    return self._hold(
                        market,
                        f"Weather {'threshold' if is_threshold else 'bracket'} "
                        f"closing in {market.time_to_close_hours:.1f}h "
                        f"(cutoff={weather_cutoff:.0f}h)",
                    )

            market_price = market.midpoint_price
            if market_price is None:
                return self._hold(market, "No market price available")

            # Weather forecast change detection: if NWS revised forecast by >=2°F,
            # invalidate any cached result for this market so we re-evaluate with fresh data.
            if mtype == "weather":
                recent_changes = get_recent_forecast_changes(since_seconds=600.0)
                if recent_changes:
                    ck_weather = _cache_key(market.id, market.question or "", market_price)
                    cached_weather = self._cache.get(ck_weather)
                    if cached_weather is not None:
                        # Force cache miss — re-evaluate with updated forecast
                        self._cache._store.pop(ck_weather, None)
                        self.logger.info(
                            "weather_cache_invalidated",
                            market_id=market.id,
                            changes=len(recent_changes),
                        )

            # Weather fast-path: bypass LLM when NOAA data is unambiguous
            if mtype == "weather":
                weather_result = await compute_weather_probability(
                    market.question, market.id, getattr(market, "close_time", market.end_date),
                )
                if weather_result is not None:
                    p_yes, w_confidence, w_reasoning = weather_result
                    # Compute edge directly (no calibration needed — this is hard data)
                    raw_edge = p_yes - market_price
                    net_edge = abs(raw_edge) - self.fee_pct - self.slippage_pct
                    net_edge = min(net_edge, self._edge_cap)  # Cap from config (default 0.30)
                    # Threshold markets (T-prefix) need less edge — one boundary,
                    # higher win rate. Brackets (B-prefix) need more — two edges.
                    is_weather_threshold = "-T" in market.id and "-B" not in market.id
                    min_edge = 0.08 if is_weather_threshold else 0.18  # Wave 23: bracket 30%→18% (was filtering all opportunities)

                    if net_edge >= min_edge:
                        if raw_edge > 0:
                            side = TradingSide.BUY_YES
                        elif raw_edge < 0:
                            side = TradingSide.BUY_NO
                        else:
                            side = TradingSide.HOLD

                        # Wave 29: Block weather buy_yes entirely.
                        # Data (Feb 10-16): threshold buy_yes 2W/9L -$6.74,
                        # bracket buy_yes 3W/7L -$1.56. NOAA overconfident
                        # on YES side — market prices are more accurate.
                        # Only weather buy_no is profitable (+$4.62 threshold, breakeven bracket).
                        if side == TradingSide.BUY_YES:
                            self.logger.info(
                                "weather_buy_yes_blocked",
                                market_id=market.id,
                                raw_edge=round(raw_edge, 4),
                                net_edge=round(net_edge, 4),
                                reason="Wave 29: weather buy_yes -$8.30 loss, blocked",
                            )
                            return self._hold(market, "Weather buy_yes blocked (Wave 29: -$8.30 loss)")

                        # Wave 30: Bracket buy_no price gate.
                        # Data: bracket buy_no profitable only when YES < 40c:
                        #   0-20c: 5W/0L +$0.80 | 20-40c: 11W/5L +$2.42
                        #   40-60c: 2W/5L -$2.68 | 60-100c: 0W/5L -$5.82
                        # When YES is cheap (<40c), NOAA confirms the market's NO bias
                        # (safe bet). When YES > 40c, NOAA disagrees with market = risky.
                        if not is_weather_threshold and side == TradingSide.BUY_NO:
                            if market_price >= 0.40:
                                self.logger.info(
                                    "weather_bracket_no_price_gate",
                                    market_id=market.id,
                                    market_price=round(market_price, 3),
                                    reason="Wave 30: bracket buy_no only profitable when YES < 40c",
                                )
                                return self._hold(market, "Bracket buy_no price gate: YES >= 40c")

                        # Payout ratio filter — skip for NOAA thresholds
                        # ("obvious bet" strategy: buy near-certain at 85-95c)
                        if side != TradingSide.HOLD and not is_weather_threshold:
                            entry_cost = market_price if side == TradingSide.BUY_YES else (1.0 - market_price)
                            payout_ratio = (1.0 - entry_cost) / entry_cost if entry_cost > 0 else 0
                            if payout_ratio < 0.12:
                                side = TradingSide.HOLD

                        conviction = "high" if net_edge >= 0.10 else "medium" if net_edge >= 0.05 else "low"

                        # Wave 29: Cap bracket buy_no confidence to reduce Kelly sizing.
                        # Data: bracket buy_no 16W/10L (62% WR) but -$2.15 PnL due to
                        # payoff asymmetry (avg loss 2.1x avg win). Capping confidence
                        # reduces position size, limiting damage from oversized losses.
                        if not is_weather_threshold:
                            w_confidence = min(w_confidence, 0.55)  # smaller Kelly sizing
                            conviction = "low"  # prevent high-conviction multiplier

                        reasoning = (
                            f"{w_reasoning} | "
                            f"mkt={market_price:.3f} raw_edge={raw_edge:+.3f} "
                            f"net_edge={net_edge:+.3f}"
                        )

                        self.logger.info(
                            "weather_fast_path",
                            market_id=market.id,
                            p_yes=round(p_yes, 4),
                            market_price=market_price,
                            raw_edge=round(raw_edge, 4),
                            net_edge=round(net_edge, 4),
                            side=side.value,
                            conviction=conviction,
                            is_bracket=not is_weather_threshold,
                        )

                        result = SignalResult(
                            estimated_prob=p_yes,
                            confidence=w_confidence,
                            edge=raw_edge,
                            recommended_side=side,
                            reasoning=reasoning,
                            signal_name=self.name,
                            market_price=market_price,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                        result.net_edge = net_edge  # type: ignore[attr-defined]
                        result.conviction = conviction  # type: ignore[attr-defined]
                        result.signal_source = "noaa_direct"  # type: ignore[attr-defined]
                        return result
                    else:
                        self.logger.debug(
                            "weather_fast_path_low_edge",
                            market_id=market.id,
                            net_edge=round(net_edge, 4),
                            min_edge=min_edge,
                        )
                        return self._hold(
                            market,
                            f"Weather direct: net edge {net_edge:.3f} < {min_edge:.3f}",
                        )
                else:
                    # NOAA couldn't produce a confident signal (z-score < 1.0).
                    # Skip LLM fallback for weather — historical 1W/33L shows LLM
                    # has no edge on weather, and near-threshold markets are the
                    # hardest to predict. Save LLM budget for non-weather markets.
                    self.logger.debug(
                        "weather_skip_noaa_ambiguous",
                        market_id=market.id,
                        msg="NOAA ambiguous, skipping LLM fallback",
                    )
                    return self._hold(market, "Weather: NOAA ambiguous, LLM has no edge")

            # Jobless claims fast-path: bypass LLM when FRED data gives clear signal
            if mtype == "economic" or "jobless" in market.question.lower() or "KXJOBLESSCLAIMS" in market.id.upper():
                claims_result = await compute_jobless_claims_probability(
                    market.question, market.id,
                )
                if claims_result is not None:
                    p_yes, jc_confidence, jc_reasoning = claims_result
                    raw_edge = p_yes - market_price
                    net_edge = abs(raw_edge) - self.fee_pct - self.slippage_pct
                    net_edge = min(net_edge, self._edge_cap)  # Cap from config (default 0.30)
                    min_edge = 0.08

                    if net_edge >= min_edge:
                        if raw_edge > 0:
                            side = TradingSide.BUY_YES
                        elif raw_edge < 0:
                            side = TradingSide.BUY_NO
                        else:
                            side = TradingSide.HOLD

                        # Wave 33: Block buy_yes on economics fast-paths.
                        # buy_yes is 8W/38L -$13.20 overall. Only index
                        # buy_yes is profitable. Economics buy_yes has no
                        # proven track record.
                        if side == TradingSide.BUY_YES:
                            self.logger.info(
                                "econ_buy_yes_blocked",
                                market_id=market.id,
                                raw_edge=round(raw_edge, 4),
                                reason="Wave 33: buy_yes blocked on economics",
                            )
                            return self._hold(market, "Economics buy_yes blocked (Wave 33)")

                        if side != TradingSide.HOLD:
                            entry_cost = market_price if side == TradingSide.BUY_YES else (1.0 - market_price)
                            payout_ratio = (1.0 - entry_cost) / entry_cost if entry_cost > 0 else 0
                            if payout_ratio < 0.12:
                                side = TradingSide.HOLD

                        conviction = "high" if net_edge >= 0.10 else "medium" if net_edge >= 0.05 else "low"

                        self.logger.info(
                            "jobless_claims_fast_path_signal",
                            market_id=market.id,
                            p_yes=round(p_yes, 4),
                            market_price=market_price,
                            net_edge=round(net_edge, 4),
                            side=side.value,
                        )

                        result = SignalResult(
                            estimated_prob=p_yes,
                            confidence=jc_confidence,
                            edge=raw_edge,
                            recommended_side=side,
                            reasoning=jc_reasoning,
                            signal_name=self.name,
                            market_price=market_price,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                        result.net_edge = net_edge  # type: ignore[attr-defined]
                        result.conviction = conviction  # type: ignore[attr-defined]
                        result.signal_source = "fred_direct"  # type: ignore[attr-defined]
                        return result
                    else:
                        return self._hold(
                            market,
                            f"Jobless claims direct: net edge {net_edge:.3f} < {min_edge:.3f}",
                        )

            # Stock index fast-path: Yahoo Finance real-time price + normal CDF
            _INDEX_PREFIXES = ("KXINXU", "KXINX-", "KXNASDAQ100",
                                "KXSPY", "KXQQQ", "KXIWM", "KXDIA",
                                "KXWTI", "KXGOLD")
            if any(market.id.upper().startswith(p) for p in _INDEX_PREFIXES):
                idx_result = await compute_stock_index_probability(
                    market.question, market.id,
                    getattr(market, "close_time", market.end_date),
                )
                if idx_result is not None:
                    p_yes, idx_confidence, idx_reasoning = idx_result
                    raw_edge = p_yes - market_price
                    net_edge = abs(raw_edge) - self.fee_pct - self.slippage_pct
                    net_edge = min(net_edge, self._edge_cap)  # Cap from config (default 0.30)
                    is_index_bracket = "-B" in market.id and "-T" not in market.id
                    min_edge = 0.20 if is_index_bracket else 0.05

                    if net_edge >= min_edge:
                        if raw_edge > 0:
                            side = TradingSide.BUY_YES
                        elif raw_edge < 0:
                            side = TradingSide.BUY_NO
                        else:
                            side = TradingSide.HOLD

                        conviction = "high" if net_edge >= 0.15 else "medium" if net_edge >= 0.08 else "low"
                        reasoning = (
                            f"{idx_reasoning} | "
                            f"mkt={market_price:.3f} raw_edge={raw_edge:+.3f} "
                            f"net_edge={net_edge:+.3f}"
                        )

                        self.logger.info(
                            "stock_index_fast_path_signal",
                            market_id=market.id,
                            p_yes=round(p_yes, 4),
                            market_price=market_price,
                            raw_edge=round(raw_edge, 4),
                            net_edge=round(net_edge, 4),
                            side=side.value,
                            conviction=conviction,
                        )

                        result = SignalResult(
                            estimated_prob=p_yes,
                            confidence=idx_confidence,
                            edge=raw_edge,
                            recommended_side=side,
                            reasoning=reasoning,
                            signal_name=self.name,
                            market_price=market_price,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                        result.net_edge = net_edge  # type: ignore[attr-defined]
                        result.conviction = conviction  # type: ignore[attr-defined]
                        result.signal_source = "yahoo_direct"  # type: ignore[attr-defined]
                        return result
                    else:
                        return self._hold(
                            market,
                            f"Stock index direct: net edge {net_edge:.3f} < {min_edge:.3f}",
                        )

            # Economic data release sniping: fast-path for CPI/jobs/GDP on release days
            if mtype == "economic" or any(w in (market.question or "").lower() for w in [
                "cpi", "inflation", "nonfarm", "payroll", "unemployment", "gdp", "ppi",
            ]):
                try:
                    from ..structured_data import check_economic_release
                    econ_result = await check_economic_release(
                        market.question, market.id,
                    )
                    if econ_result is not None:
                        p_yes, ec_confidence, ec_reasoning = econ_result
                        raw_edge = p_yes - market_price
                        net_edge = abs(raw_edge) - self.fee_pct - self.slippage_pct
                        min_edge = 0.05

                        if net_edge >= min_edge:
                            if raw_edge > 0:
                                side = TradingSide.BUY_YES
                            elif raw_edge < 0:
                                side = TradingSide.BUY_NO
                            else:
                                side = TradingSide.HOLD

                            # Wave 33: Block buy_yes on econ release path
                            if side == TradingSide.BUY_YES:
                                self.logger.info(
                                    "econ_release_buy_yes_blocked",
                                    market_id=market.id,
                                    raw_edge=round(raw_edge, 4),
                                    reason="Wave 33: buy_yes blocked on economics",
                                )
                                return self._hold(market, "Econ release buy_yes blocked (Wave 33)")

                            conviction = "high" if net_edge >= 0.10 else "medium"
                            reasoning = (
                                f"{ec_reasoning} | "
                                f"mkt={market_price:.3f} raw_edge={raw_edge:+.3f} "
                                f"net_edge={net_edge:+.3f}"
                            )

                            self.logger.info(
                                "econ_release_fast_path",
                                market_id=market.id,
                                p_yes=round(p_yes, 4),
                                market_price=market_price,
                                net_edge=round(net_edge, 4),
                                side=side.value,
                            )

                            result = SignalResult(
                                estimated_prob=p_yes,
                                confidence=ec_confidence,
                                edge=raw_edge,
                                recommended_side=side,
                                reasoning=reasoning,
                                signal_name=self.name,
                                market_price=market_price,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                            )
                            result.net_edge = net_edge  # type: ignore[attr-defined]
                            result.conviction = conviction  # type: ignore[attr-defined]
                            result.signal_source = "fred_release"  # type: ignore[attr-defined]
                            return result
                except Exception as e:
                    self.logger.debug("econ_release_check_error", error=str(e))

            # Wave 30: Skip LLM for non-fast-path markets entirely.
            # Data (Feb 10-16): LLM-only trades are 20W/41L (33% WR) -$19.36.
            # The LLM has no informational edge on markets without a data
            # fast-path (NOAA weather, Yahoo index, FRED economics).
            # All profitable trades come from fast-paths which already returned above.
            # Skipping saves ~$0.50/day LLM cost AND eliminates ~$4/day in losses.
            self.logger.debug(
                "llm_skip_no_fast_path",
                market_id=market.id,
                mtype=mtype,
                reason="No data fast-path matched, LLM has no edge",
            )
            return self._hold(market, f"No data fast-path for {mtype} — LLM skip (Wave 30)")

            # Budget check — AFTER fast-paths (NOAA/FRED/Yahoo cost $0, only LLM calls need budget)
            if self.cost_tracker and not self.cost_tracker.check_budget():
                return self._hold(market, "LLM budget exceeded")

            # Market quality gate
            if market.liquidity < self.min_market_liquidity:
                return self._hold(
                    market,
                    f"Liquidity ${market.liquidity:,.0f} < ${self.min_market_liquidity:,.0f}",
                )

            # Cache check
            ck = _cache_key(market.id, market.question or "", market_price)
            cached = self._cache.get(ck)
            if cached is not None:
                self._cache_served += 1
                return cached

            # Tier 1: screening
            if self.screening_enabled:
                self._screened_total += 1
                passed = await self._screen_market(market, market_price)
                if not passed:
                    return self._hold(market, "Screened out by tier-1 model")
                self._screened_passed += 1

            # Get news context + structured data in parallel
            # Skip news for weather markets — news search for "high temp 38°" is useless
            # and burns Brave Search quota. NOAA data is the only useful anchor.
            if mtype == "weather":
                news_articles = []
                news_context = ""
                structured_context = await get_structured_anchor(
                    market.question, market.category or "", market.id
                )
                if isinstance(structured_context, Exception):
                    structured_context = None
            else:
                news_task = self.news_aggregator.get_market_news(market)
                structured_task = get_structured_anchor(
                    market.question, market.category or "", market.id
                )
                news_articles, structured_context = await asyncio.gather(
                    news_task, structured_task, return_exceptions=True
                )
                if isinstance(news_articles, Exception):
                    self.logger.warning("news_fetch_error", error=str(news_articles))
                    news_articles = []
                if isinstance(structured_context, Exception):
                    structured_context = None
                news_context = self.news_aggregator.format_news_context(news_articles)

            # Build prompt (same for both models)
            prompt = self._build_prompt(
                market, news_context, market_price,
                structured_context=structured_context,
            )
            system = self._system_prompt()

            # Call models based on ensemble_mode
            openai_result = None
            anthropic_result = None
            mistral_result = None
            deepseek_result = None
            gemini_result = None

            if self.ensemble_mode == "full":
                # Wave 23: 5-model trimmed mean ensemble
                # Launch all available model calls in parallel
                tasks = {}
                tasks["openai"] = self._call_openai(system, prompt)
                if self._anthropic_client:
                    tasks["anthropic"] = self._call_anthropic(system, prompt)
                if self._mistral_client:
                    tasks["mistral"] = self._call_mistral(system, prompt)
                if self._deepseek_client:
                    tasks["deepseek"] = self._call_deepseek(system, prompt)
                if self._gemini_client:
                    tasks["gemini"] = self._call_gemini(system, prompt)

                task_names = list(tasks.keys())
                task_coros = list(tasks.values())
                results = await asyncio.gather(*task_coros, return_exceptions=True)

                for name, result in zip(task_names, results):
                    if isinstance(result, Exception):
                        self.logger.warning(
                            f"{name}_call_failed",
                            error=str(result),
                        )
                        continue
                    if name == "openai":
                        openai_result = result
                    elif name == "anthropic":
                        anthropic_result = result
                    elif name == "mistral":
                        mistral_result = result
                    elif name == "deepseek":
                        deepseek_result = result
                    elif name == "gemini":
                        gemini_result = result

                self.logger.info(
                    "full_ensemble_calls_complete",
                    market_id=market.id,
                    launched=len(tasks),
                    succeeded=sum(
                        1 for r in results if not isinstance(r, Exception) and r is not None
                    ),
                )

            elif self.ensemble_mode == "both":
                openai_task = self._call_openai(system, prompt)
                anthropic_task = self._call_anthropic(system, prompt)
                results = await asyncio.gather(
                    openai_task, anthropic_task, return_exceptions=True
                )
                openai_result = results[0] if not isinstance(results[0], Exception) else None
                anthropic_result = results[1] if not isinstance(results[1], Exception) else None
                if isinstance(results[0], Exception):
                    self.logger.warning("openai_call_failed", error=str(results[0]))
                if isinstance(results[1], Exception):
                    self.logger.warning("anthropic_call_failed", error=str(results[1]))
            elif self.ensemble_mode == "anthropic_only":
                try:
                    anthropic_result = await self._call_anthropic(system, prompt)
                except Exception as e:
                    self.logger.warning("anthropic_call_failed", error=str(e))
            else:
                # openai_only (default)
                try:
                    openai_result = await self._call_openai(system, prompt)
                except Exception as e:
                    self.logger.warning("openai_call_failed", error=str(e))

            # Extract probabilities from all model results
            p_values = []
            model_details = {}      # for logging: {openai_p_yes: 0.35, ...}
            model_predictions = {}  # for weighting: {model_name: p_yes}

            if openai_result and _validate_llm_response(openai_result) is None:
                p_openai = float(openai_result["p_yes"])
                p_openai = max(0.001, min(0.999, p_openai))
                p_values.append(p_openai)
                model_details["openai_p_yes"] = p_openai
                model_predictions[self.openai_model] = p_openai

            if anthropic_result and _validate_llm_response(anthropic_result) is None:
                p_anthropic = float(anthropic_result["p_yes"])
                p_anthropic = max(0.001, min(0.999, p_anthropic))
                p_values.append(p_anthropic)
                model_details["anthropic_p_yes"] = p_anthropic
                model_predictions[self.anthropic_model] = p_anthropic

            if mistral_result and _validate_llm_response(mistral_result) is None:
                p_mistral = float(mistral_result["p_yes"])
                p_mistral = max(0.001, min(0.999, p_mistral))
                p_values.append(p_mistral)
                model_details["mistral_p_yes"] = p_mistral
                model_predictions["mistral-small-latest"] = p_mistral

            if deepseek_result and _validate_llm_response(deepseek_result) is None:
                p_deepseek = float(deepseek_result["p_yes"])
                p_deepseek = max(0.001, min(0.999, p_deepseek))
                p_values.append(p_deepseek)
                model_details["deepseek_p_yes"] = p_deepseek
                model_predictions["deepseek-chat"] = p_deepseek

            if gemini_result and _validate_llm_response(gemini_result) is None:
                p_gemini = float(gemini_result["p_yes"])
                p_gemini = max(0.001, min(0.999, p_gemini))
                p_values.append(p_gemini)
                model_details["gemini_p_yes"] = p_gemini
                model_predictions["gemini-2.5-flash-preview-05-20"] = p_gemini

            if not p_values:
                return self._hold(market, "All LLM calls failed")

            # Refresh model weights periodically
            if time.monotonic() - self._weights_loaded_at > self._weights_refresh_interval:
                try:
                    self._model_weights = compute_model_weights()
                    self._weights_loaded_at = time.monotonic()
                except Exception:
                    pass  # weights loading is best-effort

            # Aggregation: trimmed mean for full mode (3+ models), weighted avg otherwise
            _high_divergence = False
            if self.ensemble_mode == "full" and len(p_values) >= 3:
                # Wave 23: trimmed mean — removes highest and lowest, averages middle
                p_yes_raw = trimmed_mean(model_predictions)
                # Use range (max - min) as divergence measure for 3+ models
                model_range = max(p_values) - min(p_values)
                if model_range > 0.35:
                    _high_divergence = True
                    self.logger.info(
                        "full_ensemble_high_spread",
                        market_id=market.id,
                        model_range=round(model_range, 3),
                        predictions={k: round(v, 3) for k, v in model_predictions.items()},
                    )
            else:
                # Legacy modes or fallback (1-2 models in full mode)
                p_yes_raw = weighted_average(model_predictions, self._model_weights)

                # Model disagreement gate (only for exactly 2 models)
                if len(p_values) == 2:
                    divergence = abs(p_values[0] - p_values[1])
                    if divergence > 0.35:
                        if self._model_weights and len(self._model_weights) >= 2:
                            _high_divergence = True
                            self.logger.info(
                                "ensemble_high_divergence_weighted",
                                market_id=market.id,
                                divergence=round(divergence, 3),
                            )
                        else:
                            self.logger.warning(
                                "ensemble_model_disagreement",
                                market_id=market.id,
                                divergence=round(divergence, 3),
                            )
                            return self._hold(
                                market,
                                f"Model disagreement too high ({divergence:.0%})",
                            )

            # Single-model flag — no cross-validation available.
            # Wave 25: removed p_yes shrinkage (was double-penalizing with
            # the 30% confidence reduction at line ~1204). Keep only the
            # confidence penalty which affects Kelly sizing (correct lever).
            _single_model = False
            if len(p_values) == 1:
                _single_model = True
                model_name = list(model_predictions.keys())[0]
                self.logger.info(
                    "single_model_signal",
                    market_id=market.id,
                    model=model_name,
                    p_yes=round(p_values[0], 4),
                )

            # Log per-model predictions for future weight computation
            if model_predictions:
                try:
                    log_model_predictions(
                        market_id=market.id,
                        model_predictions=model_predictions,
                        ensemble_p_yes=p_yes_raw,
                        market_price=market_price,
                        side="pending",
                    )
                except Exception:
                    pass  # logging is best-effort

            # Wave 23: per-type calibration from config (was hardcoded)
            type_cal = _MARKET_TYPE_CALIBRATION_LLM.get(
                mtype, MarketTypeCalibration()
            )
            # Check config-driven per-type overrides
            per_type_cfg = self._per_type_cal.get(mtype, {})
            cfg_extra_shrink = float(per_type_cfg.get("extra_shrink", type_cal.extra_shrink))
            total_shrink = min(0.40, self.calibration_shrink + cfg_extra_shrink)
            if _high_divergence:
                total_shrink = min(0.50, total_shrink + 0.10)

            # Wave 23: Platt alpha from config with per-type overrides
            platt_alpha = float(per_type_cfg.get("platt_alpha", self._platt_alpha_default))

            # Index markets: use dedicated config (9W/0L, trust more)
            if mtype == "index" or any(market.id.upper().startswith(p)
                                        for p in ("KXINXU", "KXINX-", "KXSPY", "KXQQQ", "KXIWM", "KXDIA")):
                platt_alpha = self._platt_alpha_index
                total_shrink = min(total_shrink, self._index_shrink)

            # Weather with structured data: trust data-anchored signals more
            if mtype == "weather" and isinstance(structured_context, str) and structured_context:
                platt_alpha = self._platt_alpha_weather

            # Reduce calibration when structured data shows extreme confidence.
            # Hard FRED/NOAA data should override generic LLM overconfidence adjustments.
            if isinstance(structured_context, str):
                if "FAR ABOVE" in structured_context or "FAR BELOW" in structured_context:
                    total_shrink *= 0.15
                    platt_alpha = self._platt_alpha_weather_extreme
                    self.logger.info(
                        "calibration_reduced_extreme_data",
                        market_id=market.id,
                        effective_shrink=round(total_shrink, 3),
                        platt_alpha=platt_alpha,
                    )
                elif "well above" in structured_context or "well below" in structured_context:
                    total_shrink *= 0.50
                    platt_alpha = self._platt_alpha_weather_strong
                    self.logger.info(
                        "calibration_reduced_strong_data",
                        market_id=market.id,
                        effective_shrink=round(total_shrink, 3),
                        platt_alpha=platt_alpha,
                    )
                elif mtype == "weather" and (
                    "CLOSE to threshold" in structured_context
                    or "CLOSE to bracket range" in structured_context
                    or "slightly above" in structured_context
                    or "slightly below" in structured_context
                ):
                    # NOAA forecast is within ~4°F of the market threshold/bracket.
                    # NWS forecasts have ±2-3°F accuracy, so this is within
                    # error margin — the market likely has better data. Skip.
                    self.logger.info(
                        "weather_skip_near_threshold",
                        market_id=market.id,
                        reason="NOAA forecast near threshold (within error margin)",
                    )
                    return self._hold(
                        market,
                        "Weather: NOAA forecast near threshold (within NWS error margin)",
                    )
                elif mtype == "weather" and "FAR OUTSIDE bracket range" in structured_context:
                    # NOAA forecast is 4°F+ from the bracket — genuine disagreement.
                    # Trust the data, reduce calibration to let the prediction through.
                    total_shrink *= 0.25
                    platt_alpha = 0.95  # near-passthrough
                    self.logger.info(
                        "calibration_reduced_weather_far_outside",
                        market_id=market.id,
                        effective_shrink=round(total_shrink, 3),
                        platt_alpha=platt_alpha,
                    )

            p_yes = calibrate_probability(
                p_yes_raw,
                shrink_strength=total_shrink,
                platt_alpha=platt_alpha,
            )

            self.logger.info(
                "ensemble_probabilities",
                market_id=market.id,
                **model_details,
                raw_avg=round(p_yes_raw, 4),
                calibrated=p_yes,
                models_used=len(p_values),
                platt_alpha=platt_alpha,
            )

            # Adversarial challenge: when p_yes would result in BUY_YES,
            # get a cheap second opinion. Wave 25: extended from 35-70% to
            # 25-85% — extreme predictions are MOST overconfident per KalshiBench.
            raw_edge_preliminary = p_yes - market_price
            if 0.25 <= p_yes <= 0.85 and raw_edge_preliminary > 0:
                p_yes = await self._challenge_estimate(market, p_yes, market_price)

            # Adversarial challenge for BUY_NO too.
            # Wave 25: extended from 30-65% to 15-75%.
            raw_no_edge = market_price - p_yes  # positive when NO has edge
            if 0.15 <= p_yes <= 0.75 and raw_no_edge > 0:
                p_yes = await self._challenge_estimate_no(market, p_yes, market_price)

            # Get metadata from whichever response succeeded
            primary = openai_result or anthropic_result or {}
            uncertainty = primary.get("uncertainty", "medium")
            key_facts = primary.get("key_facts", [])
            disqualifiers = primary.get("disqualifiers", [])
            rationale = primary.get("rationale", "")

            # Disqualifiers
            if disqualifiers:
                result = self._hold(
                    market,
                    f"Disqualifiers: {'; '.join(disqualifiers)}",
                )
                self._cache.put(ck, result)
                return result

            # Compute edge
            raw_edge = p_yes - market_price
            net_edge = abs(raw_edge) - self.fee_pct - self.slippage_pct
            net_edge = min(net_edge, self._edge_cap)  # Cap from config (default 0.30)

            # Wave 23: log-odds edge for better extreme-price behavior.
            # A fixed log-odds threshold naturally requires larger linear edge
            # at extreme prices (where information advantage matters more).
            logodds_edge = abs(edge_in_logodds(p_yes, market_price))

            # Category min edge — use log-odds threshold that maps to ~5% at p=0.50
            # logodds(0.55) - logodds(0.50) ≈ 0.20, so 0.20 logodds ≈ 5% at midrange
            min_logodds_edge = 0.20  # ~5% at p=0.50, ~2% at p=0.90, ~8% at p=0.15
            min_edge = _MIN_EDGE_BY_TYPE.get(mtype, 0.03)
            # Use the MORE restrictive of linear and log-odds thresholds
            logodds_min_linear = abs(logodds_to_prob(prob_to_logodds(market_price) + min_logodds_edge) - market_price)
            min_edge = max(min_edge, logodds_min_linear)
            if net_edge < min_edge:
                result = self._hold(
                    market,
                    f"Net edge {net_edge:.3f} < min {min_edge:.3f} (type={mtype})",
                )
                self._cache.put(ck, result)
                return result

            # Determine side — NO abstention zone, just edge-based
            if raw_edge > 0 and net_edge > 0:
                side = TradingSide.BUY_YES
            elif raw_edge < 0 and net_edge > 0:
                side = TradingSide.BUY_NO
            else:
                side = TradingSide.HOLD

            # Wave 33: Block LLM buy_yes ENTIRELY.
            # Data (Feb 10-16): buy_yes 8W/38L (17.4% WR), -$13.20.
            # Even with 25-30% min edge thresholds (Wave 21-22), buy_yes
            # consistently loses. Edge claims on losing trades are HIGHER
            # than winners (calibration broken). Only index buy_yes works
            # (100% WR via yahoo_direct fast-path, which returns before
            # reaching this code). Weather buy_yes already blocked above.
            if side == TradingSide.BUY_YES:
                self.logger.info(
                    "llm_buy_yes_blocked",
                    market_id=market.id,
                    raw_edge=round(raw_edge, 4),
                    net_edge=round(net_edge, 4),
                    market_price=round(market_price, 3),
                    reason="Wave 33: LLM buy_yes 8W/38L -$13.20, blocked",
                )
                result = self._hold(
                    market,
                    f"LLM buy_yes blocked (Wave 33: 17% WR, -$13.20). "
                    f"net_edge={net_edge:.3f}, mkt={market_price:.2f}",
                )
                self._cache.put(ck, result)
                return result
            elif side == TradingSide.BUY_NO:
                # NO-side discount: buying NO is structurally advantaged
                # (Becker 72.1M trades: makers buying NO earn +1.25% excess return)
                # Wave 22: apply 2% discount at ALL price levels (was only >70c)
                # Data: buy_no 58.9% WR +$90.15
                no_min_edge = max(0.03, min_edge - 0.02)  # always 2% easier entry
                if net_edge < no_min_edge:
                    result = self._hold(
                        market,
                        f"buy_no edge {net_edge:.3f} below {no_min_edge:.3f} (NO-side gate)",
                    )
                    self._cache.put(ck, result)
                    return result

            # Payout ratio filter — don't buy $0.95 contracts to win $0.05
            if side != TradingSide.HOLD:
                entry_cost = market_price if side == TradingSide.BUY_YES else (1.0 - market_price)
                payout_ratio = (1.0 - entry_cost) / entry_cost if entry_cost > 0 else 0
                if payout_ratio < 0.12:
                    side = TradingSide.HOLD
                    self.logger.debug(
                        "payout_ratio_reject",
                        market_id=market.id,
                        payout_ratio=f"{payout_ratio:.3f}",
                    )

            # Confidence from uncertainty
            confidence_map = {"low": 0.9, "medium": 0.75, "high": 0.55}
            confidence = confidence_map.get(uncertainty, 0.75)

            # Boost/penalize confidence based on model agreement
            if len(p_values) >= 3:
                # Full ensemble: use standard deviation as agreement measure
                import statistics
                model_std = statistics.stdev(p_values)
                if model_std < 0.03:
                    confidence = min(1.0, confidence + 0.10)  # tight agreement
                elif model_std > 0.12:
                    confidence = max(0.3, confidence - 0.15)  # wide disagreement
            elif len(p_values) == 2:
                divergence = abs(p_values[0] - p_values[1])
                if divergence < 0.05:
                    confidence = min(1.0, confidence + 0.10)
                elif divergence > 0.15:
                    confidence = max(0.3, confidence - 0.15)

            # Wave 23: single-model confidence penalty (30% reduction)
            if _single_model:
                confidence *= 0.70

            # Wave 22: sweet spot boost — 20-30% edge bucket is 64.5% WR +$66.85
            if 0.20 <= net_edge <= 0.30:
                confidence = min(0.95, confidence + 0.05)

            reasoning = (
                f"{rationale} | "
                f"p_yes={p_yes:.3f} mkt={market_price:.3f} "
                f"raw_edge={raw_edge:+.3f} net_edge={net_edge:+.3f} "
                f"models={len(p_values)} uncertainty={uncertainty}"
            )
            if key_facts:
                reasoning += f" | facts: {'; '.join(key_facts[:3])}"

            result = SignalResult(
                estimated_prob=p_yes,
                confidence=confidence,
                edge=raw_edge,
                recommended_side=side,
                reasoning=reasoning,
                signal_name=self.name,
                market_price=market_price,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            # Attach extras for downstream
            result.net_edge = net_edge  # type: ignore[attr-defined]
            result.conviction = "high" if net_edge >= 0.10 else "medium" if net_edge >= 0.05 else "low"  # type: ignore[attr-defined]
            result.signal_source = "llm"  # type: ignore[attr-defined]

            # Wave 23: log-odds edge for MM spread management
            result.logodds_edge = logodds_edge  # type: ignore[attr-defined]

            self.logger.info(
                "ensemble_signal",
                market_id=market.id,
                p_yes=p_yes,
                market_price=market_price,
                raw_edge=raw_edge,
                net_edge=net_edge,
                logodds_edge=round(logodds_edge, 3),
                side=side.value,
                confidence=confidence,
                models_used=len(p_values),
            )

            self._cache.put(ck, result)
            return result

        except Exception as e:
            self.logger.error("ensemble_evaluate_error", market_id=market.id, error=str(e))
            return self._hold(market, f"Error: {e}")

    def log_periodic_summary(self) -> None:
        cache_hits, cache_misses = self._cache.reset_stats()
        self.logger.info(
            "ensemble_cost_summary",
            screened_total=self._screened_total,
            screened_passed=self._screened_passed,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_served=self._cache_served,
        )
        self._screened_total = 0
        self._screened_passed = 0
        self._cache_served = 0
        self._cache.evict_expired()

    # ------------------------------------------------------------------
    # Screening (tier 1 — cheap model)
    # ------------------------------------------------------------------

    async def _screen_market(self, market: Market, market_price: float) -> bool:
        end_date_str = "Unknown"
        if market.end_date:
            end_date_str = market.end_date.strftime("%Y-%m-%d")

        prompt = (
            f"Q: {market.question}\n"
            f"Current price: {market_price:.0%} YES\n"
            f"Ends: {end_date_str} | Volume: ${market.volume_24h:,.0f}\n\n"
            "Is this market likely mispriced by >3%? "
            'Reply JSON: {{"worth_evaluating": true/false, "reason": "..."}}'
        )

        try:
            await self.rate_limiter.acquire()
            response = await self.openai_client.chat.completions.create(
                model=self.screening_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You screen prediction markets for mispricing opportunities. "
                            "Output ONLY valid JSON. Be aggressive — look for edge."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_call(
                    self.screening_model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            if not content:
                return True
            data = json.loads(content)
            return bool(data.get("worth_evaluating", True))

        except Exception as e:
            self.logger.warning("screening_error", error=str(e))
            return True  # fail open

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    async def _call_openai(self, system: str, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            await self.rate_limiter.acquire()
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_call(
                    self.openai_model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            if not content:
                return None
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                extracted = _extract_json_object(content)
                return json.loads(extracted) if extracted else None

        except Exception as e:
            self.logger.error("openai_call_error", error=str(e))
            return None

    async def _call_anthropic(self, system: str, prompt: str) -> Optional[Dict[str, Any]]:
        if not self._anthropic_client:
            return None
        try:
            response = await self._anthropic_client.messages.create(
                model=self.anthropic_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text if response.content else None
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_call(
                    self.anthropic_model,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )

            if not content:
                return None
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                extracted = _extract_json_object(content)
                return json.loads(extracted) if extracted else None

        except Exception as e:
            self.logger.warning("anthropic_call_error", error=str(e))
            return None

    # ------------------------------------------------------------------
    # Cheap ensemble model calls (Wave 23)
    # ------------------------------------------------------------------

    async def _call_mistral(self, system: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Mistral Small via OpenAI-compatible API."""
        if not self._mistral_client:
            return None
        try:
            await self.rate_limiter.acquire()
            response = await self._mistral_client.chat.completions.create(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_call(
                    "mistral-small-latest",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            if not content:
                return None
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                extracted = _extract_json_object(content)
                return json.loads(extracted) if extracted else None

        except Exception as e:
            self.logger.warning("mistral_call_error", error=str(e))
            return None

    async def _call_deepseek(self, system: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Call DeepSeek-V3 via OpenAI-compatible API."""
        if not self._deepseek_client:
            return None
        try:
            await self.rate_limiter.acquire()
            response = await self._deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_call(
                    "deepseek-chat",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            if not content:
                return None
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                extracted = _extract_json_object(content)
                return json.loads(extracted) if extracted else None

        except Exception as e:
            self.logger.warning("deepseek_call_error", error=str(e))
            return None

    async def _call_gemini(self, system: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Gemini 2.5 Flash via OpenAI-compatible API."""
        if not self._gemini_client:
            return None
        try:
            await self.rate_limiter.acquire()
            response = await self._gemini_client.chat.completions.create(
                model="gemini-2.5-flash-preview-05-20",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_call(
                    "gemini-2.5-flash-preview-05-20",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            if not content:
                return None
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                extracted = _extract_json_object(content)
                return json.loads(extracted) if extracted else None

        except Exception as e:
            self.logger.warning("gemini_call_error", error=str(e))
            return None

    # ------------------------------------------------------------------
    # Adversarial challenge — cheap second opinion on YES-leaning signals
    # ------------------------------------------------------------------

    async def _challenge_estimate(
        self, market: Market, p_yes: float, market_price: float
    ) -> float:
        """Run a cheap adversarial challenge when the signal is BUY_YES in the danger zone.

        When the initial estimate is 35-75% YES and the side would be BUY_YES,
        get a cheap GPT-4o-mini call arguing it's too high. Blend:
        60% original + 40% challenge estimate.

        Cost: ~$0.0002/call. Only triggered on danger-zone BUY_YES signals.
        """
        challenge_prompt = f"""A prediction market forecaster estimated {p_yes:.0%} YES for this question:

"{market.question}"

The market price is {market_price:.0%}. The forecaster wants to BUY YES.

Your job: critically evaluate whether {p_yes:.0%} is too high. Consider:
- What could go wrong? What obstacles exist?
- Is the forecaster anchored to a narrative?
- What's the base rate for events like this?
- Is the timeline realistic?

If you believe the estimate is reasonable, output your own independent estimate (which may be similar).
Output JSON: {{"p_yes": <your estimate, float 0.0-1.0>, "reason": "your reasoning"}}"""

        try:
            await self.rate_limiter.acquire()
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a critical analyst. Evaluate whether a probability estimate "
                            "is well-calibrated. If it's too high, explain why. If it seems "
                            "reasonable, say so. Output ONLY valid JSON."
                        ),
                    },
                    {"role": "user", "content": challenge_prompt},
                ],
                temperature=0.2,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_call(
                    "gpt-4o-mini",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            if not content:
                return p_yes

            data = json.loads(content)
            challenge_p = float(data.get("p_yes", p_yes))
            challenge_p = max(0.01, min(0.99, challenge_p))

            # Only blend when challenger meaningfully disagrees (>15% lower)
            if challenge_p < p_yes * 0.85:
                blended = 0.60 * p_yes + 0.40 * challenge_p
            else:
                blended = p_yes  # Challenger agrees — keep original

            self.logger.info(
                "adversarial_challenge",
                market_id=market.id,
                original_p_yes=p_yes,
                challenge_p_yes=challenge_p,
                blended_p_yes=round(blended, 4),
                reason=data.get("reason", "")[:100],
            )

            return round(blended, 4)

        except Exception as e:
            self.logger.warning("adversarial_challenge_error", error=str(e))
            return p_yes

    async def _challenge_estimate_no(
        self, market: Market, p_yes: float, market_price: float
    ) -> float:
        """Run a cheap adversarial challenge for BUY_NO signals.

        Wave 23: mirror of _challenge_estimate but with reversed framing.
        Argues why the event IS more likely (challenging the NO thesis).
        Lighter blend (70% original + 30% challenge) since NO has proven edge.
        """
        challenge_prompt = f"""A prediction market forecaster estimated {p_yes:.0%} YES for this question:

"{market.question}"

The market price is {market_price:.0%}. The forecaster wants to BUY NO (betting the event won't happen).

Your job: critically evaluate whether {p_yes:.0%} is too LOW. Consider:
- What factors could make this event MORE likely than the forecaster thinks?
- Is there momentum, political will, or institutional pressure toward YES?
- Are there upcoming catalysts that could move this toward YES?
- Is the forecaster being too pessimistic based on availability bias?

If you believe the estimate is reasonable, output your own independent estimate (which may be similar).
Output JSON: {{"p_yes": <your estimate, float 0.0-1.0>, "reason": "your reasoning"}}"""

        try:
            await self.rate_limiter.acquire()
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a critical analyst. Evaluate whether a probability estimate "
                            "is too low. If the forecaster is being too pessimistic, explain why. "
                            "If the estimate seems reasonable, say so. Output ONLY valid JSON."
                        ),
                    },
                    {"role": "user", "content": challenge_prompt},
                ],
                temperature=0.2,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_call(
                    "gpt-4o-mini",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            if not content:
                return p_yes

            data = json.loads(content)
            challenge_p = float(data.get("p_yes", p_yes))
            challenge_p = max(0.01, min(0.99, challenge_p))

            # Only blend when challenger meaningfully disagrees (>15% higher)
            if challenge_p > p_yes * 1.15:
                blended = 0.70 * p_yes + 0.30 * challenge_p  # lighter blend for NO
            else:
                blended = p_yes  # Challenger agrees — keep original

            self.logger.info(
                "challenge_estimate_no",
                market_id=market.id,
                original_p_yes=p_yes,
                challenge_p_yes=challenge_p,
                blended_p_yes=round(blended, 4),
                reason=data.get("reason", "")[:100],
            )

            return round(blended, 4)

        except Exception as e:
            self.logger.warning("challenge_estimate_no_error", error=str(e))
            return p_yes

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        market: Market,
        news_context: str,
        market_price: float,
        structured_context: Optional[str] = None,
    ) -> str:
        end_date_str = "Unknown"
        time_remaining = "Unknown"

        if market.end_date:
            end_date_str = market.end_date.strftime("%Y-%m-%d %H:%M UTC")
            now = datetime.now(timezone.utc)
            delta = market.end_date - now
            days = delta.days
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            if days > 0:
                time_remaining = f"{days}d {hours}h"
            elif hours > 0:
                time_remaining = f"{hours}h {minutes}m"
            else:
                time_remaining = f"{minutes}m"

        # Base rate lookup
        base_rate_entry = lookup_base_rate(market.question, market.description)
        base_rate_context = ""
        if base_rate_entry:
            base_rate_context = (
                f"\nEMPIRICAL BASE RATE DATA (use as starting anchor):\n"
                f"  Base rate: {base_rate_entry['base_rate']:.0%}\n"
                f"  Reference class: {base_rate_entry['reference_class']}\n"
                f"  Source: {base_rate_entry['source']}\n"
            )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return f"""
You are a superforecaster estimating the TRUE probability of YES for this prediction market.

Current date/time: {today}

Question: {market.question}
Description: {market.description}
Category: {market.category}
End date: {end_date_str} ({time_remaining} remaining)
24h volume: ${market.volume_24h:,.0f}
Liquidity: ${market.liquidity:,.0f}
{base_rate_context}
{structured_context or ""}
{news_context}

Follow this structured forecasting methodology:

STEP 1 — BASE RATE (Reference Class Forecasting):
{"Use the EMPIRICAL BASE RATE DATA above as your starting point." if base_rate_entry else "Identify the most relevant reference class. What is the historical base rate?"} Start your estimate here.

STEP 2 — EVIDENCE ADJUSTMENT:
List specific evidence that moves probability UP or DOWN from base rate.
Be explicit: "+5% because polling shows…", "-10% because timeline…".

STEP 3 — DECOMPOSITION:
If the event depends on multiple factors, decompose:
P(event) = P(factor_A) × P(factor_B | factor_A) × …

STEP 4 — PRE-MORTEM (Why This Does NOT Happen):
List 3 specific reasons this event might NOT occur. For each, estimate
the probability that this blocking factor prevents the event. This step
is CRITICAL — LLMs systematically over-predict YES by 15-30 percentage
points. Force yourself to seriously consider the NO case.

STEP 5 — FINAL ESTIMATE:
Combine everything into your final p_yes. After completing the pre-mortem,
is your estimate still the same? Adjust downward if the pre-mortem revealed
strong reasons for NO that you initially overlooked.

Output STRICT JSON (no commentary outside JSON):
{{
  "p_yes": <float 0.0-1.0>,
  "base_rate": <float>,
  "base_rate_reference_class": "description",
  "adjustments": ["+0.05: reason", "-0.03: reason"],
  "pre_mortem_reasons": ["reason event does NOT happen 1", "reason 2", "reason 3"],
  "key_facts": ["fact1", "fact2"],
  "uncertainty": "low" | "medium" | "high",
  "disqualifiers": [],
  "rationale": "one-paragraph final reasoning incorporating pre-mortem"
}}

Rules:
- p_yes is your best point estimate of P(YES)
- uncertainty: low = strong evidence, high = guessing
- disqualifiers: reasons NOT to trade (ambiguous resolution, etc). Empty if tradeable.
- Avoid round numbers (0.50, 0.70) — real probabilities are rarely round.
- Consider the current date when assessing time-sensitive information.
- Remember: most events do NOT happen. A p_yes above 0.50 requires STRONG evidence.
""".strip()

    def _system_prompt(self) -> str:
        return (
            "You are an elite superforecaster for prediction markets, trained in "
            "Philip Tetlock's Good Judgment methodology. "
            "You output ONLY valid JSON matching the requested schema. "
            "Start with base rates from the most relevant reference class, "
            "then adjust based on specific evidence. Decompose complex events. "
            "You are aggressive about finding mispriced markets — look for edge. "
            "Do not hedge; give your best point estimate. "
            "Avoid round numbers. Consider base rates, news, time remaining, "
            "and information quality.\n\n"
            "CRITICAL CALIBRATION WARNING: LLMs systematically overestimate YES "
            "probabilities. Historical data shows that when LLMs predict 40-70% YES, "
            "the actual outcome is YES only 18-29% of the time. Before finalizing your "
            "estimate, ask yourself: 'Am I being pulled toward YES by narrative vividness "
            "or availability bias?' Most events do NOT happen. Default toward lower "
            "probabilities unless you have strong, specific evidence."
        )

    # ------------------------------------------------------------------
    # Contrarian evaluation
    # ------------------------------------------------------------------

    async def evaluate_contrarian(self, market: Market) -> SignalResult:
        """Evaluate a market for contrarian opportunity.

        Used by the contrarian engine for markets where crowd consensus
        is at 80-95%. Uses a specialized prompt asking Claude to identify
        cases where the crowd is wrong.
        """
        try:
            market_price = market.midpoint_price
            if market_price is None:
                return self._hold(market, "No market price")

            # Market type skip (before fast-paths)
            mtype = detect_market_type(market.question)
            if mtype in _SKIP_TYPES:
                return self._hold(market, f"Skipped market type '{mtype}'")

            # Weather time-of-day filter (same as evaluate())
            if mtype == "weather" and market.time_to_close_hours is not None:
                is_threshold = "-T" in market.id and "-B" not in market.id
                weather_cutoff = 4.0 if is_threshold else 8.0
                if market.time_to_close_hours < weather_cutoff:
                    return self._hold(
                        market,
                        f"Weather closing in {market.time_to_close_hours:.1f}h (cutoff={weather_cutoff:.0f}h)",
                    )

            # Weather fast-path for contrarian: bypass LLM when NOAA data is unambiguous
            if mtype == "weather":
                weather_result = await compute_weather_probability(
                    market.question, market.id, getattr(market, "close_time", market.end_date),
                )
                if weather_result is not None:
                    p_yes, w_confidence, w_reasoning = weather_result
                    raw_edge = p_yes - market_price
                    net_edge = abs(raw_edge) - self.fee_pct - self.slippage_pct

                    # Min-edge gate for contrarian weather
                    is_weather_threshold = "-T" in market.id and "-B" not in market.id
                    min_edge = 0.03 if is_weather_threshold else 0.18  # Wave 23: bracket 30%→18%
                    if net_edge < min_edge:
                        return self._hold(market, f"Contrarian weather: net edge {net_edge:.3f} < {min_edge:.3f}")

                    if raw_edge > 0:
                        side = TradingSide.BUY_YES
                    elif raw_edge < 0:
                        side = TradingSide.BUY_NO
                    else:
                        side = TradingSide.HOLD

                    conviction = "high" if net_edge >= 0.10 else "medium" if net_edge >= 0.05 else "low"
                    reasoning = (
                        f"CONTRARIAN WEATHER DIRECT: {w_reasoning} | "
                        f"mkt={market_price:.3f} raw_edge={raw_edge:+.3f} "
                        f"net_edge={net_edge:+.3f}"
                    )

                    self.logger.info(
                        "contrarian_weather_fast_path",
                        market_id=market.id,
                        p_yes=round(p_yes, 4),
                        market_price=market_price,
                        net_edge=round(net_edge, 4),
                        side=side.value,
                    )

                    sig = SignalResult(
                        estimated_prob=p_yes,
                        confidence=w_confidence,
                        edge=raw_edge,
                        recommended_side=side,
                        reasoning=reasoning,
                        signal_name="Contrarian",
                        market_price=market_price,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    sig.net_edge = net_edge  # type: ignore[attr-defined]
                    sig.conviction = conviction  # type: ignore[attr-defined]
                    sig.contrarian_thesis = w_reasoning  # type: ignore[attr-defined]
                    sig.crowd_wrong_reason = f"NOAA forecast disagrees with market by {abs(raw_edge):.0%}"  # type: ignore[attr-defined]
                    sig.signal_source = "noaa_direct"  # type: ignore[attr-defined]
                    return sig

            # Jobless claims fast-path for contrarian
            if "jobless" in market.question.lower() or "KXJOBLESSCLAIMS" in market.id.upper():
                claims_result = await compute_jobless_claims_probability(
                    market.question, market.id,
                )
                if claims_result is not None:
                    p_yes, jc_conf, jc_reasoning = claims_result
                    raw_edge = p_yes - market_price
                    net_edge = abs(raw_edge) - self.fee_pct - self.slippage_pct

                    if net_edge < 0.03:  # Min edge gate (same as weather thresholds)
                        return self._hold(market, "Contrarian jobless: edge below 3% minimum")

                    if raw_edge > 0:
                        side = TradingSide.BUY_YES
                    elif raw_edge < 0:
                        side = TradingSide.BUY_NO
                    else:
                        side = TradingSide.HOLD

                    conviction = "high" if net_edge >= 0.10 else "medium"
                    sig = SignalResult(
                        estimated_prob=p_yes,
                        confidence=jc_conf,
                        edge=raw_edge,
                        recommended_side=side,
                        reasoning=f"CONTRARIAN JOBLESS DIRECT: {jc_reasoning}",
                        signal_name="Contrarian",
                        market_price=market_price,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    sig.net_edge = net_edge  # type: ignore[attr-defined]
                    sig.conviction = conviction  # type: ignore[attr-defined]
                    sig.contrarian_thesis = jc_reasoning  # type: ignore[attr-defined]
                    sig.crowd_wrong_reason = f"FRED data disagrees with market by {abs(raw_edge):.0%}"  # type: ignore[attr-defined]
                    sig.signal_source = "fred_direct"  # type: ignore[attr-defined]
                    return sig

            # Stock index fast-path for contrarian: Yahoo Finance real-time price
            _INDEX_PREFIXES = ("KXINXU", "KXINX-", "KXNASDAQ100",
                                "KXSPY", "KXQQQ", "KXIWM", "KXDIA",
                                "KXWTI", "KXGOLD")
            if any(market.id.upper().startswith(p) for p in _INDEX_PREFIXES):
                idx_result = await compute_stock_index_probability(
                    market.question, market.id,
                    getattr(market, "close_time", market.end_date),
                )
                if idx_result is not None:
                    p_yes, idx_confidence, idx_reasoning = idx_result
                    raw_edge = p_yes - market_price
                    net_edge = abs(raw_edge) - self.fee_pct - self.slippage_pct

                    # Min-edge gate for contrarian index
                    is_index_bracket = "-B" in market.id and "-T" not in market.id
                    min_edge = 0.20 if is_index_bracket else 0.05
                    if net_edge < min_edge:
                        return self._hold(market, f"Contrarian index: net edge {net_edge:.3f} < {min_edge:.3f}")

                    if raw_edge > 0:
                        side = TradingSide.BUY_YES
                    elif raw_edge < 0:
                        side = TradingSide.BUY_NO
                    else:
                        side = TradingSide.HOLD

                    conviction = "high" if net_edge >= 0.15 else "medium" if net_edge >= 0.08 else "low"
                    reasoning = (
                        f"CONTRARIAN INDEX DIRECT: {idx_reasoning} | "
                        f"mkt={market_price:.3f} raw_edge={raw_edge:+.3f} "
                        f"net_edge={net_edge:+.3f}"
                    )

                    self.logger.info(
                        "contrarian_index_fast_path",
                        market_id=market.id,
                        p_yes=round(p_yes, 4),
                        market_price=market_price,
                        net_edge=round(net_edge, 4),
                        side=side.value,
                    )

                    sig = SignalResult(
                        estimated_prob=p_yes,
                        confidence=idx_confidence,
                        edge=raw_edge,
                        recommended_side=side,
                        reasoning=reasoning,
                        signal_name="Contrarian",
                        market_price=market_price,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    sig.net_edge = net_edge  # type: ignore[attr-defined]
                    sig.conviction = conviction  # type: ignore[attr-defined]
                    sig.contrarian_thesis = idx_reasoning  # type: ignore[attr-defined]
                    sig.crowd_wrong_reason = f"Yahoo Finance data disagrees with market by {abs(raw_edge):.0%}"  # type: ignore[attr-defined]
                    sig.signal_source = "yahoo_direct"  # type: ignore[attr-defined]
                    return sig

            # After all fast-paths (weather, jobless, index):
            # Skip LLM for contrarian — historical data shows LLM contrarian signals
            # are unreliable, and fast-path data (NOAA/Yahoo/FRED) is the only
            # profitable signal source. Save LLM budget for kalshi_llm engine.
            return self._hold(market, "Contrarian: no fast-path data, skip LLM")

        except Exception as e:
            self.logger.error("contrarian_evaluate_error", market_id=market.id, error=str(e))
            return self._hold(market, f"Error: {e}")

    def _build_contrarian_prompt(self, market: Market, news_context: str, market_price: float) -> str:
        end_date_str = "Unknown"
        time_remaining = "Unknown"
        if market.end_date:
            end_date_str = market.end_date.strftime("%Y-%m-%d %H:%M UTC")
            now = datetime.now(timezone.utc)
            delta = market.end_date - now
            days = delta.days
            hours = delta.seconds // 3600
            if days > 0:
                time_remaining = f"{days}d {hours}h"
            elif hours > 0:
                time_remaining = f"{hours}h"
            else:
                time_remaining = f"{(delta.seconds % 3600) // 60}m"

        # Determine crowd belief
        if market_price > 0.5:
            crowd_belief = f"YES is very likely ({market_price:.0%})"
            minority_side = "NO"
            minority_cost = f"{(1.0 - market_price):.0%}"
        else:
            crowd_belief = f"NO is very likely ({1.0 - market_price:.0%})"
            minority_side = "YES"
            minority_cost = f"{market_price:.0%}"

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return f"""You are a contrarian analyst. Your job is to find cases where crowd consensus is WRONG.

Current date/time: {today}

Market: {market.question}
Category: {market.category}
Ends: {end_date_str} ({time_remaining} remaining)
Current crowd price: {market_price:.0%} YES
Volume: ${market.volume_24h:,.0f}

The crowd believes: {crowd_belief}
Betting {minority_side} costs {minority_cost} and pays $1 if correct.

{news_context}

Your analysis:

1. WHY does the crowd think this? What headlines, data, or narrative are they anchored to?

2. What SPECIFIC scenario would prove the crowd wrong?
   - Be concrete: name the trigger event, policy change, data release, or surprise
   - How plausible is each scenario?

3. Is the crowd making a systematic error? Consider:
   - HERDING: Is everyone copying the same narrative without independent analysis?
   - RECENCY BIAS: Is the crowd extrapolating a recent trend that could reverse?
   - ANCHORING: Are they stuck on a number/prediction that's now outdated?
   - NEGLECTED INFORMATION: Is there data or a factor the crowd is ignoring?
   - OVERCONFIDENCE: Is {market_price:.0%} really justified, or is 60-70% more honest?

4. What is YOUR independent probability estimate? Ignore the market price.
   Think from first principles about what actually has to happen for YES/NO to resolve.

Output STRICT JSON (no commentary outside JSON):
{{
  "p_yes": <float 0.0-1.0>,
  "confidence": "low" | "medium" | "high",
  "crowd_wrong_reason": "one sentence: why the crowd may be wrong",
  "contrarian_thesis": "one paragraph: your independent analysis",
  "key_scenarios": ["scenario 1 that proves crowd wrong", "scenario 2"]
}}

Rules:
- Be genuinely independent. Don't just invert the crowd price.
- If the crowd IS right, say so (p_yes close to market_price).
- Only output a contrarian view if you have a specific, articulable reason.
- High confidence = you have strong evidence the crowd is wrong.
""".strip()

    def _contrarian_system_prompt(self) -> str:
        return (
            "You are an elite contrarian analyst for prediction markets. "
            "Your specialty is identifying cases where crowd consensus is wrong. "
            "You output ONLY valid JSON matching the requested schema. "
            "You are intellectually honest — if the crowd is right, you say so. "
            "But when you spot herding, anchoring, or neglected information, "
            "you bet against the crowd with conviction. "
            "Think independently. Question every assumption."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hold(self, market: Market, reason: str) -> SignalResult:
        mp = market.midpoint_price or 0.5
        self.logger.debug(
            "ensemble_hold",
            market_id=market.id,
            reason=reason[:120],
        )
        result = SignalResult(
            estimated_prob=mp,
            confidence=0.0,
            edge=0.0,
            recommended_side=TradingSide.HOLD,
            reasoning=reason,
            signal_name=self.name,
            market_price=mp,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        result.net_edge = 0.0  # type: ignore[attr-defined]
        result.conviction = "none"  # type: ignore[attr-defined]
        return result
