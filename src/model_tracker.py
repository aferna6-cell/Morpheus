"""Per-model prediction tracking for adaptive ensemble weighting.

Logs each model's p_yes separately. After 30+ resolved markets per model,
computes inverse-Brier-score weights for the ensemble instead of naive averaging.

State file: state/model_predictions.jsonl
Format per line:
{
  "market_id": "TICKER",
  "timestamp": "...",
  "models": {"gpt-4o": 0.35, "claude-sonnet-...": 0.42},
  "ensemble_p_yes": 0.385,
  "market_price": 0.50,
  "side": "buy_no"
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from .utils import append_jsonl

logger = structlog.get_logger()

# Minimum resolved markets per model before using learned weights
_MIN_SAMPLES_FOR_WEIGHTS = 20

# State directory
_STATE_DIR = Path("state")


def log_model_predictions(
    market_id: str,
    model_predictions: Dict[str, float],
    ensemble_p_yes: float,
    market_price: float,
    side: str,
) -> None:
    """Log per-model predictions for a market.

    Args:
        market_id: The market ticker
        model_predictions: Dict of {model_name: p_yes} for each model
        ensemble_p_yes: The final ensemble probability
        market_price: Current market price at time of prediction
        side: The trading side chosen (buy_yes, buy_no, hold)
    """
    record = {
        "market_id": market_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": model_predictions,
        "ensemble_p_yes": round(ensemble_p_yes, 4),
        "market_price": round(market_price, 4),
        "side": side,
    }
    try:
        append_jsonl(_STATE_DIR / "model_predictions.jsonl", record)
    except Exception as e:
        logger.warning("model_prediction_log_failed", error=str(e))


def compute_model_weights(state_dir: str | Path = "state") -> Dict[str, float]:
    """Compute per-model weights from resolved predictions.

    Returns a dict of {model_name: weight} where weights sum to 1.0.
    Uses inverse Brier score: models with lower Brier scores get higher weight.
    Falls back to equal weights if insufficient data.
    """
    state_path = Path(state_dir)

    # Load model predictions
    predictions = _read_jsonl(state_path / "model_predictions.jsonl")
    if not predictions:
        return {}

    # Load resolutions
    resolutions = _read_jsonl(state_path / "resolutions.jsonl")
    if not resolutions:
        return {}

    # Build resolution lookup: market_id -> actual_outcome
    resolved: Dict[str, float] = {}
    for r in resolutions:
        mid = r.get("market_id")
        outcome = r.get("actual_outcome")
        if mid is not None and outcome is not None:
            resolved[mid] = float(outcome)

    # Compute per-model Brier scores
    model_scores: Dict[str, List[float]] = {}  # model -> list of (p - outcome)^2

    for pred in predictions:
        mid = pred.get("market_id")
        if mid not in resolved:
            continue

        actual = resolved[mid]
        models = pred.get("models", {})

        for model_name, p_yes in models.items():
            if model_name not in model_scores:
                model_scores[model_name] = []
            brier = (float(p_yes) - actual) ** 2
            model_scores[model_name].append(brier)

    # Check if we have enough samples
    weights: Dict[str, float] = {}
    all_have_enough = all(
        len(scores) >= _MIN_SAMPLES_FOR_WEIGHTS
        for scores in model_scores.values()
    )

    if not all_have_enough or not model_scores:
        # Equal weights fallback
        if model_scores:
            n = len(model_scores)
            for model in model_scores:
                weights[model] = 1.0 / n
            logger.info(
                "model_weights_equal",
                reason="insufficient_samples",
                sample_counts={k: len(v) for k, v in model_scores.items()},
            )
        return weights

    # Compute inverse Brier score weights
    # Lower Brier = better = higher weight
    avg_briers: Dict[str, float] = {}
    for model, scores in model_scores.items():
        avg_brier = sum(scores) / len(scores)
        avg_briers[model] = avg_brier

    # Inverse: weight = 1/brier, then normalize
    # Add small epsilon to avoid division by zero
    # Penalty: models with Brier > 0.25 (worse than random) get heavily penalized
    inverse_briers = {}
    for model, brier in avg_briers.items():
        if brier > 0.25:
            # Worse than random — squash weight to near-zero
            inverse_briers[model] = 0.5  # small fixed weight
        else:
            inverse_briers[model] = 1.0 / (brier + 0.001)
    total = sum(inverse_briers.values())

    for model, inv_brier in inverse_briers.items():
        weights[model] = round(inv_brier / total, 4)

    logger.info(
        "model_weights_computed",
        weights=weights,
        avg_briers={k: round(v, 4) for k, v in avg_briers.items()},
        sample_counts={k: len(v) for k, v in model_scores.items()},
    )

    return weights


def trimmed_mean(
    model_predictions: Dict[str, float],
    trim_fraction: float = 0.2,
) -> float:
    """Trimmed mean: remove extreme predictions, average the rest.

    With 5 models and trim_fraction=0.2, trims 1 from each end (the highest
    and lowest predictions), then averages the middle 3. This is robust to
    a single badly-calibrated model pulling the ensemble off.

    Falls back to simple average when <=2 models (can't trim).
    """
    if len(model_predictions) <= 2:
        return sum(model_predictions.values()) / len(model_predictions)

    values = sorted(model_predictions.values())
    n = len(values)
    trim_count = max(1, int(n * trim_fraction))
    trimmed = values[trim_count:n - trim_count]

    if not trimmed:
        # Over-trimmed (shouldn't happen with trim_fraction=0.2)
        trimmed = values

    return sum(trimmed) / len(trimmed)


def weighted_average(
    model_predictions: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Compute weighted average of model predictions.

    Falls back to simple average if weights are empty or don't cover all models.
    """
    if not model_predictions:
        return 0.5

    if not weights or not all(m in weights for m in model_predictions):
        # Simple average fallback
        return sum(model_predictions.values()) / len(model_predictions)

    # Weighted average
    total_weight = 0.0
    weighted_sum = 0.0
    for model, p_yes in model_predictions.items():
        w = weights.get(model, 0.0)
        weighted_sum += w * p_yes
        total_weight += w

    if total_weight <= 0:
        return sum(model_predictions.values()) / len(model_predictions)

    return weighted_sum / total_weight


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file."""
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
