"""A/B comparison framework — tracks which features were active per prediction.

Provides helpers to:
1. Build a features_active dict from current config
2. Break down Brier scores by feature flag (used in accuracy.py)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .utils import BotConfig


def get_active_features(config: BotConfig) -> Dict[str, bool]:
    """Build a dict of feature flags from current config.

    This captures the configuration state at prediction time so
    we can later correlate accuracy with feature combinations.
    """
    llm = config.llm
    news = config.news
    strategy = config.strategy

    return {
        "screening_enabled": bool(llm.get("screening_enabled", True)),
        "consensus_enabled": bool(llm.get("consensus_enabled", False)),
        "cache_enabled": bool(llm.get("cache_enabled", True)),
        "fetch_full_articles": bool(news.get("fetch_full_articles", True)),
        "focus_enabled": bool(config.market_filters.get("focus_enabled", True)),
        "min_conviction": str(strategy.get("min_conviction", "medium")),
    }


def brier_by_feature(
    resolutions: List[Dict[str, Any]],
    *,
    min_samples: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Break down Brier scores by feature flag.

    For each boolean feature, computes Brier score when True vs False.
    Returns dict keyed by feature name with sub-keys:
        true_brier, true_n, false_brier, false_n, delta
    """
    from collections import defaultdict

    # Collect per-feature buckets
    # feature_name -> {True: [brier_scores], False: [brier_scores]}
    buckets: Dict[str, Dict[bool, List[float]]] = defaultdict(lambda: {True: [], False: []})

    for res in resolutions:
        features = res.get("features_active")
        if not features or not isinstance(features, dict):
            continue

        predicted = float(res.get("predicted_p_yes", 0.5))
        actual = float(res.get("actual_outcome", 0.5))
        brier = (predicted - actual) ** 2

        for feat_name, feat_val in features.items():
            # Convert to bool; strings like "medium" go True
            flag = bool(feat_val) if not isinstance(feat_val, str) else True
            buckets[feat_name][flag].append(brier)

    results: Dict[str, Dict[str, Any]] = {}
    for feat_name, by_flag in buckets.items():
        true_scores = by_flag[True]
        false_scores = by_flag[False]

        if len(true_scores) < min_samples and len(false_scores) < min_samples:
            continue

        true_brier = sum(true_scores) / len(true_scores) if true_scores else None
        false_brier = sum(false_scores) / len(false_scores) if false_scores else None

        delta = None
        if true_brier is not None and false_brier is not None:
            delta = true_brier - false_brier  # negative = True is better

        results[feat_name] = {
            "true_brier": round(true_brier, 4) if true_brier is not None else None,
            "true_n": len(true_scores),
            "false_brier": round(false_brier, 4) if false_brier is not None else None,
            "false_n": len(false_scores),
            "delta": round(delta, 4) if delta is not None else None,
        }

    return results
