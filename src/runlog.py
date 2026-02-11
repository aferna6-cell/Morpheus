"""Run logging helpers.

Creates a per-run directory under ./runs (or configurable base) and writes:
- market snapshots (markets_0001.json)
- decision log as JSONL (decisions.jsonl)

Also provides prediction logging for post-trade accuracy tracking.
Predictions go into state_dir/predictions.jsonl.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .utils import append_jsonl, ensure_dir, save_json_state


def _default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


class RunRecorder:
    def __init__(self, *, runs_dir: str = "runs", run_id: Optional[str] = None):
        self.runs_dir = Path(runs_dir)
        ensure_dir(self.runs_dir)

        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

        self.run_id = run_id
        self.run_path = self.runs_dir / self.run_id
        ensure_dir(self.run_path)

        self.decisions_path = self.run_path / "decisions.jsonl"
        self.meta_path = self.run_path / "run_meta.json"

        save_json_state(
            {
                "run_id": self.run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            self.meta_path,
        )

    def write_market_snapshot(self, *, iteration: int, markets: Iterable[Any]) -> Path:
        p = self.run_path / f"markets_{iteration:04d}.json"
        payload = {
            "iteration": iteration,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "markets": list(markets),
        }
        with open(p, "w") as f:
            json.dump(payload, f, indent=2, default=_default)
        return p

    def log_decision(self, event: Dict[str, Any]) -> None:
        append_jsonl(self.decisions_path, event)


# ---------------------------------------------------------------------------
# Prediction log for accuracy tracking
# ---------------------------------------------------------------------------

def log_prediction(
    state_dir: str,
    *,
    market_id: str,
    predicted_p_yes: float,
    market_price_at_entry: float,
    side: str,
    edge: float,
    conviction: str,
    net_edge: float = 0.0,
    signal_source: str = "llm",
    features_active: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a prediction entry to state_dir/predictions.jsonl.

    Used for post-hoc accuracy evaluation — compare predicted_p_yes to
    eventual market resolution.

    Args:
        signal_source: "noaa_direct" for weather fast-path, "llm" for LLM ensemble.
        features_active: Optional dict of feature flags active at prediction
            time (for A/B comparison). See ab_tracker.get_active_features().
    """
    record = {
        "market_id": market_id,
        "predicted_p_yes": round(predicted_p_yes, 4),
        "market_price_at_entry": round(market_price_at_entry, 4),
        "side": side,
        "edge": round(edge, 4),
        "net_edge": round(net_edge, 4),
        "conviction": conviction,
        "signal_source": signal_source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if features_active is not None:
        record["features_active"] = features_active
    predictions_path = Path(state_dir) / "predictions.jsonl"
    append_jsonl(predictions_path, record)
