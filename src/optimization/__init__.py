"""Mathematical optimization engine for Polymarket arbitrage detection.

Implements algorithms from:
    "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
    arXiv:2508.03474v1

Core components:
    - Bregman divergence & KL projection for LMSR cost functions
    - Barrier Frank-Wolfe solver over marginal polytopes
    - Marginal polytope construction with dependency constraints
    - LLM-based market dependency detection
    - Full arbitrage scanner tying everything together
"""

from .bregman import (
    bregman_divergence,
    kl_divergence,
    project_onto_polytope,
    project_onto_simplex,
)
from .dependency import DependencyDetector, MarketDependency
from .frank_wolfe import FrankWolfe, FrankWolfeResult
from .polytope import (
    Polytope,
    build_multi_market_polytope,
    build_single_market_polytope,
    calculate_arbitrage_profit,
    detect_arbitrage,
    enumerate_valid_outcomes,
)
__all__ = [
    "kl_divergence",
    "bregman_divergence",
    "project_onto_simplex",
    "project_onto_polytope",
    "FrankWolfe",
    "FrankWolfeResult",
    "Polytope",
    "build_single_market_polytope",
    "build_multi_market_polytope",
    "enumerate_valid_outcomes",
    "detect_arbitrage",
    "calculate_arbitrage_profit",
    "DependencyDetector",
    "MarketDependency",
]
