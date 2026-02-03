"""LLM-based dependency detection between prediction markets.

Uses GPT-4o-mini (or compatible) to analyze pairs of Polymarket markets
and determine if they are logically dependent. Dependencies create
constraints that reduce the valid outcome space, enabling cross-market
arbitrage detection.

Approach (adapted from arXiv:2508.03474v1, Section 5):
    1. For each pair of markets, send descriptions to the LLM
    2. Ask it to enumerate valid outcome combinations
    3. Convert valid combos into linear constraints on the polytope

Example:
    Market A: "Will Trump win Pennsylvania?"  [Yes, No]
    Market B: "Will Republicans win PA by 5+?" [Yes, No]

    LLM determines:
        Valid: (A=Yes, B=Yes), (A=Yes, B=No), (A=No, B=No)
        Invalid: (A=No, B=Yes)  ← can't win by 5+ without winning

    Constraint: p(B=Yes) ≤ p(A=Yes)

References
----------
- arXiv:2508.03474v1, Section 5.2 (DeepSeek dependency detection)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)

# Attempt to import openai; graceful fallback if not installed
try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


@dataclass
class MarketDependency:
    """A detected dependency between two markets.

    Attributes
    ----------
    market_a_id : str
        First market identifier.
    market_b_id : str
        Second market identifier.
    constraint_type : str
        Type of constraint: "implication", "mutual_exclusion",
        "identical", "subset", "custom".
    valid_combos : list of tuple
        Valid (outcome_a_idx, outcome_b_idx) pairs.
    invalid_combos : list of tuple
        Invalid (outcome_a_idx, outcome_b_idx) pairs.
    confidence : float
        LLM confidence in the dependency (0-1).
    explanation : str
        Human-readable explanation.
    linear_constraints : list of dict
        Each dict has "indices", "coefficients", "rhs", "type" ("ub"/"eq").
    """

    market_a_id: str
    market_b_id: str
    constraint_type: str
    valid_combos: List[Tuple[int, int]]
    invalid_combos: List[Tuple[int, int]] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    linear_constraints: List[Dict] = field(default_factory=list)


# -------------------------------------------------------------------
# Prompt template (adapted from the DeepSeek approach in the paper)
# -------------------------------------------------------------------

_DEPENDENCY_PROMPT = """You are analyzing two prediction markets to determine if they are logically dependent.

Market A: "{question_a}"
  Outcomes: {outcomes_a}
  Description: {desc_a}

Market B: "{question_b}"
  Outcomes: {outcomes_b}
  Description: {desc_b}

Analyze whether the outcomes of these markets are logically linked. Consider:
1. Does one outcome IMPLY another? (e.g., "wins by 5+" implies "wins")
2. Are they MUTUALLY EXCLUSIVE? (e.g., both can't happen)
3. Are they about the SAME underlying event?
4. Could they be INDEPENDENT? (no logical connection)

Respond in JSON format:
{{
    "dependent": true/false,
    "constraint_type": "implication|mutual_exclusion|identical|subset|independent",
    "valid_combinations": [
        {{"a_outcome": "outcome_name", "a_index": 0, "b_outcome": "outcome_name", "b_index": 0}},
        ...
    ],
    "invalid_combinations": [
        {{"a_outcome": "outcome_name", "a_index": 0, "b_outcome": "outcome_name", "b_index": 0, "reason": "..."}},
        ...
    ],
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of the dependency"
}}

List ALL valid and invalid outcome combinations. Be precise about indices.
Only mark as dependent if there is a CLEAR logical relationship.
"""


class DependencyDetector:
    """Detect logical dependencies between prediction markets using an LLM.

    Parameters
    ----------
    model : str
        OpenAI model to use. Default: "gpt-4o-mini" for cost efficiency.
    api_key : optional str
        OpenAI API key. Falls back to OPENAI_API_KEY env var.
    max_concurrent : int
        Maximum concurrent LLM calls.
    cache : bool
        Whether to cache results (keyed by market question hashes).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_concurrent: int = 5,
        cache: bool = True,
    ):
        self.model = model
        self.max_concurrent = max_concurrent
        self.cache = cache
        self._cache: Dict[str, MarketDependency] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

        self._api_key = api_key
        self._client = None  # Lazy init to avoid requiring API key at import time

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is not None:
            return self._client
        if not _HAS_OPENAI:
            return None
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            logger.warning("No OpenAI API key available for dependency detection")
            return None
        self._client = openai.AsyncOpenAI(api_key=key)
        return self._client

    def _cache_key(self, q_a: str, q_b: str) -> str:
        """Deterministic cache key for a market pair (order-independent)."""
        pair = tuple(sorted([q_a, q_b]))
        return hashlib.md5(json.dumps(pair).encode()).hexdigest()

    async def detect_dependencies(
        self,
        markets: List[Dict[str, Any]],
    ) -> List[MarketDependency]:
        """Detect dependencies between all pairs of markets.

        Parameters
        ----------
        markets : list of dict
            Each dict should have:
                - "id": str
                - "question": str
                - "outcomes": list of str (e.g., ["Yes", "No"])
                - "description": str (optional)

        Returns
        -------
        list of MarketDependency
            Only non-independent dependencies are returned.
        """
        client = self._get_client()
        if client is None:
            logger.warning("openai not available, skipping dependency detection")
            return []

        # Generate all unique pairs
        pairs = []
        for i in range(len(markets)):
            for j in range(i + 1, len(markets)):
                pairs.append((markets[i], markets[j]))

        logger.info("detecting_dependencies", n_markets=len(markets), n_pairs=len(pairs))

        # Run LLM calls concurrently with rate limiting
        tasks = [self._detect_pair(a, b) for a, b in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        dependencies = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("dependency_detection_failed", error=str(result))
                continue
            if result is not None:
                dependencies.append(result)

        logger.info("dependencies_detected", count=len(dependencies))
        return dependencies

    async def _detect_pair(
        self,
        market_a: Dict[str, Any],
        market_b: Dict[str, Any],
    ) -> Optional[MarketDependency]:
        """Detect dependency between a single pair of markets."""
        q_a = market_a.get("question", "")
        q_b = market_b.get("question", "")

        # Check cache
        cache_key = self._cache_key(q_a, q_b)
        if self.cache and cache_key in self._cache:
            return self._cache[cache_key]

        prompt = _DEPENDENCY_PROMPT.format(
            question_a=q_a,
            outcomes_a=market_a.get("outcomes", ["Yes", "No"]),
            desc_a=market_a.get("description", "N/A")[:500],
            question_b=q_b,
            outcomes_b=market_b.get("outcomes", ["Yes", "No"]),
            desc_b=market_b.get("description", "N/A")[:500],
        )

        client = self._get_client()
        if client is None:
            return None

        async with self._semaphore:
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise logical analyst for prediction markets. Respond only in valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                data = json.loads(content)

            except Exception as e:
                logger.error(
                    "llm_call_failed",
                    market_a=market_a["id"],
                    market_b=market_b["id"],
                    error=str(e),
                )
                return None

        if not data.get("dependent", False):
            return None

        # Parse valid/invalid combinations
        valid_combos = [
            (c["a_index"], c["b_index"])
            for c in data.get("valid_combinations", [])
        ]
        invalid_combos = [
            (c["a_index"], c["b_index"])
            for c in data.get("invalid_combinations", [])
        ]

        # Convert to linear constraints
        n_a = len(market_a.get("outcomes", ["Yes", "No"]))
        n_b = len(market_b.get("outcomes", ["Yes", "No"]))
        linear_constraints = self._combos_to_constraints(
            valid_combos, invalid_combos, n_a, n_b,
            market_a["id"], market_b["id"],
        )

        dep = MarketDependency(
            market_a_id=market_a["id"],
            market_b_id=market_b["id"],
            constraint_type=data.get("constraint_type", "custom"),
            valid_combos=valid_combos,
            invalid_combos=invalid_combos,
            confidence=data.get("confidence", 0.5),
            explanation=data.get("explanation", ""),
            linear_constraints=linear_constraints,
        )

        if self.cache:
            self._cache[cache_key] = dep

        return dep

    @staticmethod
    def _combos_to_constraints(
        valid_combos: List[Tuple[int, int]],
        invalid_combos: List[Tuple[int, int]],
        n_a: int,
        n_b: int,
        market_a_id: str,
        market_b_id: str,
    ) -> List[Dict]:
        """Convert valid/invalid outcome combinations to linear constraints.

        For each invalid combination (i, j), we need:
            p(A=i, B=j) = 0

        In the marginal representation (no joint distribution), we use
        the implication constraint approach:

        If (A=i) → NOT (B=j), then:
            p(B=j) ≤ 1 - p(A=i)
            equivalently: p(A=i) + p(B=j) ≤ 1

        For implication (A=i) → (B=j):
            p(A=i) ≤ p(B=j)
            equivalently: p(A=i) - p(B=j) ≤ 0

        Returns list of constraint dicts compatible with DependencyConstraint.
        """
        constraints = []

        # For each invalid combination, add constraint: p(A=i) + p(B=j) ≤ 1
        for a_idx, b_idx in invalid_combos:
            # In the combined variable vector, market A occupies indices 0..n_a-1
            # and market B occupies n_a..n_a+n_b-1.
            # (Actual global indices will be assigned when building the polytope.)
            constraints.append({
                "market_a_id": market_a_id,
                "market_b_id": market_b_id,
                "a_outcome_idx": a_idx,
                "b_outcome_idx": b_idx,
                "type": "incompatible",
                # p(A=a_idx) + p(B=b_idx) ≤ 1
                "description": f"p(A={a_idx}) + p(B={b_idx}) <= 1",
            })

        # Detect implication patterns
        # If all valid combos with A=i also have B=j, then A=i → B=j
        for a_val in range(n_a):
            b_vals_when_a = set()
            for va, vb in valid_combos:
                if va == a_val:
                    b_vals_when_a.add(vb)

            if len(b_vals_when_a) == 1:
                b_implied = list(b_vals_when_a)[0]
                # A=a_val → B=b_implied, so p(A=a_val) ≤ p(B=b_implied)
                constraints.append({
                    "market_a_id": market_a_id,
                    "market_b_id": market_b_id,
                    "a_outcome_idx": a_val,
                    "b_outcome_idx": b_implied,
                    "type": "implication",
                    "description": f"p(A={a_val}) <= p(B={b_implied})",
                })

        return constraints
