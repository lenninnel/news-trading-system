"""
Cluster detection for multi-strategy signal convergence.

When multiple independent strategies agree on direction, the cluster
detector boosts confidence. When they disagree, it flags CONFLICTING.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from strategies.base import StrategyResult

log = logging.getLogger(__name__)

# Signal direction buckets
_BUY_SIGNALS = {"STRONG BUY", "BUY", "WEAK BUY"}
_SELL_SIGNALS = {"SELL", "STRONG SELL", "WEAK SELL"}

# Strength ordering for picking the "strongest" in a cluster
_STRENGTH_ORDER = {
    "STRONG BUY": 6, "BUY": 5, "WEAK BUY": 4,
    "HOLD": 3,
    "WEAK SELL": 2, "SELL": 1, "STRONG SELL": 0,
    "CONFLICTING": -1,
}


@dataclass
class ClusterResult:
    """Output of cluster detection across multiple strategy results."""

    cluster_signal: str
    confidence: float
    agreeing_strategies: list[str] = field(default_factory=list)
    disagreeing_strategies: list[str] = field(default_factory=list)
    cluster_strength: int = 0


class ClusterDetector:
    """Detect convergence across multiple strategy signals."""

    # Minimum confidence for a strategy result to count toward cluster
    MIN_CONFIDENCE = 0.35

    def detect(self, strategy_results: list[StrategyResult]) -> ClusterResult:
        """Analyse a list of strategy results for signal convergence."""
        if not strategy_results:
            return ClusterResult(
                cluster_signal="HOLD",
                confidence=0.25,
                cluster_strength=0,
            )

        # Filter to confident-enough results (confidence is 0-100 scale)
        confident = [
            r for r in strategy_results
            if r.confidence / 100.0 >= self.MIN_CONFIDENCE
        ]

        # Bucket into directions
        buy_side = [r for r in confident if r.signal.upper() in _BUY_SIGNALS]
        sell_side = [r for r in confident if r.signal.upper() in _SELL_SIGNALS]

        buy_names = [r.strategy_name for r in buy_side]
        sell_names = [r.strategy_name for r in sell_side]

        # No directional signals at all
        if not buy_side and not sell_side:
            hold_confidences = [
                r.confidence / 100.0
                for r in confident
                if r.signal.upper() == "HOLD"
            ]
            if hold_confidences:
                # Mirror the HOLD-vs-direction convention in coordinator.py
                # (non_hold * 0.8) by taking max * 0.8 — preserves per-event
                # variance instead of compressing every HOLD to a constant.
                hold_conf = max(hold_confidences) * 0.8
            else:
                hold_conf = 0.25
            return ClusterResult(
                cluster_signal="HOLD",
                confidence=hold_conf,
                cluster_strength=0,
                disagreeing_strategies=[],
                agreeing_strategies=[r.strategy_name for r in confident],
            )

        # Strategies split between buy and sell
        if buy_side and sell_side:
            return ClusterResult(
                cluster_signal="CONFLICTING",
                confidence=0.10,
                agreeing_strategies=[],
                disagreeing_strategies=buy_names + sell_names,
                cluster_strength=0,
            )

        # All directional strategies agree on one side
        if buy_side:
            winners = buy_side
            agreeing = buy_names
            disagreeing: list[str] = []
        else:
            winners = sell_side
            agreeing = sell_names
            disagreeing = []

        # Pick the strongest signal from the agreeing strategies
        strongest = max(winners, key=lambda r: _STRENGTH_ORDER.get(r.signal.upper(), 0))
        base_conf = strongest.confidence / 100.0

        # Boost: +0.10 per additional agreeing strategy (cap at 1.0)
        extra = len(winners) - 1
        boosted_conf = min(1.0, base_conf + extra * 0.10)

        log.info(
            "Cluster: %d strategies agree on %s (conf %.2f → %.2f, +%d boost)",
            len(winners), strongest.signal, base_conf, boosted_conf, extra,
        )

        return ClusterResult(
            cluster_signal=strongest.signal,
            confidence=boosted_conf,
            agreeing_strategies=agreeing,
            disagreeing_strategies=disagreeing,
            cluster_strength=len(winners),
        )
