"""
Cluster detection for multi-strategy signal convergence.

A cluster is *convergence*: at least ``MIN_AGREEING_STRATEGIES`` distinct
vote sources pointing the same way, each at or above ``MIN_CONFIDENCE``.
When that holds the detector boosts confidence (+0.10 per extra agreeing
vote).  When strategies split it flags CONFLICTING.  When only a single
source points somewhere (the pre-2026-09-07 "cluster of one" — 81 % of
the executed era trades) the verdict is HOLD with
``gate_status='rejected_min_agreement'`` so the run stays visible in
signal_events with its vote count, direction and voters.

Both thresholds are configured in ``config/settings.py``
(``CLUSTER_MIN_AGREEING_STRATEGIES``, ``CLUSTER_VOTE_SOURCES``) and
mirrored as class attributes so tests can pin them per instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import (
    CLUSTER_MIN_AGREEING_STRATEGIES,
    CLUSTER_VOTE_SOURCES,
)
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

# gate_status values (also the signal_events.cluster_gate vocabulary)
GATE_NO_VOTES = "no_votes"                     # empty input
GATE_NO_DIRECTIONAL = "no_directional"         # every confident vote is HOLD
GATE_CONFLICTING = "conflicting"               # confident votes on both sides
GATE_REJECTED = "rejected_min_agreement"       # one side, too few sources
GATE_PASSED = "passed"                         # one side, enough sources


@dataclass
class ClusterResult:
    """Output of cluster detection across multiple strategy results."""

    cluster_signal: str
    confidence: float
    agreeing_strategies: list[str] = field(default_factory=list)
    disagreeing_strategies: list[str] = field(default_factory=list)
    cluster_strength: int = 0
    # B3 attribution carry-along (never feeds back into any decision):
    # which agreeing strategy won the max()-by-rank pick, and how much
    # boost was actually applied on top of its base confidence.  None on
    # the HOLD / CONFLICTING exits — there is no strongest supplier there.
    strongest_supplier: str | None = None
    boost_applied: float | None = None
    # Agreement gate (2026-09-07).  gate_status is one of the GATE_*
    # constants; vote_direction is "BUY"/"SELL" whenever confident votes
    # exist on exactly one side (passed AND rejected), else None;
    # distinct_sources is the number of vote sources behind
    # agreeing_strategies after CLUSTER_VOTE_SOURCES folding.
    gate_status: str = GATE_NO_VOTES
    vote_direction: str | None = None
    distinct_sources: int = 0


class ClusterDetector:
    """Detect convergence across multiple strategy signals."""

    # Minimum confidence for a strategy result to count toward cluster
    MIN_CONFIDENCE = 0.35

    # Minimum number of distinct vote sources that must point the same
    # way before the cluster is allowed to be directional.  Single source
    # of truth: config.settings.CLUSTER_MIN_AGREEING_STRATEGIES.
    MIN_AGREEING_STRATEGIES = CLUSTER_MIN_AGREEING_STRATEGIES

    # strategy_name → vote source; unmapped names are their own source.
    VOTE_SOURCES: dict[str, str] = CLUSTER_VOTE_SOURCES

    def _source_of(self, strategy_name: str) -> str:
        return self.VOTE_SOURCES.get(strategy_name, strategy_name)

    @staticmethod
    def _hold_confidence(strategy_results: list[StrategyResult]) -> float:
        """HOLD confidence from the *unfiltered* HOLD votes.

        MIN_CONFIDENCE is a noise filter for directional (BUY/SELL)
        votes — but for HOLDs, the sub-threshold values ARE the data
        (a 0/4 Momentum HOLD at 10% is real "no setup", not noise).
        Filtering on confident here was the Phase A regression that
        left Combined HOLD pinned at the 0.25 fallback.

        Mirrors the HOLD-vs-direction convention in coordinator.py
        (non_hold * 0.8) by taking max * 0.8 — preserves per-event
        variance instead of compressing every HOLD to a constant.
        """
        hold_confidences = [
            r.confidence / 100.0
            for r in strategy_results
            if r.signal.upper() == "HOLD"
        ]
        if hold_confidences:
            return max(hold_confidences) * 0.8
        return 0.25

    def detect(self, strategy_results: list[StrategyResult]) -> ClusterResult:
        """Analyse a list of strategy results for signal convergence."""
        if not strategy_results:
            return ClusterResult(
                cluster_signal="HOLD",
                confidence=0.25,
                cluster_strength=0,
                gate_status=GATE_NO_VOTES,
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
            return ClusterResult(
                cluster_signal="HOLD",
                confidence=self._hold_confidence(strategy_results),
                cluster_strength=0,
                disagreeing_strategies=[],
                agreeing_strategies=[r.strategy_name for r in confident],
                gate_status=GATE_NO_DIRECTIONAL,
            )

        # Strategies split between buy and sell
        if buy_side and sell_side:
            return ClusterResult(
                cluster_signal="CONFLICTING",
                confidence=0.10,
                agreeing_strategies=[],
                disagreeing_strategies=buy_names + sell_names,
                cluster_strength=0,
                gate_status=GATE_CONFLICTING,
            )

        # All directional strategies agree on one side
        if buy_side:
            winners = buy_side
            agreeing = buy_names
            direction = "BUY"
        else:
            winners = sell_side
            agreeing = sell_names
            direction = "SELL"
        disagreeing: list[str] = []

        # ── Agreement gate ────────────────────────────────────────────
        # Count distinct vote *sources*, not vote rows: two strategies
        # mapped to one source in VOTE_SOURCES are one voice.
        sources = {self._source_of(name) for name in agreeing}
        if len(sources) < self.MIN_AGREEING_STRATEGIES:
            log.debug(
                "Cluster-gate: rejected direction=%s votes=%d sources=%d "
                "min=%d voters=%s",
                direction, len(winners), len(sources),
                self.MIN_AGREEING_STRATEGIES, agreeing,
            )
            return ClusterResult(
                cluster_signal="HOLD",
                confidence=self._hold_confidence(strategy_results),
                agreeing_strategies=agreeing,
                disagreeing_strategies=disagreeing,
                cluster_strength=len(winners),
                gate_status=GATE_REJECTED,
                vote_direction=direction,
                distinct_sources=len(sources),
            )

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
            strongest_supplier=strongest.strategy_name,
            boost_applied=round(boosted_conf - base_conf, 2),
            gate_status=GATE_PASSED,
            vote_direction=direction,
            distinct_sources=len(sources),
        )
