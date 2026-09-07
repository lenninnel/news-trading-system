"""Tests for the cluster detection module."""

import pytest

from strategies.base import StrategyResult
from orchestrator.cluster_detector import ClusterDetector, ClusterResult


def _make_result(signal: str, confidence: float, name: str = "TestStrategy") -> StrategyResult:
    """Helper to create a StrategyResult for testing."""
    return StrategyResult(
        signal=signal,
        confidence=confidence,
        strategy_name=name,
    )


class TestClusterDetector:
    """Unit tests for ClusterDetector.detect()."""

    def setup_method(self):
        self.detector = ClusterDetector()

    def test_all_agree_buy(self):
        results = [
            _make_result("STRONG BUY", 75, "Momentum"),
            _make_result("BUY", 60, "Pullback"),
            _make_result("WEAK BUY", 45, "NewsCatalyst"),
        ]
        cluster = self.detector.detect(results)
        assert cluster.cluster_signal == "STRONG BUY"
        assert cluster.confidence > 0.75  # boosted
        assert cluster.cluster_strength == 3
        assert len(cluster.agreeing_strategies) == 3
        assert len(cluster.disagreeing_strategies) == 0

    def test_strategies_split(self):
        results = [
            _make_result("BUY", 60, "Momentum"),
            _make_result("SELL", 55, "Pullback"),
        ]
        cluster = self.detector.detect(results)
        assert cluster.cluster_signal == "CONFLICTING"
        assert cluster.confidence == 0.10
        assert cluster.cluster_strength == 0
        assert cluster.gate_status == "conflicting"

    def test_no_directional_signals(self):
        # Phase C: HOLD votes are aggregated from the unfiltered list, so
        # sub-MIN_CONFIDENCE HOLDs (25%) now drive the cluster confidence
        # instead of being discarded. max(25, 25)/100 * 0.8 = 0.20.
        results = [
            _make_result("HOLD", 25, "Momentum"),
            _make_result("HOLD", 25, "Pullback"),
        ]
        cluster = self.detector.detect(results)
        assert cluster.cluster_signal == "HOLD"
        assert cluster.confidence == 0.20
        assert cluster.gate_status == "no_directional"
        assert cluster.vote_direction is None

    def test_single_strategy_rejected_by_agreement_gate(self):
        """A solo directional vote is not a cluster (gate, 2026-09-07)."""
        results = [_make_result("BUY", 70, "Momentum")]
        cluster = self.detector.detect(results)
        assert cluster.cluster_signal == "HOLD"
        assert cluster.gate_status == "rejected_min_agreement"
        assert cluster.vote_direction == "BUY"
        assert cluster.cluster_strength == 1
        assert cluster.distinct_sources == 1
        assert cluster.agreeing_strategies == ["Momentum"]
        # no HOLD votes to derive a confidence from → fallback 0.25
        assert cluster.confidence == 0.25
        assert cluster.strongest_supplier is None
        assert cluster.boost_applied is None

    def test_single_strategy_no_boost_when_threshold_is_one(self):
        """MIN_AGREEING_STRATEGIES=1 restores the pre-gate behaviour."""
        self.detector.MIN_AGREEING_STRATEGIES = 1
        results = [_make_result("BUY", 70, "Momentum")]
        cluster = self.detector.detect(results)
        assert cluster.cluster_signal == "BUY"
        assert cluster.confidence == 0.70  # no boost
        assert cluster.cluster_strength == 1
        assert cluster.gate_status == "passed"

    def test_low_confidence_filtered(self):
        """Signals below MIN_CONFIDENCE (0.35 = 35%) don't count toward cluster
        — and therefore don't count toward the agreement gate either."""
        results = [
            _make_result("BUY", 60, "Momentum"),
            _make_result("BUY", 20, "Pullback"),  # too low, filtered
        ]
        cluster = self.detector.detect(results)
        assert cluster.cluster_signal == "HOLD"
        assert cluster.gate_status == "rejected_min_agreement"
        assert cluster.cluster_strength == 1  # only Momentum counted
        assert cluster.agreeing_strategies == ["Momentum"]

    def test_low_confidence_filtered_threshold_one(self):
        self.detector.MIN_AGREEING_STRATEGIES = 1
        results = [
            _make_result("BUY", 60, "Momentum"),
            _make_result("BUY", 20, "Pullback"),  # too low, filtered
        ]
        cluster = self.detector.detect(results)
        assert cluster.cluster_signal == "BUY"
        assert cluster.cluster_strength == 1  # only Momentum counted
        assert cluster.confidence == 0.60  # no boost

    def test_empty_results(self):
        cluster = self.detector.detect([])
        assert cluster.cluster_signal == "HOLD"
        assert cluster.confidence == 0.25
        assert cluster.gate_status == "no_votes"

    def test_confidence_capped_at_one(self):
        """Even with many strategies, confidence cannot exceed 1.0."""
        results = [
            _make_result("STRONG BUY", 95, f"Strategy{i}")
            for i in range(5)
        ]
        cluster = self.detector.detect(results)
        assert cluster.confidence <= 1.0

    def test_two_buy_one_hold(self):
        """HOLD strategies don't count as disagreeing -- only directional ones do."""
        results = [
            _make_result("BUY", 60, "Momentum"),
            _make_result("WEAK BUY", 45, "Pullback"),
            _make_result("HOLD", 25, "NewsCatalyst"),
        ]
        cluster = self.detector.detect(results)
        assert cluster.cluster_signal in ("BUY", "WEAK BUY")
        assert cluster.cluster_strength == 2
        assert "CONFLICTING" not in cluster.cluster_signal
        assert cluster.gate_status == "passed"
        assert cluster.vote_direction == "BUY"
        assert cluster.distinct_sources == 2


class TestAgreementGate:
    """Minimum-agreement gate (2026-09-07): a cluster is convergence."""

    def setup_method(self):
        self.detector = ClusterDetector()

    def test_threshold_comes_from_settings(self):
        from config.settings import (
            CLUSTER_MIN_AGREEING_STRATEGIES,
            CLUSTER_VOTE_SOURCES,
        )
        assert ClusterDetector.MIN_AGREEING_STRATEGIES == CLUSTER_MIN_AGREEING_STRATEGIES
        assert ClusterDetector.MIN_AGREEING_STRATEGIES == 2
        assert ClusterDetector.VOTE_SOURCES is CLUSTER_VOTE_SOURCES
        # default: the three production strategies are distinct sources
        assert len(set(CLUSTER_VOTE_SOURCES.values())) == 3

    def test_two_agreeing_pass(self):
        cluster = self.detector.detect([
            _make_result("BUY", 60, "Momentum"),
            _make_result("WEAK BUY", 45, "Pullback"),
            _make_result("HOLD", 25, "NewsCatalyst"),
        ])
        assert cluster.cluster_signal == "BUY"
        assert cluster.confidence == 0.70  # 0.60 + 0.10 boost
        assert cluster.gate_status == "passed"
        assert cluster.cluster_strength == 2
        assert cluster.distinct_sources == 2
        assert cluster.strongest_supplier == "Momentum"

    def test_sell_side_solo_rejected_with_direction(self):
        cluster = self.detector.detect([
            _make_result("SELL", 60, "NewsCatalyst"),
            _make_result("HOLD", 30, "Momentum"),
            _make_result("HOLD", 20, "Pullback"),
        ])
        assert cluster.cluster_signal == "HOLD"
        assert cluster.gate_status == "rejected_min_agreement"
        assert cluster.vote_direction == "SELL"
        assert cluster.agreeing_strategies == ["NewsCatalyst"]
        assert cluster.disagreeing_strategies == []
        # HOLD confidence follows the no-directional convention:
        # max(HOLD confs) * 0.8 = 0.30 * 0.8
        assert cluster.confidence == pytest.approx(0.24)

    def test_rejected_never_carries_supplier_or_boost(self):
        cluster = self.detector.detect([
            _make_result("STRONG BUY", 85, "Momentum"),
            _make_result("HOLD", 34, "Pullback"),
        ])
        assert cluster.cluster_signal == "HOLD"
        assert cluster.strongest_supplier is None
        assert cluster.boost_applied is None

    def test_threshold_three_rejects_pairs(self):
        self.detector.MIN_AGREEING_STRATEGIES = 3
        pair = self.detector.detect([
            _make_result("BUY", 60, "Momentum"),
            _make_result("BUY", 55, "Pullback"),
        ])
        assert pair.gate_status == "rejected_min_agreement"
        assert pair.cluster_strength == 2
        triple = self.detector.detect([
            _make_result("BUY", 60, "Momentum"),
            _make_result("BUY", 55, "Pullback"),
            _make_result("WEAK BUY", 40, "NewsCatalyst"),
        ])
        assert triple.gate_status == "passed"
        assert triple.cluster_signal == "BUY"

    def test_vote_sources_fold_coupled_strategies(self):
        """Two strategies mapped to one source count once."""
        self.detector.VOTE_SOURCES = {
            "Momentum": "sentiment+volume",
            "NewsCatalyst": "sentiment+volume",
            "Pullback": "Pullback",
        }
        folded = self.detector.detect([
            _make_result("BUY", 60, "Momentum"),
            _make_result("WEAK BUY", 45, "NewsCatalyst"),
            _make_result("HOLD", 25, "Pullback"),
        ])
        assert folded.cluster_signal == "HOLD"
        assert folded.gate_status == "rejected_min_agreement"
        assert folded.cluster_strength == 2      # raw votes stay visible
        assert folded.distinct_sources == 1      # but one source
        assert folded.agreeing_strategies == ["Momentum", "NewsCatalyst"]

        independent = self.detector.detect([
            _make_result("BUY", 60, "Momentum"),
            _make_result("WEAK BUY", 45, "Pullback"),
            _make_result("HOLD", 25, "NewsCatalyst"),
        ])
        assert independent.gate_status == "passed"
        assert independent.distinct_sources == 2

    def test_unmapped_strategy_is_its_own_source(self):
        cluster = self.detector.detect([
            _make_result("BUY", 60, "Momentum"),
            _make_result("BUY", 60, "SomethingNew"),
        ])
        assert cluster.gate_status == "passed"
        assert cluster.distinct_sources == 2

    def test_conflicting_takes_precedence_over_gate(self):
        """Opposing confident votes are CONFLICTING, not a rejected solo."""
        cluster = self.detector.detect([
            _make_result("BUY", 60, "Momentum"),
            _make_result("SELL", 60, "NewsCatalyst"),
        ])
        assert cluster.cluster_signal == "CONFLICTING"
        assert cluster.gate_status == "conflicting"
        assert cluster.vote_direction is None
