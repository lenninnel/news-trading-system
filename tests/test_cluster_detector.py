"""Tests for the cluster detection module."""

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

    def test_single_strategy_no_boost(self):
        results = [_make_result("BUY", 70, "Momentum")]
        cluster = self.detector.detect(results)
        assert cluster.cluster_signal == "BUY"
        assert cluster.confidence == 0.70  # no boost
        assert cluster.cluster_strength == 1

    def test_low_confidence_filtered(self):
        """Signals below MIN_CONFIDENCE (0.35 = 35%) don't count toward cluster."""
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
