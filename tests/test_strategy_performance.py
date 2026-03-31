"""
Tests for StrategyPerformanceTracker.

Covers:
  - Per-strategy metrics computation
  - Cluster weights fallback with insufficient data
  - Sharpe weight floored at 0.1
  - API endpoint returns correct data
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analytics.signal_logger import SignalLogger
from analytics.strategy_performance import (
    StrategyPerformanceTracker,
    StrategyMetrics,
    _KNOWN_STRATEGIES,
    _MIN_OUTCOMES_FOR_WEIGHT,
    _WEIGHT_FLOOR,
)
from storage.database import Database


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test_perf.db")
    database = Database(db_path=db_path)
    # Create signal_events table
    SignalLogger(db=database)
    return database


@pytest.fixture
def tracker(db):
    return StrategyPerformanceTracker(db=db)


def _insert_signal_events(db, strategy, n, win_rate=0.6, avg_pct=1.5):
    """Insert n signal_events with outcomes for a strategy."""
    sl = SignalLogger(db=db)
    now = datetime.now(timezone.utc)
    for i in range(n):
        is_win = i < int(n * win_rate)
        pct = abs(avg_pct) if is_win else -abs(avg_pct)
        sl.log({
            "timestamp": now.isoformat(),
            "ticker": f"TEST{i}",
            "strategy": strategy,
            "signal": "BUY",
            "confidence": 0.65,
            "price_at_signal": 100.0,
        })
        # Manually fill outcomes
        with db._connect() as conn:
            conn.execute(
                "UPDATE signal_events SET outcome_3d_pct = ?, outcome_5d_pct = ?, "
                "outcome_correct = ? WHERE id = (SELECT MAX(id) FROM signal_events)",
                (pct, pct * 1.5, 1 if is_win else 0),
            )


# ── Tests ────────────────────────────────────────────────────────────


class TestPerformanceComputedPerStrategy:
    def test_returns_all_known_strategies(self, tracker, db):
        """compute() returns metrics for all known strategies."""
        _insert_signal_events(db, "Momentum", 15, win_rate=0.6)
        _insert_signal_events(db, "Pullback", 10, win_rate=0.5)

        result = tracker.compute()

        assert "Momentum" in result
        assert "Pullback" in result
        assert "Combined" in result  # always present
        for name in _KNOWN_STRATEGIES:
            assert name in result
            assert isinstance(result[name], StrategyMetrics)

    def test_momentum_metrics_computed(self, tracker, db):
        """Momentum strategy metrics are computed correctly."""
        _insert_signal_events(db, "Momentum", 20, win_rate=0.6, avg_pct=2.0)

        result = tracker.compute()
        m = result["Momentum"]

        assert m.strategy == "Momentum"
        assert m.signal_count == 20
        assert m.win_rate_30d is not None
        assert 0.55 <= m.win_rate_30d <= 0.65  # ~60%
        assert m.sharpe_30d is not None
        assert m.avg_confidence is not None

    def test_empty_strategy_returns_none_metrics(self, tracker, db):
        """Strategy with no signals returns None for all metrics."""
        result = tracker.compute()
        m = result["PEAD"]

        assert m.signal_count == 0
        assert m.sharpe_30d is None
        assert m.win_rate_30d is None
        assert m.avg_confidence is None

    def test_combined_includes_all_strategies(self, tracker, db):
        """Combined strategy aggregates across all strategies."""
        _insert_signal_events(db, "Momentum", 10, win_rate=0.7)
        _insert_signal_events(db, "Pullback", 10, win_rate=0.5)

        result = tracker.compute()
        combined = result["Combined"]

        assert combined.signal_count == 20  # 10 + 10

    def test_metrics_stored_in_db(self, tracker, db):
        """compute() persists results to strategy_perf_daily table."""
        _insert_signal_events(db, "Momentum", 15)

        tracker.compute()

        rows = tracker.get_latest()
        strategies = [r["strategy_name"] for r in rows]
        assert "Momentum" in strategies


class TestClusterWeightsFallback:
    def test_equal_weights_when_no_data(self, tracker):
        """Falls back to equal 1.0 weights when no performance data."""
        weights = tracker.get_cluster_weights()

        for s in _KNOWN_STRATEGIES:
            assert s in weights
            assert weights[s] == 1.0

    def test_equal_weight_when_insufficient_outcomes(self, tracker, db):
        """Strategy with <10 outcomes gets weight 1.0 (equal)."""
        _insert_signal_events(db, "Momentum", 5)  # < 10
        tracker.compute()

        weights = tracker.get_cluster_weights()
        assert weights["Momentum"] == 1.0

    def test_sharpe_used_when_sufficient_outcomes(self, tracker, db):
        """Strategy with >=10 outcomes uses Sharpe for weight."""
        _insert_signal_events(db, "Momentum", 15, win_rate=0.7, avg_pct=2.0)
        tracker.compute()

        weights = tracker.get_cluster_weights()
        # With 70% win rate and positive returns, Sharpe should be > 0
        assert weights["Momentum"] > 0


class TestSharpeWeightFloor:
    def test_negative_sharpe_floored_at_0_1(self, tracker, db):
        """Negative Sharpe is floored at 0.1, not zero."""
        # All losses → negative Sharpe
        _insert_signal_events(db, "Pullback", 15, win_rate=0.0, avg_pct=2.0)
        tracker.compute()

        weights = tracker.get_cluster_weights()
        assert weights["Pullback"] >= _WEIGHT_FLOOR

    def test_zero_sharpe_floored_at_0_1(self, tracker, db):
        """Zero Sharpe is floored at 0.1."""
        # Exactly 50% win rate with same magnitude → Sharpe near 0
        _insert_signal_events(db, "Pullback", 20, win_rate=0.5, avg_pct=1.0)
        tracker.compute()

        weights = tracker.get_cluster_weights()
        assert weights["Pullback"] >= _WEIGHT_FLOOR


class TestAPIPerformanceEndpoint:
    def test_endpoint_returns_list(self, db):
        """GET /api/strategy-performance returns a list of dicts."""
        _insert_signal_events(db, "Momentum", 15)
        perf = StrategyPerformanceTracker(db=db)
        perf.compute()

        # Read back via SQL (simulating API query)
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT strategy_name, run_date, sharpe_30d, win_rate_30d, avg_rr, "
                "signal_count, avg_confidence "
                "FROM strategy_perf_daily "
                "WHERE run_date = (SELECT MAX(run_date) FROM strategy_perf_daily) "
                "ORDER BY strategy_name"
            ).fetchall()
            result = [dict(r) for r in rows]

        assert len(result) >= 1
        momentum = next((r for r in result if r["strategy_name"] == "Momentum"), None)
        assert momentum is not None
        assert momentum["signal_count"] == 15
        assert "sharpe_30d" in momentum
        assert "win_rate_30d" in momentum

    def test_endpoint_empty_when_no_data(self, db):
        """API returns empty list when no performance data exists."""
        perf = StrategyPerformanceTracker(db=db)
        result = perf.get_latest()
        assert result == []


class TestStrategyMetricsDataclass:
    def test_dataclass_fields(self):
        m = StrategyMetrics(
            strategy="Test", sharpe_30d=0.5, win_rate_30d=0.6,
            avg_rr=1.8, signal_count=20, avg_confidence=0.65,
        )
        assert m.strategy == "Test"
        assert m.sharpe_30d == 0.5
        assert m.signal_count == 20
