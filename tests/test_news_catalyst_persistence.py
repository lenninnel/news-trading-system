"""Tests for NewsCatalyst signal persistence to signal_events.

Covers:
  1. NewsCatalyst results are logged to signal_events via _run_news_catalyst
  2. Logged signals are readable via the /api/signals endpoint
  3. ClusterDetector correctly counts NewsCatalyst signals
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.signal_logger import SignalLogger
from orchestrator.cluster_detector import ClusterDetector
from strategies.base import StrategyResult
from strategies.news_catalyst import NewsCatalystStrategy
from storage.database import Database


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_bars(days=30, base_price=100.0, last_volume_mult=1.5, last_pct_change=1.0):
    """Synthetic OHLCV bars with controllable last-bar stats."""
    dates = pd.date_range(end="2026-03-25", periods=days, freq="B")
    rng = np.random.default_rng(42)
    prices = base_price + rng.normal(0, 0.5, days).cumsum()
    if days >= 2:
        prices[-1] = prices[-2] * (1 + last_pct_change / 100)
    volumes = np.full(days, 1_000_000, dtype=float)
    volumes[-1] = int(volumes[-1] * last_volume_mult)
    return pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.01,
        "Low": prices * 0.99,
        "Close": prices,
        "Volume": volumes,
    }, index=dates)


def _make_sentiment(avg_score=0.6, signal="BUY", n_headlines=3):
    """Build a sentiment dict matching what the coordinator produces."""
    scored = [
        {"score": avg_score, "headline": f"h{i}", "source": "newsapi"}
        for i in range(n_headlines)
    ]
    return {
        "avg_score": avg_score,
        "signal": signal,
        "scored": scored,
        "source_breakdown": {"newsapi": {"count": n_headlines, "avg": avg_score}},
    }


# ── Test 1: NewsCatalyst logs to signal_events ───────────────────────────


class TestNewsCatalystLogsToSignalEvents:
    """_run_news_catalyst must call signal_logger.log with the right fields."""

    def test_news_catalyst_logs_to_signal_events(self):
        """A BUY NewsCatalyst result must be persisted to signal_events."""
        from orchestrator.coordinator import Coordinator

        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock(spec=SignalLogger)

        bars = _make_bars(last_volume_mult=1.5, last_pct_change=1.0)
        sentiment = _make_sentiment(avg_score=0.6, signal="BUY", n_headlines=3)

        coordinator._run_news_catalyst("AAPL", bars, sentiment, session="EOD")

        # signal_logger.log must have been called (once for the NewsCatalyst result)
        coordinator.signal_logger.log.assert_called_once()
        logged = coordinator.signal_logger.log.call_args[0][0]

        assert logged["ticker"] == "AAPL"
        assert logged["strategy"] == "NewsCatalyst"
        assert logged["session"] == "EOD"
        assert logged["signal"] in ("BUY", "WEAK BUY", "HOLD", "SELL", "WEAK SELL")
        # news_score should be populated (mapped from indicators)
        assert logged["news_score"] is not None
        assert logged["price_at_signal"] is not None

    def test_news_catalyst_hold_still_logged(self):
        """Even a HOLD result (no news) must be logged — every signal gets persisted."""
        from orchestrator.coordinator import Coordinator

        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock(spec=SignalLogger)

        bars = _make_bars()
        # No scored headlines → news_data will be None → HOLD
        sentiment = {"avg_score": 0.0, "signal": "HOLD", "scored": []}

        coordinator._run_news_catalyst("TSLA", bars, sentiment, session="US_OPEN")

        coordinator.signal_logger.log.assert_called_once()
        logged = coordinator.signal_logger.log.call_args[0][0]
        assert logged["ticker"] == "TSLA"
        assert logged["strategy"] == "NewsCatalyst"
        assert logged["signal"] == "HOLD"

    def test_news_catalyst_fields_populated(self):
        """Verify news_score, volume_ratio, and price_at_signal are set for BUY."""
        from orchestrator.coordinator import Coordinator

        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock(spec=SignalLogger)

        bars = _make_bars(last_volume_mult=2.0, last_pct_change=2.0)
        sentiment = _make_sentiment(avg_score=0.8, signal="BUY", n_headlines=5)

        coordinator._run_news_catalyst("META", bars, sentiment, session="EOD")

        logged = coordinator.signal_logger.log.call_args[0][0]
        # news_score comes from indicators
        assert logged["news_score"] is not None
        assert logged["news_score"] > 0
        # volume_ratio comes from rvol in indicators
        assert logged["volume_ratio"] is not None
        assert logged["volume_ratio"] > 1.0  # we set 2.0x volume spike
        assert logged["price_at_signal"] is not None


# ── Test 2: Signals readable via API ─────────────────────────────────────


class TestNewsCatalystSignalReadableViaApi:
    """NewsCatalyst signals written to signal_events must be queryable."""

    def test_news_catalyst_signal_readable_via_api(self, tmp_path):
        """Write a NewsCatalyst signal and read it back via SignalLogger.get_signals."""
        db = Database(str(tmp_path / "test.db"))
        logger = SignalLogger(db=db)

        # Simulate what _log_strategy_result writes for a NewsCatalyst BUY
        logger.log({
            "ticker": "AAPL",
            "session": "EOD",
            "strategy": "NewsCatalyst",
            "signal": "BUY",
            "confidence": 0.65,
            "rsi": None,
            "sma_ratio": None,
            "volume_ratio": 1.8,
            "sentiment_score": None,
            "news_score": 0.85,
            "social_score": None,
            "bull_case": None,
            "bear_case": None,
            "debate_outcome": None,
            "price_at_signal": 175.50,
            "trade_executed": 0,
            "trade_id": None,
        })

        # Read it back
        signals = logger.get_signals("AAPL", days=1)
        assert len(signals) == 1
        s = signals[0]
        assert s["ticker"] == "AAPL"
        assert s["strategy"] == "NewsCatalyst"
        assert s["signal"] == "BUY"
        assert s["confidence"] == 0.65
        assert s["news_score"] == 0.85
        assert s["volume_ratio"] == 1.8
        assert s["price_at_signal"] == 175.50

    def test_news_catalyst_appears_alongside_momentum(self, tmp_path):
        """Both NewsCatalyst and Momentum signals should coexist in signal_events."""
        db = Database(str(tmp_path / "test.db"))
        logger = SignalLogger(db=db)

        # Log a Momentum signal
        logger.log({
            "ticker": "AAPL",
            "strategy": "Momentum",
            "signal": "BUY",
            "confidence": 0.60,
        })
        # Log a NewsCatalyst signal
        logger.log({
            "ticker": "AAPL",
            "strategy": "NewsCatalyst",
            "signal": "WEAK BUY",
            "confidence": 0.45,
            "news_score": 0.75,
        })

        signals = logger.get_signals("AAPL", days=1)
        strategies = {s["strategy"] for s in signals}
        assert "Momentum" in strategies
        assert "NewsCatalyst" in strategies
        assert len(signals) == 2


# ── Test 3: ClusterDetector counts NewsCatalyst ──────────────────────────


class TestClusterDetectorCountsNewsCatalyst:
    """ClusterDetector must include NewsCatalyst results in cluster strength."""

    def test_cluster_detector_counts_news_catalyst(self):
        """Three-way agreement (Momentum + Pullback + NewsCatalyst) → strength 3."""
        detector = ClusterDetector()
        results = [
            StrategyResult(signal="BUY", confidence=65.0, strategy_name="Momentum"),
            StrategyResult(signal="WEAK BUY", confidence=45.0, strategy_name="Pullback"),
            StrategyResult(signal="BUY", confidence=65.0, strategy_name="NewsCatalyst"),
        ]
        cluster = detector.detect(results)
        assert cluster.cluster_strength == 3
        assert "NewsCatalyst" in cluster.agreeing_strategies

    def test_news_catalyst_disagree_flags_conflicting(self):
        """NewsCatalyst SELL vs Momentum BUY → CONFLICTING."""
        detector = ClusterDetector()
        results = [
            StrategyResult(signal="BUY", confidence=60.0, strategy_name="Momentum"),
            StrategyResult(signal="SELL", confidence=65.0, strategy_name="NewsCatalyst"),
        ]
        cluster = detector.detect(results)
        assert cluster.cluster_signal == "CONFLICTING"
        assert "NewsCatalyst" in cluster.disagreeing_strategies

    def test_news_catalyst_hold_does_not_reduce_cluster(self):
        """NewsCatalyst HOLD should not count as disagreeing."""
        detector = ClusterDetector()
        results = [
            StrategyResult(signal="BUY", confidence=60.0, strategy_name="Momentum"),
            StrategyResult(signal="BUY", confidence=55.0, strategy_name="Pullback"),
            StrategyResult(signal="HOLD", confidence=25.0, strategy_name="NewsCatalyst"),
        ]
        cluster = detector.detect(results)
        assert cluster.cluster_strength == 2  # only Momentum + Pullback
        assert "NewsCatalyst" not in cluster.disagreeing_strategies


# ── Test 4: _build_news_data helper ──────────────────────────────────────


class TestBuildNewsData:
    """Coordinator._build_news_data must correctly map sentiment → news_data."""

    def test_builds_news_data_from_sentiment(self):
        from orchestrator.coordinator import Coordinator

        sentiment = _make_sentiment(avg_score=0.6, signal="BUY", n_headlines=3)
        news_data = Coordinator._build_news_data(sentiment)

        assert news_data is not None
        assert news_data["headline_count"] == 3
        assert news_data["sentiment_direction"] == "BUY"
        # 0.6 → (0.6 + 1) / 2 = 0.8
        assert abs(news_data["news_score"] - 0.8) < 0.001

    def test_returns_none_when_no_scored(self):
        from orchestrator.coordinator import Coordinator

        sentiment = {"avg_score": 0.0, "signal": "HOLD", "scored": []}
        news_data = Coordinator._build_news_data(sentiment)
        assert news_data is None

    def test_negative_sentiment_maps_below_half(self):
        from orchestrator.coordinator import Coordinator

        sentiment = _make_sentiment(avg_score=-0.5, signal="SELL", n_headlines=2)
        news_data = Coordinator._build_news_data(sentiment)

        # -0.5 → (-0.5 + 1) / 2 = 0.25
        assert abs(news_data["news_score"] - 0.25) < 0.001
        assert news_data["sentiment_direction"] == "SELL"
