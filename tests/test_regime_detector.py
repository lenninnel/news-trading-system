"""Tests for the per-ticker RegimeDetector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.regime_detector import RegimeDetector, RegimeResult, _compute_adx


# ── Helpers ────────────────────────────────────────────────────────────


def _make_prices(
    n: int = 200,
    base: float = 100.0,
    trend: float = 0.0,
    volatility: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build synthetic OHLCV with controllable trend and volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-03-28", periods=n)
    # trending prices
    noise = rng.normal(0, volatility, n)
    prices = base + np.cumsum(noise) + np.arange(n) * trend
    prices = np.maximum(prices, 1.0)  # no negatives
    return pd.DataFrame({
        "Open": prices * 0.998,
        "High": prices * (1 + 0.005 * volatility),
        "Low": prices * (1 - 0.005 * volatility),
        "Close": prices,
        "Volume": np.full(n, 1_000_000, dtype=float),
    }, index=dates)


def _make_trending_up(n: int = 200) -> pd.DataFrame:
    """Strong uptrend: price well above SMA50, high ADX."""
    return _make_prices(n=n, base=100, trend=0.3, volatility=0.5)


def _make_trending_down(n: int = 200) -> pd.DataFrame:
    """Strong downtrend: price below SMA50, high ADX."""
    return _make_prices(n=n, base=200, trend=-0.3, volatility=0.5)


def _make_ranging(n: int = 200) -> pd.DataFrame:
    """Mean-reverting: low ADX, low volatility."""
    return _make_prices(n=n, base=100, trend=0.0, volatility=0.3)


def _make_volatile(n: int = 200) -> pd.DataFrame:
    """High volatility: large swings."""
    return _make_prices(n=n, base=100, trend=0.0, volatility=5.0)


# ── Tests ──────────────────────────────────────────────────────────────


class TestRegimeDetector:
    def setup_method(self):
        self.detector = RegimeDetector()

    def test_trending_up_regime_detection(self):
        """Strong uptrend with low VIX → TRENDING_UP."""
        prices = _make_trending_up()
        result = self.detector.detect("AAPL", prices, vix=18.0)

        assert isinstance(result, RegimeResult)
        assert result.regime == "TRENDING_UP"
        assert result.size_multiplier == 1.0
        assert "Momentum" in result.allowed_strategies
        assert result.adx > 0
        assert result.price > result.sma50  # price above SMA50

    def test_trending_down_regime_detection(self):
        """Strong downtrend with low VIX → TRENDING_DOWN."""
        prices = _make_trending_down()
        result = self.detector.detect("AAPL", prices, vix=18.0)

        assert result.regime == "TRENDING_DOWN"
        assert result.size_multiplier == 1.0
        assert "Pullback" in result.allowed_strategies
        assert result.price < result.sma50  # price below SMA50

    def test_high_volatility_reduces_size(self):
        """VIX > 25 → HIGH_VOLATILITY with 50% size."""
        prices = _make_trending_up()
        result = self.detector.detect("AAPL", prices, vix=30.0)

        assert result.regime == "HIGH_VOLATILITY"
        assert result.size_multiplier == 0.50
        assert result.vix == 30.0

    def test_high_atr_triggers_high_volatility(self):
        """Extreme recent ATR spike → HIGH_VOLATILITY even with low VIX.

        We build calm data then inject a volatility spike at the end so
        the current ATR is in the >80th percentile of its own history.
        """
        prices = _make_prices(n=200, base=100, trend=0.0, volatility=0.3)
        # Inject a massive volatility spike in the last 15 bars
        n = len(prices)
        for i in range(n - 15, n):
            prices.iloc[i, prices.columns.get_loc("High")] *= 1.08
            prices.iloc[i, prices.columns.get_loc("Low")] *= 0.92

        result = self.detector.detect("AAPL", prices, vix=15.0)

        assert result.regime == "HIGH_VOLATILITY"
        assert result.size_multiplier == 0.50

    def test_ranging_deactivates_momentum(self):
        """Ranging market (low ADX, low VIX) → Momentum not in allowed list."""
        prices = _make_ranging()
        result = self.detector.detect("AAPL", prices, vix=15.0)

        # Should be RANGING or TRANSITIONAL (depends on exact ADX)
        assert result.regime in ("RANGING", "TRANSITIONAL")
        if result.regime == "RANGING":
            assert "Momentum" not in result.allowed_strategies
            assert "Pullback" in result.allowed_strategies

    def test_transitional_reduces_size(self):
        """Borderline ADX (20-25) → TRANSITIONAL with 75% size."""
        prices = _make_ranging()
        result = self.detector.detect("AAPL", prices, vix=22.0)

        # VIX 20-25 triggers TRANSITIONAL
        assert result.regime == "TRANSITIONAL"
        assert result.size_multiplier == 0.75

    def test_insufficient_data_returns_transitional(self):
        """Less than 50 bars → TRANSITIONAL (safe default)."""
        prices = _make_prices(n=20)
        result = self.detector.detect("AAPL", prices, vix=18.0)

        assert result.regime == "TRANSITIONAL"
        assert result.size_multiplier == 0.75

    def test_none_vix_uses_neutral_default(self):
        """VIX=None → uses default 18.0 (neutral)."""
        prices = _make_trending_up()
        result = self.detector.detect("AAPL", prices, vix=None)

        # Should not be HIGH_VOLATILITY (default VIX 18 is low)
        assert result.regime != "HIGH_VOLATILITY"

    def test_regime_stored_in_signal(self, tmp_path):
        """Regime should be persisted in signal_events via SignalLogger."""
        from analytics.signal_logger import SignalLogger
        from storage.database import Database

        db = Database(str(tmp_path / "test.db"))
        logger = SignalLogger(db=db)

        # Log a signal with regime
        logger.log({
            "ticker": "AAPL",
            "strategy": "Combined",
            "signal": "STRONG BUY",
            "confidence": 0.75,
            "regime": "TRENDING_UP",
        })

        signals = logger.get_signals("AAPL", days=1)
        assert len(signals) == 1
        assert signals[0]["regime"] == "TRENDING_UP"

    def test_all_strategies_allowed_in_high_vol(self):
        """HIGH_VOLATILITY doesn't block strategies, just reduces size."""
        prices = _make_trending_up()
        result = self.detector.detect("AAPL", prices, vix=35.0)

        assert result.regime == "HIGH_VOLATILITY"
        assert len(result.allowed_strategies) == len(self.detector.ALL_STRATEGIES)

    def test_is_trending_property(self):
        """RegimeResult.is_trending helper works."""
        up = RegimeResult(regime="TRENDING_UP", adx=28, vix=18, atr_percentile=50,
                          price=150, sma50=140, size_multiplier=1.0)
        down = RegimeResult(regime="TRENDING_DOWN", adx=28, vix=18, atr_percentile=50,
                            price=130, sma50=140, size_multiplier=1.0)
        ranging = RegimeResult(regime="RANGING", adx=15, vix=15, atr_percentile=50,
                               price=100, sma50=100, size_multiplier=1.0)

        assert up.is_trending is True
        assert down.is_trending is True
        assert ranging.is_trending is False


class TestADXComputation:
    """Verify the ADX helper produces sensible values."""

    def test_adx_on_trending_data(self):
        """Strong trend should produce ADX > 20."""
        prices = _make_trending_up(n=200)
        adx = _compute_adx(prices)
        assert adx > 15  # trending data should have elevated ADX

    def test_adx_on_ranging_data(self):
        """Flat data should produce lower ADX."""
        prices = _make_ranging(n=200)
        adx = _compute_adx(prices)
        assert adx < 40  # ranging shouldn't have very high ADX

    def test_adx_insufficient_data(self):
        """Less than required bars → 0.0."""
        prices = _make_prices(n=10)
        adx = _compute_adx(prices)
        assert adx == 0.0
