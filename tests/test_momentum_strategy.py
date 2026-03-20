"""Tests for MomentumStrategy — strategy-specific TA module.

Covers:
    - Rule logic: all 4 conditions → STRONG BUY, 3/4 → BUY, <=2 → HOLD
    - Overbought RSI blocks STRONG BUY
    - Volume filter
    - End-to-end with synthetic bars (uptrend, ranging, overbought)
    - Stop-loss / take-profit calculation
    - Edge cases and error handling

All tests use synthetic data or direct rule testing — no network calls.

Run with:
    python3 -m pytest tests/test_momentum_strategy.py -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.momentum import MomentumStrategy


# ===========================================================================
# Helpers — synthetic data builders
# ===========================================================================

def _make_gentle_uptrend(n: int = 200, seed: int = 10) -> pd.DataFrame:
    """Gentle uptrend: price > SMA20 > SMA50, moderate RSI."""
    rng = np.random.default_rng(seed)
    prices = np.linspace(100, 120, n) + rng.normal(0, 0.5, n)
    dates = pd.bdate_range(end="2025-06-15", periods=n)
    volume = np.full(n, 5_000_000, dtype=float)
    volume[-1] = 8_000_000  # 1.6x avg
    return pd.DataFrame({
        "Open": prices * 0.998, "High": prices * 1.005,
        "Low": prices * 0.995, "Close": prices, "Volume": volume,
    }, index=dates)


def _make_ranging_market(n: int = 100, seed: int = 20) -> pd.DataFrame:
    """Flat sideways market — SMA20 ≈ SMA50, no trend."""
    rng = np.random.default_rng(seed)
    prices = 120.0 + rng.normal(0, 1.5, n)
    dates = pd.bdate_range(end="2025-06-15", periods=n)
    volume = np.full(n, 5_000_000, dtype=float)
    return pd.DataFrame({
        "Open": prices * 0.998, "High": prices * 1.005,
        "Low": prices * 0.995, "Close": prices, "Volume": volume,
    }, index=dates)


def _make_overbought(n: int = 100, seed: int = 30) -> pd.DataFrame:
    """Sharp rally → RSI well above 65."""
    rng = np.random.default_rng(seed)
    flat = np.full(n - 20, 100.0) + rng.normal(0, 0.3, n - 20)
    rally = np.linspace(100, 200, 20) + rng.normal(0, 0.2, 20)
    prices = np.concatenate([flat, rally])
    dates = pd.bdate_range(end="2025-06-15", periods=n)
    volume = np.full(n, 8_000_000, dtype=float)
    return pd.DataFrame({
        "Open": prices * 0.998, "High": prices * 1.005,
        "Low": prices * 0.995, "Close": prices, "Volume": volume,
    }, index=dates)


def _make_low_volume_uptrend(n: int = 100, seed: int = 40) -> pd.DataFrame:
    """Uptrend with low volume on last bar."""
    rng = np.random.default_rng(seed)
    prices = np.linspace(100, 155, n) + rng.normal(0, 0.3, n)
    dates = pd.bdate_range(end="2025-06-15", periods=n)
    volume = np.full(n, 5_000_000, dtype=float)
    volume[-1] = 4_000_000  # 0.8x avg
    return pd.DataFrame({
        "Open": prices * 0.998, "High": prices * 1.005,
        "Low": prices * 0.995, "Close": prices, "Volume": volume,
    }, index=dates)


# ===========================================================================
# Rule logic tests (direct _apply_rules — no indicator calculation)
# ===========================================================================

class TestMomentumRulesDirectly:
    """Test _apply_rules with constructed indicator dicts."""

    def test_all_four_conditions_strong_buy(self):
        """4/4 conditions → STRONG BUY, 70-85% confidence."""
        ind = {
            "price": 150.0, "rsi": 57.5, "sma20": 148.0, "sma50": 145.0,
            "vol_ratio": 1.5, "atr": 2.0,
        }
        signal, confidence, reasoning = MomentumStrategy._apply_rules(ind, "BUY")
        assert signal == "STRONG BUY"
        assert 70.0 <= confidence <= 85.0
        assert len(reasoning) == 4

    def test_three_conditions_buy(self):
        """3/4 conditions → BUY, 45-60% confidence."""
        ind = {
            "price": 150.0, "rsi": 57.5, "sma20": 148.0, "sma50": 145.0,
            "vol_ratio": 1.5, "atr": 2.0,
        }
        # HOLD sentiment → only 3 conditions met
        signal, confidence, reasoning = MomentumStrategy._apply_rules(ind, "HOLD")
        assert signal == "BUY"
        assert 45.0 <= confidence <= 60.0
        assert len(reasoning) == 3

    def test_two_conditions_hold(self):
        """2/4 conditions → HOLD."""
        ind = {
            "price": 150.0, "rsi": 70.0,  # outside 50-65
            "sma20": 148.0, "sma50": 145.0,
            "vol_ratio": 1.0,  # below 1.3x
            "atr": 2.0,
        }
        signal, confidence, reasoning = MomentumStrategy._apply_rules(ind, "HOLD")
        assert signal == "HOLD"
        assert confidence == 25.0

    def test_rsi_below_50_hold(self):
        """RSI below 50 = momentum condition fails."""
        ind = {
            "price": 150.0, "rsi": 45.0, "sma20": 148.0, "sma50": 145.0,
            "vol_ratio": 1.5, "atr": 2.0,
        }
        signal, confidence, reasoning = MomentumStrategy._apply_rules(ind, "BUY")
        # 3/4 (uptrend + volume + sentiment) → BUY, not STRONG BUY
        assert signal == "BUY"

    def test_rsi_above_65_not_strong_buy(self):
        """RSI above 65 = momentum condition fails."""
        ind = {
            "price": 150.0, "rsi": 70.0, "sma20": 148.0, "sma50": 145.0,
            "vol_ratio": 1.5, "atr": 2.0,
        }
        signal, confidence, reasoning = MomentumStrategy._apply_rules(ind, "BUY")
        # 3/4 → BUY
        assert signal == "BUY"
        assert signal != "STRONG BUY"

    def test_no_uptrend_blocks_strong_buy(self):
        """Price below SMA20 = uptrend condition fails."""
        ind = {
            "price": 140.0, "rsi": 55.0, "sma20": 148.0, "sma50": 145.0,
            "vol_ratio": 1.5, "atr": 2.0,
        }
        signal, confidence, reasoning = MomentumStrategy._apply_rules(ind, "BUY")
        assert signal != "STRONG BUY"

    def test_low_volume_blocks_strong_buy(self):
        """Volume below 1.3x = volume condition fails."""
        ind = {
            "price": 150.0, "rsi": 57.5, "sma20": 148.0, "sma50": 145.0,
            "vol_ratio": 1.1, "atr": 2.0,
        }
        signal, confidence, reasoning = MomentumStrategy._apply_rules(ind, "BUY")
        # 3/4 → BUY
        assert signal == "BUY"
        assert signal != "STRONG BUY"

    def test_sell_sentiment_blocks(self):
        """SELL sentiment = sentiment condition fails."""
        ind = {
            "price": 150.0, "rsi": 57.5, "sma20": 148.0, "sma50": 145.0,
            "vol_ratio": 1.5, "atr": 2.0,
        }
        signal, confidence, reasoning = MomentumStrategy._apply_rules(ind, "SELL")
        assert signal == "BUY"  # 3/4

    def test_weak_buy_counts_as_aligned(self):
        """WEAK BUY satisfies sentiment condition."""
        ind = {
            "price": 150.0, "rsi": 57.5, "sma20": 148.0, "sma50": 145.0,
            "vol_ratio": 1.5, "atr": 2.0,
        }
        signal, confidence, reasoning = MomentumStrategy._apply_rules(ind, "WEAK BUY")
        assert signal == "STRONG BUY"

    def test_missing_indicators_hold(self):
        """None values → HOLD with 0 confidence."""
        ind = {"price": None, "rsi": None, "sma20": None, "sma50": None,
               "vol_ratio": 0, "atr": None}
        signal, confidence, reasoning = MomentumStrategy._apply_rules(ind, "BUY")
        assert signal == "HOLD"
        assert confidence == 0.0

    def test_confidence_scales_with_rsi_center(self):
        """RSI at sweet-spot center (57.5) should give max confidence."""
        ind_center = {
            "price": 150.0, "rsi": 57.5, "sma20": 148.0, "sma50": 145.0,
            "vol_ratio": 1.5, "atr": 2.0,
        }
        ind_edge = {
            "price": 150.0, "rsi": 50.0, "sma20": 148.0, "sma50": 145.0,
            "vol_ratio": 1.5, "atr": 2.0,
        }
        _, conf_center, _ = MomentumStrategy._apply_rules(ind_center, "BUY")
        _, conf_edge, _ = MomentumStrategy._apply_rules(ind_edge, "BUY")
        assert conf_center >= conf_edge


# ===========================================================================
# End-to-end tests with synthetic bars
# ===========================================================================

class TestMomentumEndToEnd:
    """Integration tests with synthetic OHLCV data."""

    def test_gentle_uptrend_with_bullish_sentiment(self):
        """Gentle uptrend produces uptrend + RSI + volume conditions."""
        strat = MomentumStrategy()
        bars = _make_gentle_uptrend()
        result = strat.analyze("NVDA", bars, sentiment_signal="BUY")

        # With gentle uptrend: price > SMA20 > SMA50, RSI ~55-65
        assert result.signal == "STRONG BUY"
        assert 70.0 <= result.confidence <= 85.0
        assert result.strategy_name == "MomentumStrategy"

    def test_gentle_uptrend_reasoning(self):
        strat = MomentumStrategy()
        bars = _make_gentle_uptrend()
        result = strat.analyze("NVDA", bars, sentiment_signal="BUY")

        reasons = " ".join(result.reasoning).lower()
        assert "uptrend" in reasons
        assert "rsi" in reasons
        assert "volume" in reasons
        assert "sentiment" in reasons

    def test_ranging_market_hold(self):
        strat = MomentumStrategy()
        bars = _make_ranging_market()
        result = strat.analyze("NVDA", bars, sentiment_signal="HOLD")

        assert result.signal == "HOLD"
        assert result.confidence <= 30.0

    def test_ranging_with_sentiment_still_hold(self):
        strat = MomentumStrategy()
        bars = _make_ranging_market()
        result = strat.analyze("NVDA", bars, sentiment_signal="BUY")

        assert result.signal == "HOLD"

    def test_overbought_not_strong_buy(self):
        strat = MomentumStrategy()
        bars = _make_overbought()
        result = strat.analyze("NVDA", bars, sentiment_signal="BUY")

        rsi = result.indicators.get("rsi")
        assert rsi is not None and rsi > 65
        assert result.signal != "STRONG BUY"

    def test_low_volume_blocks_strong_buy(self):
        strat = MomentumStrategy()
        bars = _make_low_volume_uptrend()
        result = strat.analyze("NVDA", bars, sentiment_signal="BUY")

        assert result.signal != "STRONG BUY"
        assert result.indicators.get("vol_ratio", 0) < 1.3

    def test_stop_loss_and_take_profit(self):
        strat = MomentumStrategy()
        bars = _make_gentle_uptrend()
        result = strat.analyze("NVDA", bars, sentiment_signal="BUY")

        assert result.entry_price is not None
        assert result.stop_loss is not None
        assert result.take_profit is not None
        expected_stop = round(result.entry_price * 0.98, 4)
        assert result.stop_loss == expected_stop
        assert result.take_profit > result.entry_price


class TestMomentumEdgeCases:
    """Edge cases and metadata."""

    def test_insufficient_data(self):
        strat = MomentumStrategy()
        dates = pd.bdate_range(end="2025-06-15", periods=5)
        bars = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [5e6] * 5,
        }, index=dates)
        result = strat.analyze("NVDA", bars, sentiment_signal="BUY")
        assert result.signal == "HOLD"

    def test_preferred_tickers(self):
        strat = MomentumStrategy()
        assert "META" in strat.PREFERRED_TICKERS
        assert "JPM" in strat.PREFERRED_TICKERS

    def test_indicators_populated(self):
        strat = MomentumStrategy()
        bars = _make_gentle_uptrend()
        result = strat.analyze("NVDA", bars, sentiment_signal="BUY")
        for key in ("price", "rsi", "sma20", "sma50", "vol_ratio", "atr"):
            assert key in result.indicators
            assert result.indicators[key] is not None
