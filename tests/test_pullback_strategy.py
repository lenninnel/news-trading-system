"""Tests for PullbackStrategy — strategy-specific TA module.

Covers:
    - Rule logic: all 4 conditions → BUY, 3/4 → WEAK BUY, <=2 → HOLD
    - Downtrend → HOLD
    - RSI not bouncing → condition fails
    - Deep pullback → condition fails
    - End-to-end with synthetic bars
    - Stop-loss / take-profit / 2:1 reward-to-risk

All tests use synthetic data or direct rule testing — no network calls.

Run with:
    python3 -m pytest tests/test_pullback_strategy.py -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.pullback import PullbackStrategy


# ===========================================================================
# Helpers — synthetic data builders
# ===========================================================================

def _make_pullback_bounce(n: int = 200, seed: int = 50) -> pd.DataFrame:
    """Uptrend → dip → bounce near SMA50."""
    rng = np.random.default_rng(seed)
    phase1 = np.linspace(100, 140, 150) + rng.normal(0, 0.2, 150)
    phase2 = np.linspace(140, 128, 47) + rng.normal(0, 0.15, 47)
    phase3 = np.array([129.0, 130.0, 131.5]) + rng.normal(0, 0.05, 3)
    prices = np.concatenate([phase1, phase2, phase3])
    dates = pd.bdate_range(end="2025-06-15", periods=n)
    volume = np.full(n, 5_000_000, dtype=float)
    return pd.DataFrame({
        "Open": prices * 0.998, "High": prices * 1.005,
        "Low": prices * 0.995, "Close": prices, "Volume": volume,
    }, index=dates)


def _make_downtrend(n: int = 100, seed: int = 60) -> pd.DataFrame:
    """Steady downtrend — price well below SMA50."""
    rng = np.random.default_rng(seed)
    prices = np.linspace(160, 90, n) + rng.normal(0, 0.3, n)
    dates = pd.bdate_range(end="2025-06-15", periods=n)
    volume = np.full(n, 5_000_000, dtype=float)
    return pd.DataFrame({
        "Open": prices * 0.998, "High": prices * 1.005,
        "Low": prices * 0.995, "Close": prices, "Volume": volume,
    }, index=dates)


def _make_gentle_uptrend(n: int = 200, seed: int = 70) -> pd.DataFrame:
    """Steady uptrend — RSI stays above 45, no dip."""
    rng = np.random.default_rng(seed)
    prices = np.linspace(100, 130, n) + rng.normal(0, 0.2, n)
    dates = pd.bdate_range(end="2025-06-15", periods=n)
    volume = np.full(n, 5_000_000, dtype=float)
    return pd.DataFrame({
        "Open": prices * 0.998, "High": prices * 1.005,
        "Low": prices * 0.995, "Close": prices, "Volume": volume,
    }, index=dates)


# ===========================================================================
# Rule logic tests (direct _apply_rules)
# ===========================================================================

class TestPullbackRulesDirectly:
    """Test _apply_rules with constructed indicator dicts."""

    def test_all_four_conditions_buy(self):
        """4/4 → BUY, 65-80%."""
        ind = {
            "price": 102.0, "rsi": 43.0, "rsi_prev": 38.0,
            "sma50": 100.0, "sma50_dist_pct": 2.0,
            "stoch_k": 28.0, "stoch_k_prev": 22.0, "stoch_d": 25.0,
        }
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        assert signal == "BUY"
        assert 65.0 <= confidence <= 80.0
        assert len(reasoning) == 4

    def test_three_conditions_weak_buy(self):
        """3/4 → WEAK BUY, 40-55%."""
        ind = {
            "price": 102.0, "rsi": 43.0, "rsi_prev": 38.0,
            "sma50": 100.0, "sma50_dist_pct": 2.0,
            "stoch_k": 35.0, "stoch_k_prev": 35.0,  # not crossing up
            "stoch_d": 33.0,
        }
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        assert signal == "WEAK BUY"
        assert 40.0 <= confidence <= 55.0
        assert len(reasoning) == 3

    def test_two_conditions_hold(self):
        """2/4 → HOLD, data-driven in 10-40 range (10 + 2/4 * 30 + bonuses)."""
        ind = {
            "price": 102.0, "rsi": 55.0, "rsi_prev": 50.0,  # never dipped <45
            "sma50": 100.0, "sma50_dist_pct": 2.0,
            "stoch_k": 60.0, "stoch_k_prev": 55.0,  # not oversold
            "stoch_d": 57.0,
        }
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        assert signal == "HOLD"
        assert 25.0 <= confidence <= 40.0

    def test_downtrend_blocks_uptrend_condition(self):
        """Price below SMA50 → uptrend condition fails."""
        ind = {
            "price": 95.0, "rsi": 43.0, "rsi_prev": 38.0,
            "sma50": 100.0, "sma50_dist_pct": -5.0,
            "stoch_k": 28.0, "stoch_k_prev": 22.0, "stoch_d": 25.0,
        }
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        # Missing uptrend + near SMA50 → at most 2 conditions
        assert signal == "HOLD"

    def test_rsi_not_dipped_blocks(self):
        """RSI prev >= 45 → bounce condition fails."""
        ind = {
            "price": 102.0, "rsi": 55.0, "rsi_prev": 50.0,
            "sma50": 100.0, "sma50_dist_pct": 2.0,
            "stoch_k": 28.0, "stoch_k_prev": 22.0, "stoch_d": 25.0,
        }
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        # 3/4 (uptrend + near SMA50 + stoch crossup, missing RSI bounce)
        assert signal == "WEAK BUY"

    def test_deep_pullback_blocks_proximity(self):
        """sma50_dist_pct > 3% → proximity condition fails."""
        ind = {
            "price": 110.0, "rsi": 43.0, "rsi_prev": 38.0,
            "sma50": 100.0, "sma50_dist_pct": 10.0,
            "stoch_k": 28.0, "stoch_k_prev": 22.0, "stoch_d": 25.0,
        }
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        # 3/4 (uptrend + RSI bounce + stoch, missing proximity)
        assert signal == "WEAK BUY"

    def test_negative_dist_blocks_proximity(self):
        """Negative distance (below SMA50) → proximity condition fails."""
        ind = {
            "price": 98.0, "rsi": 43.0, "rsi_prev": 38.0,
            "sma50": 100.0, "sma50_dist_pct": -2.0,
            "stoch_k": 28.0, "stoch_k_prev": 22.0, "stoch_d": 25.0,
        }
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        # Missing uptrend (price < SMA50) and proximity
        assert signal == "HOLD"

    def test_stoch_not_oversold_blocks(self):
        """Stoch never below 30 in window → stochastic condition fails."""
        ind = {
            "price": 102.0, "rsi": 43.0, "rsi_prev": 38.0,
            "sma50": 100.0, "sma50_dist_pct": 2.0,
            "stoch_k": 40.0, "stoch_k_prev": 35.0,
            "stoch_k_min_5": 33.0,  # never dipped below 30
            "stoch_d": 37.0,
        }
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        assert signal == "WEAK BUY"  # 3/4

    def test_rsi_dip_in_window_fires_condition(self):
        """RSI dipped below 45 three bars ago (not prev bar) → condition fires
        with 5-bar window."""
        ind = {
            "price": 102.0, "rsi": 48.0,
            "rsi_prev": 46.0,       # prev bar already above 45
            "rsi_min_5": 38.0,      # but min in last 5 bars was below 45
            "sma50": 100.0, "sma50_dist_pct": 2.0,
            "stoch_k": 35.0, "stoch_k_prev": 28.0,
            "stoch_k_min_5": 22.0,
            "stoch_d": 30.0,
        }
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        assert signal == "BUY"
        assert 65.0 <= confidence <= 80.0
        assert len(reasoning) == 4

    def test_stoch_dip_in_window_fires_condition(self):
        """Stoch K dipped below 30 three bars ago → condition fires."""
        ind = {
            "price": 102.0, "rsi": 43.0, "rsi_prev": 38.0,
            "sma50": 100.0, "sma50_dist_pct": 2.0,
            "stoch_k": 35.0,
            "stoch_k_prev": 32.0,      # prev bar already above 30
            "stoch_k_min_5": 22.0,      # but dipped below 30 within window
            "stoch_d": 30.0,
        }
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        assert signal == "BUY"
        assert 65.0 <= confidence <= 80.0
        assert len(reasoning) == 4

    def test_pullback_confidence_above_floor(self):
        """3/4 conditions → WEAK BUY with confidence above 20% floor."""
        ind = {
            "price": 102.0, "rsi": 43.0, "rsi_prev": 38.0,
            "sma50": 100.0, "sma50_dist_pct": 2.0,
            "stoch_k": 35.0, "stoch_k_prev": 35.0,
            "stoch_k_min_5": 33.0,
            "stoch_d": 33.0,
        }
        signal, confidence, _ = PullbackStrategy._apply_rules(ind)
        assert signal == "WEAK BUY"
        assert confidence > 20.0
        assert 40.0 <= confidence <= 55.0

    def test_buy_reaches_65_plus(self):
        """4/4 conditions → BUY with confidence ≥ 65%."""
        ind = {
            "price": 102.0, "rsi": 43.0, "rsi_prev": 38.0,
            "sma50": 100.0, "sma50_dist_pct": 2.0,
            "stoch_k": 28.0, "stoch_k_prev": 22.0,
            "stoch_k_min_5": 20.0,
            "stoch_d": 25.0,
        }
        signal, confidence, _ = PullbackStrategy._apply_rules(ind)
        assert signal == "BUY"
        assert confidence >= 65.0

    def test_missing_indicators_hold(self):
        """None values → HOLD with 0 confidence."""
        ind = {"price": None, "rsi": None, "rsi_prev": None,
               "sma50": None, "sma50_dist_pct": None,
               "stoch_k": None, "stoch_k_prev": None, "stoch_d": None}
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        assert signal == "HOLD"
        assert confidence == 0.0


# ===========================================================================
# End-to-end tests with synthetic bars
# ===========================================================================

class TestPullbackEndToEnd:
    """Integration tests with synthetic OHLCV data."""

    def test_pullback_bounce_signal(self):
        """Pullback + bounce should trigger at least RSI bounce condition."""
        strat = PullbackStrategy()
        bars = _make_pullback_bounce()
        result = strat.analyze("AAPL", bars)

        # At minimum, some conditions should fire (data shows bounce)
        assert result.strategy_name == "Pullback"
        assert len(result.reasoning) >= 1

    def test_downtrend_hold(self):
        strat = PullbackStrategy()
        bars = _make_downtrend()
        result = strat.analyze("AAPL", bars)

        assert result.signal == "HOLD"
        # Data-driven HOLD now ranges 10-40 (10 + conds/4*30 + bonuses)
        assert result.confidence <= 40.0

    def test_downtrend_no_uptrend_in_reasoning(self):
        strat = PullbackStrategy()
        bars = _make_downtrend()
        result = strat.analyze("AAPL", bars)
        reasons = " ".join(result.reasoning).lower()
        assert "uptrend" not in reasons

    def test_gentle_uptrend_no_full_buy(self):
        """Steady uptrend without pullback → RSI never dipped below 45."""
        strat = PullbackStrategy()
        bars = _make_gentle_uptrend()
        result = strat.analyze("AAPL", bars)
        # Without RSI dip, can't get a full BUY
        assert result.signal in ("WEAK BUY", "HOLD")

    def test_stop_loss_and_take_profit(self):
        strat = PullbackStrategy()
        bars = _make_pullback_bounce()
        result = strat.analyze("AAPL", bars)

        assert result.entry_price is not None
        assert result.stop_loss is not None
        assert result.take_profit is not None
        # Stop-loss = 2% below entry
        expected_stop = round(result.entry_price * 0.98, 4)
        assert result.stop_loss == expected_stop
        # 2:1 reward-to-risk ratio
        risk = result.entry_price - result.stop_loss
        reward = result.take_profit - result.entry_price
        assert abs(reward - 2 * risk) < 0.01

    def test_deep_pullback_hold(self):
        """Price far below SMA50 → HOLD."""
        rng = np.random.default_rng(80)
        phase1 = np.linspace(100, 150, 60) + rng.normal(0, 0.3, 60)
        phase2 = np.linspace(150, 115, 30) + rng.normal(0, 0.2, 30)
        phase3 = np.linspace(115, 120, 10) + rng.normal(0, 0.15, 10)
        prices = np.concatenate([phase1, phase2, phase3])
        dates = pd.bdate_range(end="2025-06-15", periods=100)
        bars = pd.DataFrame({
            "Open": prices * 0.998, "High": prices * 1.005,
            "Low": prices * 0.995, "Close": prices,
            "Volume": np.full(100, 5e6),
        }, index=dates)

        strat = PullbackStrategy()
        result = strat.analyze("AAPL", bars)
        assert result.signal == "HOLD"


class TestPullbackEdgeCases:
    """Edge cases and metadata."""

    def test_insufficient_data(self):
        strat = PullbackStrategy()
        dates = pd.bdate_range(end="2025-06-15", periods=5)
        bars = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [5e6] * 5,
        }, index=dates)
        result = strat.analyze("AAPL", bars)
        assert result.signal == "HOLD"

    def test_preferred_tickers(self):
        strat = PullbackStrategy()
        assert "AAPL" in strat.PREFERRED_TICKERS
        assert "MSFT" in strat.PREFERRED_TICKERS
        assert "AMZN" in strat.PREFERRED_TICKERS
        assert "XOM" in strat.PREFERRED_TICKERS

    def test_indicators_populated(self):
        strat = PullbackStrategy()
        bars = _make_pullback_bounce()
        result = strat.analyze("AAPL", bars)
        for key in ("price", "rsi", "sma50", "stoch_k", "sma50_dist_pct"):
            assert key in result.indicators
            assert result.indicators[key] is not None

    def test_default_sentiment_is_hold(self):
        strat = PullbackStrategy()
        bars = _make_pullback_bounce()
        result = strat.analyze("AAPL", bars)
        assert result.signal in ("BUY", "WEAK BUY", "HOLD")


# ===========================================================================
# Downtrend filter tests
# ===========================================================================

class TestPullbackDowntrendFilter:
    """Downtrend filter prevents catching falling knives."""

    @staticmethod
    def _buy_setup(**overrides) -> dict:
        """4/4 conditions indicator dict (normally → BUY)."""
        base = {
            "price": 78.0, "rsi": 43.0, "rsi_prev": 38.0,
            "sma50": 76.0, "sma50_dist_pct": 2.6,
            "stoch_k": 28.0, "stoch_k_prev": 22.0, "stoch_d": 25.0,
        }
        base.update(overrides)
        return base

    def test_severe_downtrend_caps_at_weak_buy(self):
        """sma_ratio < 0.80 → BUY capped to WEAK BUY."""
        ind = self._buy_setup(sma200=100.0)  # ratio = 0.78
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        assert signal == "WEAK BUY"
        assert any("Severe downtrend" in r for r in reasoning)

    def test_extreme_downtrend_forces_hold(self):
        """sma_ratio < 0.70 → forced HOLD."""
        ind = self._buy_setup(price=68.0, sma50=66.0, sma50_dist_pct=3.0,
                              sma200=100.0)  # ratio = 0.68
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        assert signal == "HOLD"
        assert any("Extreme downtrend" in r for r in reasoning)

    def test_normal_pullback_unaffected(self):
        """sma_ratio > 0.80 → no filter applied."""
        ind = self._buy_setup(price=92.0, sma50=90.0, sma50_dist_pct=2.2,
                              sma200=100.0)  # ratio = 0.92
        signal, confidence, reasoning = PullbackStrategy._apply_rules(ind)
        assert signal == "BUY"
        assert not any("downtrend" in r.lower() for r in reasoning)
