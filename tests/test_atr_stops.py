"""
Tests for ATR-based dynamic stop-loss and take-profit.

Covers:
  - Wider stops for high-volatility stocks
  - Tighter stops for low-volatility stocks
  - Minimum R:R ratio enforced
  - Position size scales inversely with ATR
  - Fallback to fixed stops when ATR unavailable
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.risk_agent import RiskAgent


# ── Helpers ─────────────────────────────────────────────────────────

def make_agent() -> RiskAgent:
    db = MagicMock()
    db.log_risk_calculation.return_value = 99
    return RiskAgent(db=db)


def make_prices(daily_pct_change: float, n: int = 30, base: float = 100.0) -> pd.Series:
    """Generate a price series with a given average daily % move."""
    rng = np.random.RandomState(42)
    returns = rng.normal(0.001, daily_pct_change, n)
    prices = base * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=pd.bdate_range("2026-01-01", periods=n))


# ── Tests ───────────────────────────────────────────────────────────


class TestATRStopWiderForHighVol:
    def test_high_vol_gets_wider_stop(self):
        """TSLA-like stock (4% daily vol) should have a wider stop distance."""
        agent = make_agent()
        prices = make_prices(daily_pct_change=0.04, base=350.0)
        entry = float(prices.iloc[-1])

        result = agent.calculate_atr_stops(
            "TSLA", entry, direction="BUY", prices=prices,
        )

        assert result["atr_available"] is True
        assert result["stop_distance"] > 5.0  # >$5 stop for $350 stock
        # Stop should be significantly below entry
        assert result["stop_loss"] < entry - 5.0

    def test_high_vol_stop_distance_proportional_to_atr(self):
        """Stop distance should be ~1.5x ATR."""
        agent = make_agent()
        prices = make_prices(daily_pct_change=0.03, base=200.0)
        entry = float(prices.iloc[-1])

        result = agent.calculate_atr_stops(
            "ENPH", entry, direction="BUY", prices=prices,
            atr_stop_multiplier=1.5,
        )

        expected_stop = result["atr"] * 1.5
        assert abs(result["stop_distance"] - expected_stop) < 0.01


class TestATRStopTighterForLowVol:
    def test_low_vol_gets_tighter_stop(self):
        """JPM-like stock (0.8% daily vol) should have a tighter stop."""
        agent = make_agent()
        prices = make_prices(daily_pct_change=0.008, base=220.0)
        entry = float(prices.iloc[-1])

        result = agent.calculate_atr_stops(
            "JPM", entry, direction="BUY", prices=prices,
        )

        assert result["atr_available"] is True
        assert result["stop_distance"] < 8.0  # <$8 for $220 stock

    def test_low_vol_narrower_than_high_vol(self):
        """Low-vol stock should have tighter stops than high-vol stock."""
        agent = make_agent()

        low_vol = make_prices(daily_pct_change=0.008, base=200.0)
        high_vol = make_prices(daily_pct_change=0.04, base=200.0)

        r_low = agent.calculate_atr_stops("JPM", 200.0, prices=low_vol)
        r_high = agent.calculate_atr_stops("TSLA", 200.0, prices=high_vol)

        assert r_low["stop_distance"] < r_high["stop_distance"]


class TestMinimumRRRatioEnforced:
    def test_rr_ratio_at_least_2(self):
        """R:R should always be >= 2.0 even if ATR multipliers give less."""
        agent = make_agent()
        prices = make_prices(daily_pct_change=0.02, base=100.0)
        entry = float(prices.iloc[-1])

        # Use 1.5 stop, 2.5 TP — that's only 1.67:1 R:R
        # But the code enforces min 2:1
        result = agent.calculate_atr_stops(
            "TEST", entry, direction="BUY", prices=prices,
            atr_stop_multiplier=1.5, atr_tp_multiplier=2.5,
        )

        assert result["rr_ratio"] >= 2.0

    def test_rr_ratio_with_default_multipliers(self):
        """Default 1.5 stop / 3.0 TP gives exactly 2:1 R:R."""
        agent = make_agent()
        prices = make_prices(daily_pct_change=0.02, base=100.0)
        entry = float(prices.iloc[-1])

        result = agent.calculate_atr_stops(
            "TEST", entry, direction="BUY", prices=prices,
            atr_stop_multiplier=1.5, atr_tp_multiplier=3.0,
        )

        assert result["rr_ratio"] == 2.0


class TestPositionSizeScalesWithATR:
    def test_high_vol_smaller_position(self):
        """High-vol stock should get fewer shares (same risk budget)."""
        agent = make_agent()

        low_vol = make_prices(daily_pct_change=0.008, base=100.0)
        high_vol = make_prices(daily_pct_change=0.04, base=100.0)

        r_low = agent.calculate_atr_stops(
            "JPM", 100.0, prices=low_vol, account_balance=10_000.0,
        )
        r_high = agent.calculate_atr_stops(
            "TSLA", 100.0, prices=high_vol, account_balance=10_000.0,
        )

        assert r_high["shares"] < r_low["shares"]

    def test_position_respects_risk_budget(self):
        """Position risk should not exceed account_risk_pct of account."""
        agent = make_agent()
        prices = make_prices(daily_pct_change=0.02, base=100.0)

        result = agent.calculate_atr_stops(
            "TEST", 100.0, prices=prices,
            account_balance=10_000.0, account_risk_pct=0.01,
        )

        max_risk = result["shares"] * result["stop_distance"]
        assert max_risk <= 10_000.0 * 0.01 + 1.0  # +$1 tolerance for rounding


class TestFallbackToFixedWhenATRUnavailable:
    def test_insufficient_data_returns_unavailable(self):
        """When ATR cannot be computed, atr_available=False."""
        agent = make_agent()
        agent._fetch_atr = MagicMock(return_value=None)

        result = agent.calculate_atr_stops("ZZZZZ", 100.0)

        assert result.get("atr_available") is False

    def test_run_falls_back_to_fixed_when_atr_fails(self):
        """RiskAgent.run() uses fixed stops when ATR is unavailable."""
        agent = make_agent()

        # Mock _fetch_atr to return None (unavailable)
        agent._fetch_atr = MagicMock(return_value=None)

        with patch("agents.risk_agent.get_days_to_earnings", return_value=None), \
             patch("agents.risk_agent.USE_ATR_STOPS", True):
            result = agent.run(
                ticker="TEST", signal="STRONG BUY", confidence=75.0,
                current_price=100.0, account_balance=10_000.0,
            )

        assert not result["skipped"]
        # Fixed stop for STRONG BUY = 2%
        assert result["stop_loss"] is not None
        assert result["take_profit"] is not None

    def test_run_uses_atr_stops_when_available(self):
        """RiskAgent.run() uses ATR stops when available."""
        agent = make_agent()

        # Mock _fetch_atr to return a known value
        agent._fetch_atr = MagicMock(return_value=3.0)  # $3 ATR on $100 stock

        with patch("agents.risk_agent.get_days_to_earnings", return_value=None), \
             patch("agents.risk_agent.USE_ATR_STOPS", True):
            result = agent.run(
                ticker="TEST", signal="STRONG BUY", confidence=75.0,
                current_price=100.0, account_balance=10_000.0,
            )

        assert not result["skipped"]
        # ATR stop = 100 - (3.0 × 1.5) = 95.50
        assert result["stop_loss"] == pytest.approx(95.5, abs=0.01)
        # ATR TP = 100 + (3.0 × 3.0) = 109.00
        assert result["take_profit"] == pytest.approx(109.0, abs=0.01)


class TestATRStopsSellDirection:
    def test_sell_stops_reversed(self):
        """SELL direction: stop above entry, TP below."""
        agent = make_agent()
        prices = make_prices(daily_pct_change=0.02, base=100.0)
        entry = float(prices.iloc[-1])

        result = agent.calculate_atr_stops(
            "TEST", entry, direction="SELL", prices=prices,
        )

        assert result["stop_loss"] > entry
        assert result["take_profit"] < entry
