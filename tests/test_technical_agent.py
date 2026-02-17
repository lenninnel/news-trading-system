"""
Unit tests for TechnicalAgent signal logic.

Tests are intentionally offline: _apply_signal_rules() is a pure function
that takes a dict of indicator values, so no network or DB access is needed.

Run with:
    python3 -m pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from agents.technical_agent import TechnicalAgent


# Shorthand for calling the static method under test
def signal(ind: dict) -> tuple:
    return TechnicalAgent._apply_signal_rules(ind)


# ---------------------------------------------------------------------------
# Helpers — base indicator dict that sits well inside HOLD territory
# ---------------------------------------------------------------------------

def _hold_indicators(**overrides) -> dict:
    """Return a neutral indicator set, optionally overriding specific keys."""
    base = {
        "rsi":             50.0,
        "macd":            0.5,
        "macd_signal":     0.5,
        "macd_hist":       0.0,
        "macd_bull_cross": False,
        "macd_bear_cross": False,
        "sma_20":          200.0,
        "sma_50":          195.0,
        "bb_upper":        210.0,
        "bb_lower":        190.0,
        "price":           200.0,
    }
    base.update(overrides)
    return base


# ===========================================================================
# HOLD — no conditions triggered
# ===========================================================================

class TestHold:
    def test_neutral_indicators_give_hold(self):
        sig, reasons = signal(_hold_indicators())
        assert sig == "HOLD"

    def test_rsi_at_boundary_30_is_hold(self):
        sig, _ = signal(_hold_indicators(rsi=30.0))
        assert sig == "HOLD"  # strictly less-than, so 30.0 is NOT a BUY

    def test_rsi_at_boundary_70_is_hold(self):
        sig, _ = signal(_hold_indicators(rsi=70.0))
        assert sig == "HOLD"  # strictly greater-than, so 70.0 is NOT a SELL

    def test_price_at_bb_lower_is_hold(self):
        sig, _ = signal(_hold_indicators(price=190.0, bb_lower=190.0))
        assert sig == "HOLD"  # strictly less-than, so equal is NOT a BUY

    def test_price_at_bb_upper_is_hold(self):
        sig, _ = signal(_hold_indicators(price=210.0, bb_upper=210.0))
        assert sig == "HOLD"  # strictly greater-than, so equal is NOT a SELL

    def test_none_indicators_give_hold(self):
        """All indicators None (data unavailable) should not crash and give HOLD."""
        ind = {k: None for k in _hold_indicators()}
        sig, _ = signal(ind)
        assert sig == "HOLD"


# ===========================================================================
# BUY — individual triggers
# ===========================================================================

class TestBuySignals:
    def test_rsi_below_30_triggers_buy(self):
        sig, reasons = signal(_hold_indicators(rsi=29.9))
        assert sig == "BUY"
        assert any("RSI" in r and "oversold" in r for r in reasons)

    def test_rsi_deep_oversold_triggers_buy(self):
        sig, _ = signal(_hold_indicators(rsi=10.0))
        assert sig == "BUY"

    def test_macd_bull_crossover_triggers_buy(self):
        sig, reasons = signal(_hold_indicators(macd_bull_cross=True))
        assert sig == "BUY"
        assert any("bullish crossover" in r for r in reasons)

    def test_price_below_bb_lower_triggers_buy(self):
        sig, reasons = signal(_hold_indicators(price=189.9, bb_lower=190.0))
        assert sig == "BUY"
        assert any("below lower Bollinger" in r for r in reasons)

    def test_multiple_buy_conditions_all_appear_in_reasons(self):
        ind = _hold_indicators(rsi=25.0, macd_bull_cross=True, price=185.0, bb_lower=190.0)
        sig, reasons = signal(ind)
        assert sig == "BUY"
        assert len(reasons) == 3  # RSI + crossover + BB lower


# ===========================================================================
# SELL — individual triggers
# ===========================================================================

class TestSellSignals:
    def test_rsi_above_70_triggers_sell(self):
        sig, reasons = signal(_hold_indicators(rsi=70.1))
        assert sig == "SELL"
        assert any("RSI" in r and "overbought" in r for r in reasons)

    def test_rsi_deep_overbought_triggers_sell(self):
        sig, _ = signal(_hold_indicators(rsi=90.0))
        assert sig == "SELL"

    def test_macd_bear_crossover_triggers_sell(self):
        sig, reasons = signal(_hold_indicators(macd_bear_cross=True))
        assert sig == "SELL"
        assert any("bearish crossover" in r for r in reasons)

    def test_price_above_bb_upper_triggers_sell(self):
        sig, reasons = signal(_hold_indicators(price=210.1, bb_upper=210.0))
        assert sig == "SELL"
        assert any("above upper Bollinger" in r for r in reasons)

    def test_multiple_sell_conditions_all_appear_in_reasons(self):
        ind = _hold_indicators(rsi=80.0, macd_bear_cross=True, price=215.0, bb_upper=210.0)
        sig, reasons = signal(ind)
        assert sig == "SELL"
        assert len(reasons) == 3


# ===========================================================================
# Priority: BUY beats SELL on conflicting signals (same bar)
# ===========================================================================

class TestSignalPriority:
    def test_buy_takes_priority_over_sell_on_conflict(self):
        """RSI oversold AND overbought simultaneously is impossible, but
        BUY from BB-lower should override SELL from BB-upper (edge case)."""
        ind = _hold_indicators(
            rsi=25.0,       # triggers BUY
            macd_bear_cross=True,  # triggers SELL
            price=189.9,
            bb_lower=190.0,  # triggers BUY
            bb_upper=210.0,
        )
        sig, _ = signal(ind)
        assert sig == "BUY"

    def test_buy_priority_with_rsi_oversold_and_macd_bear(self):
        ind = _hold_indicators(rsi=25.0, macd_bear_cross=True)
        sig, _ = signal(ind)
        assert sig == "BUY"
