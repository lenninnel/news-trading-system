"""
Unit tests for Coordinator signal fusion and confidence scoring.

Both combine_signals() and confidence() are static methods with no I/O,
so these tests run entirely offline and complete in milliseconds.

Run with:
    python3 -m pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from orchestrator.coordinator import Coordinator

combine  = Coordinator.combine_signals
conf     = Coordinator.confidence


# ===========================================================================
# combine_signals — all 9 cells of the fusion matrix
# ===========================================================================

class TestCombineSignals:
    """Verify every cell in the fusion matrix."""

    # --- STRONG signals ---
    def test_buy_buy_gives_strong_buy(self):
        assert combine("BUY", "BUY") == "STRONG BUY"

    def test_sell_sell_gives_strong_sell(self):
        assert combine("SELL", "SELL") == "STRONG SELL"

    # --- WEAK signals ---
    def test_buy_hold_gives_weak_buy(self):
        assert combine("BUY", "HOLD") == "WEAK BUY"

    def test_sell_hold_gives_weak_sell(self):
        assert combine("SELL", "HOLD") == "WEAK SELL"

    # --- CONFLICTING ---
    def test_buy_sell_gives_conflicting(self):
        assert combine("BUY", "SELL") == "CONFLICTING"

    def test_sell_buy_gives_conflicting(self):
        assert combine("SELL", "BUY") == "CONFLICTING"

    # --- HOLD (technical signal ignored when sentiment is neutral) ---
    def test_hold_hold_gives_hold(self):
        assert combine("HOLD", "HOLD") == "HOLD"

    def test_hold_buy_gives_hold(self):
        assert combine("HOLD", "BUY") == "HOLD"

    def test_hold_sell_gives_hold(self):
        assert combine("HOLD", "SELL") == "HOLD"


# ===========================================================================
# confidence — scaling behaviour
# ===========================================================================

class TestConfidence:
    """Confidence should scale with sentiment strength and combined signal type."""

    # --- STRONG BUY / SELL: range [0.60, 1.00] ---

    def test_strong_buy_full_sentiment_gives_max_confidence(self):
        assert conf("STRONG BUY", 1.0) == 1.0

    def test_strong_sell_full_sentiment_gives_max_confidence(self):
        assert conf("STRONG SELL", -1.0) == 1.0

    def test_strong_buy_minimum_sentiment_gives_0_6(self):
        # avg_score = 0.0 → 0.6 + 0*0.4 = 0.6
        assert conf("STRONG BUY", 0.0) == 0.6

    def test_strong_signal_scales_between_0_6_and_1_0(self):
        c = conf("STRONG BUY", 0.5)
        assert 0.6 <= c <= 1.0

    # --- WEAK BUY / SELL: range [0.20, 0.60] ---

    def test_weak_buy_full_sentiment_gives_0_6(self):
        assert conf("WEAK BUY", 1.0) == 0.6

    def test_weak_sell_full_sentiment_gives_0_6(self):
        assert conf("WEAK SELL", -1.0) == 0.6

    def test_weak_buy_zero_sentiment_gives_0_2(self):
        assert conf("WEAK BUY", 0.0) == 0.2

    def test_weak_signal_scales_between_0_2_and_0_6(self):
        c = conf("WEAK SELL", 0.3)
        assert 0.2 <= c <= 0.6

    # --- CONFLICTING: fixed 0.10 ---

    def test_conflicting_is_always_0_10(self):
        assert conf("CONFLICTING", 1.0)  == 0.10
        assert conf("CONFLICTING", -1.0) == 0.10
        assert conf("CONFLICTING", 0.0)  == 0.10

    # --- HOLD: fixed 0.25 ---

    def test_hold_is_always_0_25(self):
        assert conf("HOLD", 1.0)  == 0.25
        assert conf("HOLD", -1.0) == 0.25
        assert conf("HOLD", 0.0)  == 0.25

    # --- Confidence is always within [0, 1] ---

    @pytest.mark.parametrize("signal,score", [
        ("STRONG BUY",  0.31),
        ("STRONG SELL", -0.31),
        ("WEAK BUY",    0.31),
        ("WEAK SELL",   -0.31),
        ("CONFLICTING", 0.5),
        ("HOLD",        0.0),
    ])
    def test_confidence_always_in_0_to_1(self, signal, score):
        c = conf(signal, score)
        assert 0.0 <= c <= 1.0

    # --- Symmetry: positive and negative avg_score give same confidence ---

    @pytest.mark.parametrize("signal", ["STRONG BUY", "STRONG SELL"])
    def test_strong_signals_symmetric_around_zero(self, signal):
        assert conf(signal, 0.7) == conf(signal, -0.7)

    @pytest.mark.parametrize("signal", ["WEAK BUY", "WEAK SELL"])
    def test_weak_signals_symmetric_around_zero(self, signal):
        assert conf(signal, 0.4) == conf(signal, -0.4)

    # --- Monotonicity: higher |avg_score| → higher confidence ---

    def test_strong_buy_confidence_increases_with_sentiment_strength(self):
        low  = conf("STRONG BUY", 0.3)
        high = conf("STRONG BUY", 0.9)
        assert high > low

    def test_weak_buy_confidence_increases_with_sentiment_strength(self):
        low  = conf("WEAK BUY", 0.3)
        high = conf("WEAK BUY", 0.9)
        assert high > low
