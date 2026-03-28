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

    # --- WEAK BUY / SELL: range [0.35, 0.60] ---
    # Floor raised from 0.30 to 0.35 and technical confidence blended in so
    # that a normal trending market produces 40-60% confidence.

    def test_weak_buy_full_sentiment_gives_0_6(self):
        assert conf("WEAK BUY", 1.0) == 0.6

    def test_weak_sell_full_sentiment_gives_0_6(self):
        assert conf("WEAK SELL", -1.0) == 0.6

    def test_weak_buy_zero_sentiment_gives_0_35(self):
        # Floor raised to 0.35 (sentiment-only fallback)
        assert conf("WEAK BUY", 0.0) == 0.35

    def test_weak_signal_scales_between_0_35_and_0_6(self):
        c = conf("WEAK SELL", 0.3)
        assert 0.35 <= c <= 0.6

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


# ===========================================================================
# confidence — volume adjustments
# ===========================================================================

class TestConfidenceVolume:
    """Volume confirmation should boost confidence; low RVOL should reduce it."""

    def test_volume_confirmed_boosts_strong_buy(self):
        base = conf("STRONG BUY", 0.5)
        boosted = conf("STRONG BUY", 0.5, volume_confirmed=True)
        assert boosted == round(base + 0.10, 2)

    def test_volume_confirmed_boosts_weak_sell(self):
        base = conf("WEAK SELL", 0.5)
        boosted = conf("WEAK SELL", 0.5, volume_confirmed=True)
        assert boosted == round(base + 0.10, 2)

    def test_low_rvol_does_not_reduce_strong_buy(self):
        """Low rvol penalty was removed — rvol < 1.0 is normal intraday."""
        base = conf("STRONG BUY", 0.5)
        same = conf("STRONG BUY", 0.5, rvol=0.5)
        assert same == base

    def test_low_rvol_does_not_reduce_weak_sell(self):
        """Low rvol penalty was removed — rvol < 1.0 is normal intraday."""
        base = conf("WEAK SELL", 0.5)
        same = conf("WEAK SELL", 0.5, rvol=0.5)
        assert same == base

    def test_volume_does_not_affect_hold(self):
        assert conf("HOLD", 0.5, volume_confirmed=True) == conf("HOLD", 0.5)
        assert conf("HOLD", 0.5, rvol=0.3) == conf("HOLD", 0.5)

    def test_volume_does_not_affect_conflicting(self):
        assert conf("CONFLICTING", 0.5, volume_confirmed=True) == conf("CONFLICTING", 0.5)

    def test_confidence_clamped_to_1_with_boost(self):
        # STRONG BUY with max sentiment = 1.0, then +0.10 should still be 1.0
        c = conf("STRONG BUY", 1.0, volume_confirmed=True)
        assert c == 1.0

    def test_confidence_floor_with_low_rvol(self):
        # WEAK BUY with 0.0 sentiment = 0.35 (floor raised; no rvol penalty)
        c = conf("WEAK BUY", 0.0, rvol=0.3)
        assert c == 0.35
        assert c >= 0.0


# ===========================================================================
# confidence -- technical confidence blending
# ===========================================================================

class TestConfidenceTechnicalBlend:
    """When technical_confidence is provided, it blends with sentiment strength."""

    def test_technical_confidence_boosts_weak_signal(self):
        """A WEAK BUY with moderate sentiment (0.35) and strong technical (0.5)
        should produce confidence in the 40-60% range."""
        # blended = 0.35*0.6 + 0.5*0.4 = 0.21 + 0.20 = 0.41
        # base = 0.35 + 0.41*0.25 = 0.4525 -> 0.45
        c = conf("WEAK BUY", 0.35, technical_confidence=0.5)
        assert 0.40 <= c <= 0.60, f"Expected 40-60% for normal trend, got {c}"

    def test_strong_signal_with_high_technical_confidence(self):
        """STRONG BUY with moderate sentiment + high technical confidence
        should produce high confidence."""
        # blended = 0.4*0.6 + 0.65*0.4 = 0.24 + 0.26 = 0.50
        # base = 0.6 + 0.50*0.4 = 0.80
        c = conf("STRONG BUY", 0.4, technical_confidence=0.65)
        assert 0.75 <= c <= 0.85, f"Expected 75-85%, got {c}"

    def test_technical_confidence_none_falls_back_to_sentiment(self):
        """When technical_confidence is None, the result should match the
        sentiment-only calculation (backward compatibility)."""
        with_none = conf("WEAK BUY", 0.4, technical_confidence=None)
        without = conf("WEAK BUY", 0.4)
        assert with_none == without

    def test_weak_buy_moderate_sentiment_strong_technical_hits_target(self):
        """Core scenario: sentiment avg_score ~0.35 (just above BUY_THRESHOLD),
        technical agent returned BUY but combined is WEAK BUY (because tech=HOLD
        for this scenario).  Technical adjusted_confidence = 0.5.
        The result should be in the 40-60% range."""
        c = conf("WEAK BUY", 0.35, technical_confidence=0.5)
        assert 0.40 <= c <= 0.60

    def test_technical_does_not_affect_hold(self):
        """HOLD confidence is fixed at 0.25 regardless of technical confidence."""
        c = conf("HOLD", 0.5, technical_confidence=0.8)
        assert c == 0.25

    def test_technical_does_not_affect_conflicting(self):
        """CONFLICTING confidence is fixed at 0.10 regardless."""
        c = conf("CONFLICTING", 0.5, technical_confidence=0.8)
        assert c == 0.10

    def test_higher_technical_confidence_increases_weak_signal(self):
        """Monotonicity: higher technical confidence -> higher confidence."""
        low = conf("WEAK BUY", 0.3, technical_confidence=0.25)
        high = conf("WEAK BUY", 0.3, technical_confidence=0.60)
        assert high > low

    def test_higher_technical_confidence_increases_strong_signal(self):
        """Monotonicity: higher technical confidence -> higher confidence."""
        low = conf("STRONG BUY", 0.3, technical_confidence=0.25)
        high = conf("STRONG BUY", 0.3, technical_confidence=0.60)
        assert high > low

    def test_weak_signal_never_exceeds_0_6_without_volume(self):
        """WEAK signals are capped at 0.60 before volume adjustment."""
        c = conf("WEAK BUY", 1.0, technical_confidence=1.0)
        assert c == 0.60

    def test_weak_signal_with_volume_can_exceed_0_6(self):
        """Volume boost on top of max WEAK should produce 0.70."""
        c = conf("WEAK BUY", 1.0, technical_confidence=1.0, volume_confirmed=True)
        assert c == 0.70


# ===========================================================================
# End-to-end: bullish setup produces BUY with 40-60% confidence
# ===========================================================================

class TestBullishSetupConfidence:
    """Verify that a realistic bullish market scenario produces the expected
    confidence level when sentiment and technical signals are combined."""

    def test_typical_bullish_trend_produces_40_to_60_confidence(self):
        """
        Scenario: normal trending market.
        - Sentiment: avg_score = 0.35 (modestly bullish, above BUY threshold).
        - Technical: RSI=45 (mildly oversold, +0.5), MACD histogram positive (+1),
          bullish SMA alignment (+1) -> score 2.5 -> BUY signal.
          Technical adjusted_confidence = 0.5 (base for BUY).

        Combined: BUY + BUY = STRONG BUY.
        Confidence = 0.6 + blended*0.4 where blended = 0.35*0.6 + 0.5*0.4 = 0.41.
        Result = 0.6 + 0.41*0.4 = 0.764 -> ~76%.

        When technical is HOLD (more common in mild trends):
        Combined: BUY + HOLD = WEAK BUY.
        Confidence = 0.35 + blended*0.25 = 0.35 + 0.41*0.25 = 0.4525 -> 45%.
        """
        # WEAK BUY case (most common: sentiment BUY + technical HOLD)
        c_weak = conf("WEAK BUY", 0.35, technical_confidence=0.5)
        assert 0.40 <= c_weak <= 0.60, (
            f"WEAK BUY with moderate sentiment + technical should be 40-60%, got {c_weak}"
        )

        # STRONG BUY case (both agree)
        c_strong = conf("STRONG BUY", 0.35, technical_confidence=0.5)
        assert c_strong > c_weak, "STRONG BUY should always beat WEAK BUY"
        assert c_strong >= 0.60, f"STRONG BUY should be at least 60%, got {c_strong}"

    def test_end_to_end_signal_rules_plus_confidence(self):
        """
        Full integration: build indicators -> apply signal rules -> compute
        combined confidence.  Verifies the entire chain from raw indicators
        to a final 40-60% confidence number.
        """
        from agents.technical_agent import TechnicalAgent

        # Known bullish indicators
        ind = {
            "rsi": 45.0,                 # mildly oversold -> +0.5
            "macd_bull_cross": True,      # bullish crossover -> +2
            "macd_bear_cross": False,
            "macd_hist": 0.5,            # positive momentum -> +1
            "price": 160.0,
            "bb_lower": 148.0,
            "bb_upper": 172.0,
            "sma_20": 158.0,             # bullish alignment -> +1
            "sma_50": 154.0,
            "golden_cross_recent": False,
            "death_cross_recent": False,
        }

        signal, reasons = TechnicalAgent._apply_signal_rules(ind)
        assert signal == "BUY", f"Expected BUY, got {signal}; reasons: {reasons}"

        # Compute technical confidence
        tech_conf, _ = TechnicalAgent._apply_confidence_adjustments(signal, {
            **ind,
            "adx": None, "sma_200": None, "price": 160.0,
            "trend_strength": None,
            "bull_flag_breakout": False,
            "wedge_type": None, "wedge_breakout": False,
            "death_cross_recent": False,
        })
        assert tech_conf == 0.5, f"Base tech confidence for BUY should be 0.5, got {tech_conf}"

        # Combine: sentiment BUY (avg_score=0.35) + technical BUY -> STRONG BUY
        combined = combine("BUY", signal)
        assert combined == "STRONG BUY"

        c = conf(combined, 0.35, technical_confidence=tech_conf)
        assert c >= 0.60, f"STRONG BUY should be >= 60%, got {c}"

        # More common: sentiment BUY + technical HOLD -> WEAK BUY
        combined_weak = combine("BUY", "HOLD")
        c_weak = conf(combined_weak, 0.35, technical_confidence=tech_conf)
        assert 0.40 <= c_weak <= 0.60, (
            f"WEAK BUY with moderate sentiment + tech confidence 0.5 "
            f"should be 40-60%, got {c_weak}"
        )


# ===========================================================================
# Post-debate floor enforcement
# ===========================================================================

class TestDebateFloorEnforcement:
    """After a bull/bear debate penalty, confidence must not drop below the
    signal-type minimum floor.  This mirrors the inline enforcement in
    Coordinator.run() that clamps ``conf`` after overwriting it with
    ``debate_result.adjusted_confidence``."""

    # The floor map used in coordinator.py after the debate block:
    _SIGNAL_FLOORS = {
        "STRONG BUY": 0.60, "STRONG SELL": 0.60,
        "WEAK BUY": 0.35, "WEAK SELL": 0.35,
        "HOLD": 0.25,
        "CONFLICTING": 0.10,
    }

    @staticmethod
    def _apply_floor(signal: str, debate_confidence: float) -> float:
        """Replicate the coordinator's post-debate floor enforcement."""
        floors = {
            "STRONG BUY": 0.60, "STRONG SELL": 0.60,
            "WEAK BUY": 0.35, "WEAK SELL": 0.35,
            "HOLD": 0.25,
            "CONFLICTING": 0.10,
        }
        floor = floors.get(signal, 0.0)
        return max(debate_confidence, floor)

    def test_weak_buy_never_below_035(self):
        """Even with a harsh debate penalty, WEAK BUY stays >= 0.35."""
        assert self._apply_floor("WEAK BUY", 0.15) == 0.35
        assert self._apply_floor("WEAK BUY", 0.0) == 0.35
        assert self._apply_floor("WEAK BUY", 0.34) == 0.35

    def test_weak_buy_above_floor_unchanged(self):
        """If debate confidence is already above the floor, it stays."""
        assert self._apply_floor("WEAK BUY", 0.45) == 0.45

    def test_weak_sell_never_below_035(self):
        assert self._apply_floor("WEAK SELL", 0.10) == 0.35

    def test_strong_buy_never_below_060(self):
        """STRONG BUY must stay >= 0.60 after debate."""
        assert self._apply_floor("STRONG BUY", 0.40) == 0.60
        assert self._apply_floor("STRONG BUY", 0.59) == 0.60

    def test_strong_buy_above_floor_unchanged(self):
        assert self._apply_floor("STRONG BUY", 0.75) == 0.75

    def test_strong_sell_never_below_060(self):
        assert self._apply_floor("STRONG SELL", 0.30) == 0.60

    def test_hold_stays_at_025(self):
        """HOLD floor is 0.25."""
        assert self._apply_floor("HOLD", 0.20) == 0.25
        assert self._apply_floor("HOLD", 0.25) == 0.25

    def test_conflicting_stays_at_010(self):
        """CONFLICTING floor is 0.10."""
        assert self._apply_floor("CONFLICTING", 0.05) == 0.10
        assert self._apply_floor("CONFLICTING", 0.10) == 0.10

    def test_weak_buy_confidence_never_below_030(self):
        """WEAK BUY confidence is never below 0.30 (it's actually >= 0.35)."""
        for debate_conf in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.29]:
            result = self._apply_floor("WEAK BUY", debate_conf)
            assert result >= 0.30, f"WEAK BUY conf {result} < 0.30 when debate gave {debate_conf}"

    def test_strong_buy_confidence_never_below_055(self):
        """STRONG BUY confidence is never below 0.55 (it's actually >= 0.60)."""
        for debate_conf in [0.0, 0.10, 0.30, 0.50, 0.54]:
            result = self._apply_floor("STRONG BUY", debate_conf)
            assert result >= 0.55, f"STRONG BUY conf {result} < 0.55 when debate gave {debate_conf}"


# ===========================================================================
# Session propagation
# ===========================================================================

class TestSessionPropagation:
    """Verify that a session name passed to _log_signal_event and
    _log_strategy_result reaches the SignalLogger."""

    def test_session_passed_to_signal_logger_via_log_signal_event(self):
        """When session='US_OPEN' is passed to _log_signal_event, the
        SignalLogger.log() call should receive session='US_OPEN'."""
        from unittest.mock import MagicMock, patch

        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock()

        fake_result = {
            "ticker": "AAPL",
            "combined_signal": "STRONG BUY",
            "confidence": 0.85,
            "technical": {
                "indicators": {"price": 180.0, "rsi": 45.0, "sma_50": 175.0, "rvol": 1.5},
            },
            "sentiment": {"avg_score": 0.6, "source_breakdown": {}},
            "debate": None,
            "execution": {},
        }

        coordinator._log_signal_event(fake_result, session="US_OPEN")

        coordinator.signal_logger.log.assert_called_once()
        logged = coordinator.signal_logger.log.call_args[0][0]
        assert logged["session"] == "US_OPEN"

    def test_session_passed_to_signal_logger_via_log_strategy_result(self):
        """When session='MIDDAY' is passed to _log_strategy_result, the
        SignalLogger.log() call should receive session='MIDDAY'."""
        from unittest.mock import MagicMock

        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock()

        fake_strategy = MagicMock()
        fake_strategy.strategy_name = "TrendFollowing"
        fake_strategy.signal = "BUY"
        fake_strategy.confidence = 75.0
        fake_strategy.indicators = {"price": 180.0, "rsi": 45.0, "sma_50": 175.0}

        coordinator._log_strategy_result("AAPL", fake_strategy, session="MIDDAY")

        coordinator.signal_logger.log.assert_called_once()
        logged = coordinator.signal_logger.log.call_args[0][0]
        assert logged["session"] == "MIDDAY"

    def test_session_defaults_to_none(self):
        """Without a session argument, session should be None (backward compat)."""
        from unittest.mock import MagicMock

        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock()

        fake_result = {
            "ticker": "AAPL",
            "combined_signal": "HOLD",
            "confidence": 0.25,
            "technical": {"indicators": {}},
            "sentiment": {"avg_score": 0.0, "source_breakdown": {}},
            "debate": None,
            "execution": {},
        }

        coordinator._log_signal_event(fake_result)

        logged = coordinator.signal_logger.log.call_args[0][0]
        assert logged["session"] is None
