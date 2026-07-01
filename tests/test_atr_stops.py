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


# ===========================================================================
# ATR stop-distance floor (Freeze-Lift Fix 3/4, Q-012 Familie 1)
# ===========================================================================

def _exact_atr_pct_series(entry: float, atr_pct: float, n: int = 30) -> pd.Series:
    """Deterministic close series whose ATR(14) = atr_pct * entry.

    Geometric ramp with constant per-bar return `atr_pct`, arranged so the LAST
    close equals `entry`. Then mean(|pct_change|, 14) == atr_pct exactly and
    _fetch_atr returns atr_pct * last_close == atr_pct * entry — no randomness,
    no market-data dependency.
    """
    idx = np.arange(n) - (n - 1)               # last exponent = 0 → last close = entry
    prices = entry * (1.0 + atr_pct) ** idx
    return pd.Series(prices, index=pd.bdate_range("2026-01-01", periods=n))


class TestATRStopFloor:
    """k = 1.0% floor lifts only the degenerate near-zero stops (Q-012 F1)."""

    _MULT = 1.5  # atr_stop_multiplier, passed explicitly for determinism

    def test_flat_tape_floors_stop_to_one_percent_buy(self):
        """Near-zero ATR (flat tape) → stop distance floored to exactly 1.0%."""
        agent = make_agent()
        entry = 100.0
        # atr_pct = 0.01% → raw stop distance = 0.015% of entry (degenerate).
        prices = _exact_atr_pct_series(entry, 0.0001)
        r = agent.calculate_atr_stops(
            "XOM", entry, direction="BUY", prices=prices,
            atr_stop_multiplier=self._MULT,
        )
        assert r["atr_available"] is True
        assert r["stop_distance"] == pytest.approx(0.01 * entry, abs=1e-6)  # 1.0%
        assert r["stop_loss"] == pytest.approx(entry * 0.99, abs=1e-4)      # 99.00
        # Downside protection restored: exactly 1.00% below entry.
        stop_pct = (entry - r["stop_loss"]) / entry
        assert stop_pct == pytest.approx(0.01, abs=1e-6)

    def test_flat_tape_floors_stop_to_one_percent_sell(self):
        """Floor is direction-agnostic — SELL stop floored to +1.0% above entry."""
        agent = make_agent()
        entry = 100.0
        prices = _exact_atr_pct_series(entry, 0.0001)
        r = agent.calculate_atr_stops(
            "XOM", entry, direction="SELL", prices=prices,
            atr_stop_multiplier=self._MULT,
        )
        assert r["stop_distance"] == pytest.approx(0.01 * entry, abs=1e-6)
        assert r["stop_loss"] == pytest.approx(entry * 1.01, abs=1e-4)      # 101.00
        stop_pct = (r["stop_loss"] - entry) / entry
        assert stop_pct == pytest.approx(0.01, abs=1e-6)

    def test_healthy_stop_untouched_by_floor(self):
        """A healthy ATR (atr_pct 1.5% → 2.25% stop) is NOT touched by the floor."""
        agent = make_agent()
        entry = 100.0
        prices = _exact_atr_pct_series(entry, 0.015)   # atr_pct = 1.5%
        r = agent.calculate_atr_stops(
            "TEST", entry, direction="BUY", prices=prices,
            atr_stop_multiplier=self._MULT,
        )
        # Raw ATR stop distance = 0.015 * 100 * 1.5 = 2.25 (> 1.0% floor).
        assert r["stop_distance"] == pytest.approx(2.25, abs=1e-3)
        stop_pct = (entry - r["stop_loss"]) / entry
        assert stop_pct == pytest.approx(0.0225, abs=1e-4)  # unchanged 2.25%

    def test_boundary_atr_equals_floor_no_change(self):
        """When the ATR stop distance equals the floor exactly, nothing changes."""
        agent = make_agent()
        entry = 100.0
        # atr_pct * mult = 0.01 → atr_pct = 0.01 / 1.5. Raw stop == floor.
        prices = _exact_atr_pct_series(entry, 0.01 / self._MULT)
        r = agent.calculate_atr_stops(
            "TEST", entry, direction="BUY", prices=prices,
            atr_stop_multiplier=self._MULT,
        )
        assert r["stop_distance"] == pytest.approx(0.01 * entry, abs=1e-3)
        stop_pct = (entry - r["stop_loss"]) / entry
        assert stop_pct == pytest.approx(0.01, abs=1e-4)

    def test_floor_does_not_change_rr_or_tp(self):
        """tp_distance and the 2:1 RR minimum are untouched; RR stays >= 2:1 and
        TP is measured off ATR, not the floored stop."""
        agent = make_agent()
        entry = 100.0
        prices = _exact_atr_pct_series(entry, 0.0001)   # degenerate → floor fires
        r = agent.calculate_atr_stops(
            "XOM", entry, direction="BUY", prices=prices,
            atr_stop_multiplier=self._MULT, atr_tp_multiplier=3.0,
        )
        # RR recomputes against the floored stop and must remain >= 2:1.
        assert r["rr_ratio"] >= 2.0
        # tp_distance was raised to 2× the floored stop by the existing 2:1 rule
        # (raw ATR tp ≈ 0 here); this is the unchanged line 505-506 behaviour.
        assert r["tp_distance"] == pytest.approx(2.0 * r["stop_distance"], abs=1e-3)


# ── REPLAY: the real Q-012 Familie-1 near-zero-stop cases ───────────────────
#
# The four documented Combined entries whose close-to-close ATR collapsed the
# stop onto the entry (recorded downside protection in parentheses). Each is
# reproduced by an ATR that yields the recorded near-zero stop %, then re-run
# through calculate_atr_stops to show the floor lifts it to exactly 1.0%.
#
# NOTE: the exact recorded entry fills live in R's Q-012 health check /
# production data (out of scope for this build+test task). The floor is
# percent-based (1.0% of entry), so the "after" is exactly 1.00% for ANY entry;
# representative era entries are used purely to render dollar stops.
_FAMILY1_CASES = [
    # (ticker, date,        recorded_stop_pct, entry_price)
    ("XOM", "2026-05-26", 0.0004, 105.00),   # -0.04%
    ("XOM", "2026-05-27", 0.0028, 105.00),   # -0.28%
    ("CVX", "2026-05-27", 0.0020, 155.00),   # -0.20%
    ("JPM", "2026-05-27", 0.0002, 265.00),   # -0.02%
]
_MULT = 1.5


class TestFamily1Replay:

    @pytest.mark.parametrize("ticker,date,rec_pct,entry", _FAMILY1_CASES)
    def test_each_cripple_lifted_to_one_percent(self, ticker, date, rec_pct, entry):
        agent = make_agent()
        # Build an ATR that (raw) reproduces the recorded near-zero stop %.
        atr_pct = rec_pct / _MULT
        prices = _exact_atr_pct_series(entry, atr_pct)

        # BEFORE: the raw ATR stop distance (what shipped) — near-zero.
        atr = agent._fetch_atr(ticker, prices)
        raw_stop_dist = atr * _MULT
        before_pct = raw_stop_dist / entry
        assert before_pct == pytest.approx(rec_pct, rel=1e-3)
        assert before_pct < 0.003  # all four are < 0.3% (degenerate)

        # AFTER: with the floor, every cripple sits at exactly 1.0%.
        r = agent.calculate_atr_stops(
            ticker, entry, direction="BUY", prices=prices,
            atr_stop_multiplier=_MULT,
        )
        after_pct = (entry - r["stop_loss"]) / entry
        assert after_pct == pytest.approx(0.01, abs=1e-6)
        assert r["stop_distance"] == pytest.approx(0.01 * entry, abs=1e-4)

    def test_healthy_control_p50_unchanged(self):
        """A healthy P50 stop (~2.28%) is NOT touched by the floor."""
        agent = make_agent()
        entry = 100.0
        prices = _exact_atr_pct_series(entry, 0.0228 / _MULT)  # → 2.28% raw stop
        r = agent.calculate_atr_stops(
            "CTRL", entry, direction="BUY", prices=prices,
            atr_stop_multiplier=_MULT,
        )
        after_pct = (entry - r["stop_loss"]) / entry
        assert after_pct == pytest.approx(0.0228, abs=1e-4)  # unchanged

    def test_replay_summary(self, capsys):
        """Emit the before/after evidence for the four Familie-1 cripples."""
        agent = make_agent()
        lines = []
        for ticker, date, rec_pct, entry in _FAMILY1_CASES:
            prices = _exact_atr_pct_series(entry, rec_pct / _MULT)
            r = agent.calculate_atr_stops(
                ticker, entry, direction="BUY", prices=prices,
                atr_stop_multiplier=_MULT,
            )
            before = rec_pct * 100
            after = (entry - r["stop_loss"]) / entry * 100
            lines.append(
                f"  {ticker} {date}: before -{before:.2f}%  →  after -{after:.2f}% "
                f"(stop ${entry:.2f} → ${r['stop_loss']:.2f})"
            )
        # All four lifted to exactly 1.00%.
        assert all("after -1.00%" in ln for ln in lines)
        print("\nQ-012 Familie-1 ATR-floor replay (near-zero stop → 1.0% floor):")
        for ln in lines:
            print(ln)
        print(f"  Floor lifted {len(lines)}/{len(_FAMILY1_CASES)} degenerate stops "
              f"to exactly 1.0%; healthy stops (>= P5 1.08%) untouched.")
