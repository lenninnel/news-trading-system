"""
Tests for ATR-based dynamic stop-loss and take-profit.

Covers:
  - Wilder ATR(14) on High/Low/Close from the daily_ohlc store,
    verified against a hand-computed reference case
  - True Range picks up overnight gaps (|H−C_prev| / |L−C_prev|)
  - No yfinance fallback: <15 bars in the store → None + loud WARNING
  - Wider stops for high-volatility stocks, tighter for low-vol
  - Minimum R:R ratio enforced
  - Position size scales inversely with ATR
  - Fallback to fixed stops when ATR unavailable
  - 1% stop-distance floor still fires on near-zero vol (Q-012 Familie 1)
"""

import logging
import os
import sys
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.risk_agent import RiskAgent


# ── Helpers ─────────────────────────────────────────────────────────

def make_bars(n: int = 30, close: float = 100.0, tr: float = 2.0) -> list[dict]:
    """n daily_ohlc rows with constant close and H/L straddling it, so every
    True Range equals `tr` exactly (H−L = tr dominates both gap terms) and
    Wilder ATR(14) == tr regardless of smoothing length.

    adj_close is deliberately wrong (close/2): the ATR must use raw close
    (Fix D3 store convention), so a correct result proves adj_close is unread.
    """
    start = date(2026, 7, 1)
    return [
        {
            "ticker": "TEST",
            "date": (start + timedelta(days=i)).isoformat(),
            "open": close,
            "high": close + tr / 2,
            "low": close - tr / 2,
            "close": close,
            "adj_close": close / 2,
            "volume": 1_000,
            "source": "test",
            "quality_flag": None,
        }
        for i in range(n)
    ]


def make_agent(bars: list[dict] | None = None, bars_by_ticker: dict | None = None) -> RiskAgent:
    db = MagicMock()
    db.log_risk_calculation.return_value = 99
    if bars_by_ticker is not None:
        db.get_daily_ohlc.side_effect = (
            lambda ticker, start, end: bars_by_ticker.get(ticker, [])
        )
    else:
        db.get_daily_ohlc.return_value = bars if bars is not None else []
    return RiskAgent(db=db)


# ── Wilder ATR reference cases ──────────────────────────────────────


class TestWilderATRReference:
    def test_hand_computed_wilder_atr(self):
        """16 bars with TR_i = i (i = 1..15), hand-computed:

        seed  = mean(TR_1..TR_14) = (1+2+...+14)/14 = 105/14 = 7.5
        final = (7.5·13 + 15)/14  = 112.5/14        = 8.0357142857...
        """
        bars = [{"date": "2026-07-01", "high": 100.0, "low": 100.0, "close": 100.0}]
        for i in range(1, 16):
            bars.append({
                "date": f"2026-07-{i + 1:02d}",
                "high": 100.0 + i / 2,
                "low": 100.0 - i / 2,
                "close": 100.0,
            })
        agent = make_agent(bars=bars)

        atr = agent._fetch_atr("TEST")

        assert atr == pytest.approx(112.5 / 14, abs=1e-9)

    def test_true_range_picks_up_gap(self):
        """A gap-up bar's TR is |H−C_prev|, not the intraday H−L.

        14 bars TR=2 seed the ATR at 2.0; the 15th bar gaps to H=112/L=110
        against C_prev=100 → TR = max(2, 12, 10) = 12:

        final = (2.0·13 + 12)/14 = 38/14 = 2.7142857...
        """
        bars = make_bars(n=15, close=100.0, tr=2.0)
        bars.append({
            "date": "2026-08-01",
            "high": 112.0, "low": 110.0, "close": 111.0,
        })
        agent = make_agent(bars=bars)

        atr = agent._fetch_atr("TEST")

        assert atr == pytest.approx(38 / 14, abs=1e-9)

    def test_constant_tr_series_returns_tr(self):
        """Constant-TR bars → ATR equals that TR exactly at any length."""
        agent = make_agent(bars=make_bars(n=40, close=250.0, tr=3.7))
        assert agent._fetch_atr("TEST") == pytest.approx(3.7, abs=1e-9)


class TestStoreOnlyNoFallback:
    def test_below_15_bars_returns_none_and_warns(self, caplog):
        agent = make_agent(bars=make_bars(n=14))
        with caplog.at_level(logging.WARNING, logger="agents.risk_agent"):
            atr = agent._fetch_atr("TEST")
        assert atr is None
        assert any("ATR unavailable" in r.getMessage() for r in caplog.records)

    def test_empty_store_never_calls_yfinance(self, caplog):
        """The yfinance fallback is gone: an empty store yields None without
        any network fetch."""
        agent = make_agent(bars=[])
        with patch("agents.risk_agent.yf") as yf_mock, \
             caplog.at_level(logging.WARNING, logger="agents.risk_agent"):
            atr = agent._fetch_atr("ZZZZZ")
        assert atr is None
        yf_mock.download.assert_not_called()
        assert any("ATR unavailable" in r.getMessage() for r in caplog.records)

    def test_null_high_low_returns_none_and_warns(self, caplog):
        bars = make_bars(n=20)
        bars[10]["high"] = None
        agent = make_agent(bars=bars)
        with caplog.at_level(logging.WARNING, logger="agents.risk_agent"):
            atr = agent._fetch_atr("TEST")
        assert atr is None
        assert any("NULL" in r.getMessage() for r in caplog.records)

    def test_db_read_failure_returns_none_and_warns(self, caplog):
        agent = make_agent()
        agent._db.get_daily_ohlc.side_effect = RuntimeError("db locked")
        with caplog.at_level(logging.WARNING, logger="agents.risk_agent"):
            atr = agent._fetch_atr("TEST")
        assert atr is None
        assert any("daily_ohlc read failed" in r.getMessage() for r in caplog.records)


# ── Stop/TP behaviour on top of the ATR ─────────────────────────────


class TestATRStopWiderForHighVol:
    def test_high_vol_gets_wider_stop(self):
        """TSLA-like stock (~4% daily range on $350) → wide stop distance."""
        entry = 350.0
        agent = make_agent(bars=make_bars(close=entry, tr=0.04 * entry))

        result = agent.calculate_atr_stops("TSLA", entry, direction="BUY")

        assert result["atr_available"] is True
        assert result["stop_distance"] > 5.0
        assert result["stop_loss"] < entry - 5.0

    def test_stop_distance_proportional_to_atr(self):
        """Stop distance should be ~1.5x ATR."""
        entry = 200.0
        agent = make_agent(bars=make_bars(close=entry, tr=0.03 * entry))

        result = agent.calculate_atr_stops(
            "ENPH", entry, direction="BUY", atr_stop_multiplier=1.5,
        )

        expected_stop = result["atr"] * 1.5
        assert abs(result["stop_distance"] - expected_stop) < 0.01


class TestATRStopTighterForLowVol:
    def test_low_vol_gets_tighter_stop(self):
        """JPM-like stock (~0.8% daily range on $220) → tight stop."""
        entry = 220.0
        agent = make_agent(bars=make_bars(close=entry, tr=0.008 * entry))

        result = agent.calculate_atr_stops("JPM", entry, direction="BUY")

        assert result["atr_available"] is True
        assert result["stop_distance"] < 8.0

    def test_low_vol_narrower_than_high_vol(self):
        agent = make_agent(bars_by_ticker={
            "JPM": make_bars(close=200.0, tr=0.008 * 200.0),
            "TSLA": make_bars(close=200.0, tr=0.04 * 200.0),
        })

        r_low = agent.calculate_atr_stops("JPM", 200.0)
        r_high = agent.calculate_atr_stops("TSLA", 200.0)

        assert r_low["stop_distance"] < r_high["stop_distance"]


class TestMinimumRRRatioEnforced:
    def test_rr_ratio_at_least_2(self):
        """R:R should always be >= 2.0 even if ATR multipliers give less."""
        agent = make_agent(bars=make_bars(close=100.0, tr=2.0))

        # 1.5 stop, 2.5 TP — that's only 1.67:1, code enforces min 2:1
        result = agent.calculate_atr_stops(
            "TEST", 100.0, direction="BUY",
            atr_stop_multiplier=1.5, atr_tp_multiplier=2.5,
        )

        assert result["rr_ratio"] >= 2.0

    def test_rr_ratio_with_default_multipliers(self):
        """Default 1.5 stop / 3.0 TP gives exactly 2:1 R:R."""
        agent = make_agent(bars=make_bars(close=100.0, tr=2.0))

        result = agent.calculate_atr_stops(
            "TEST", 100.0, direction="BUY",
            atr_stop_multiplier=1.5, atr_tp_multiplier=3.0,
        )

        assert result["rr_ratio"] == 2.0


class TestPositionSizeScalesWithATR:
    def test_high_vol_smaller_position(self):
        """High-vol stock should get fewer shares (same risk budget)."""
        agent = make_agent(bars_by_ticker={
            "JPM": make_bars(close=100.0, tr=0.008 * 100.0),
            "TSLA": make_bars(close=100.0, tr=0.04 * 100.0),
        })

        r_low = agent.calculate_atr_stops("JPM", 100.0, account_balance=10_000.0)
        r_high = agent.calculate_atr_stops("TSLA", 100.0, account_balance=10_000.0)

        assert r_high["shares"] < r_low["shares"]

    def test_position_respects_risk_budget(self):
        """Position risk should not exceed account_risk_pct of account."""
        agent = make_agent(bars=make_bars(close=100.0, tr=2.0))

        result = agent.calculate_atr_stops(
            "TEST", 100.0, account_balance=10_000.0, account_risk_pct=0.01,
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
        agent = make_agent(bars=make_bars(close=100.0, tr=2.0))

        result = agent.calculate_atr_stops("TEST", 100.0, direction="SELL")

        assert result["stop_loss"] > 100.0
        assert result["take_profit"] < 100.0


# ===========================================================================
# ATR stop-distance floor (Freeze-Lift Fix 3/4, Q-012 Familie 1)
# ===========================================================================


class TestATRStopFloor:
    """k = 1.0% floor lifts only the degenerate near-zero stops (Q-012 F1)."""

    _MULT = 1.5  # atr_stop_multiplier, passed explicitly for determinism

    def test_flat_tape_floors_stop_to_one_percent_buy(self):
        """Near-zero ATR (flat tape) → stop distance floored to exactly 1.0%."""
        entry = 100.0
        # TR = 0.01% of entry → raw stop distance = 0.015% (degenerate).
        agent = make_agent(bars=make_bars(close=entry, tr=0.0001 * entry))
        r = agent.calculate_atr_stops(
            "XOM", entry, direction="BUY", atr_stop_multiplier=self._MULT,
        )
        assert r["atr_available"] is True
        assert r["stop_distance"] == pytest.approx(0.01 * entry, abs=1e-6)  # 1.0%
        assert r["stop_loss"] == pytest.approx(entry * 0.99, abs=1e-4)      # 99.00
        stop_pct = (entry - r["stop_loss"]) / entry
        assert stop_pct == pytest.approx(0.01, abs=1e-6)

    def test_flat_tape_floors_stop_to_one_percent_sell(self):
        """Floor is direction-agnostic — SELL stop floored to +1.0% above entry."""
        entry = 100.0
        agent = make_agent(bars=make_bars(close=entry, tr=0.0001 * entry))
        r = agent.calculate_atr_stops(
            "XOM", entry, direction="SELL", atr_stop_multiplier=self._MULT,
        )
        assert r["stop_distance"] == pytest.approx(0.01 * entry, abs=1e-6)
        assert r["stop_loss"] == pytest.approx(entry * 1.01, abs=1e-4)      # 101.00
        stop_pct = (r["stop_loss"] - entry) / entry
        assert stop_pct == pytest.approx(0.01, abs=1e-6)

    def test_healthy_stop_untouched_by_floor(self):
        """A healthy ATR (1.5% TR → 2.25% stop) is NOT touched by the floor."""
        entry = 100.0
        agent = make_agent(bars=make_bars(close=entry, tr=0.015 * entry))
        r = agent.calculate_atr_stops(
            "TEST", entry, direction="BUY", atr_stop_multiplier=self._MULT,
        )
        # Raw ATR stop distance = 1.5 * 100 * 0.015 = 2.25 (> 1.0% floor).
        assert r["stop_distance"] == pytest.approx(2.25, abs=1e-3)
        stop_pct = (entry - r["stop_loss"]) / entry
        assert stop_pct == pytest.approx(0.0225, abs=1e-4)  # unchanged 2.25%

    def test_boundary_atr_equals_floor_no_change(self):
        """When the ATR stop distance equals the floor exactly, nothing changes."""
        entry = 100.0
        # tr * mult = 1.0 → tr = 1.0 / 1.5. Raw stop == floor.
        agent = make_agent(bars=make_bars(close=entry, tr=0.01 * entry / self._MULT))
        r = agent.calculate_atr_stops(
            "TEST", entry, direction="BUY", atr_stop_multiplier=self._MULT,
        )
        assert r["stop_distance"] == pytest.approx(0.01 * entry, abs=1e-3)
        stop_pct = (entry - r["stop_loss"]) / entry
        assert stop_pct == pytest.approx(0.01, abs=1e-4)

    def test_floor_does_not_change_rr_or_tp(self):
        """tp_distance and the 2:1 RR minimum are untouched; RR stays >= 2:1 and
        TP is measured off ATR, not the floored stop."""
        entry = 100.0
        agent = make_agent(bars=make_bars(close=entry, tr=0.0001 * entry))
        r = agent.calculate_atr_stops(
            "XOM", entry, direction="BUY",
            atr_stop_multiplier=self._MULT, atr_tp_multiplier=3.0,
        )
        # RR recomputes against the floored stop and must remain >= 2:1.
        assert r["rr_ratio"] >= 2.0
        # tp_distance was raised to 2× the floored stop by the existing 2:1 rule
        # (raw ATR tp ≈ 0 here).
        assert r["tp_distance"] == pytest.approx(2.0 * r["stop_distance"], abs=1e-3)


# ── REPLAY: the real Q-012 Familie-1 near-zero-stop cases ───────────────────
#
# The four documented Combined entries whose (former close-to-close) ATR
# collapsed the stop onto the entry. Each is reproduced by seeding the store
# with bars whose Wilder ATR yields the recorded near-zero stop %, then re-run
# through calculate_atr_stops to show the floor lifts it to exactly 1.0%.
_FAMILY1_CASES = [
    # (ticker, date,        recorded_stop_pct, entry_price)
    ("XOM", "2026-05-26", 0.0004, 105.00),   # -0.04%
    ("XOM", "2026-05-27", 0.0028, 105.00),   # -0.28%
    ("CVX", "2026-05-27", 0.0020, 155.00),   # -0.20%
    ("JPM", "2026-05-27", 0.0002, 265.00),   # -0.02%
]
_MULT = 1.5


class TestFamily1Replay:

    @pytest.mark.parametrize("ticker,date_,rec_pct,entry", _FAMILY1_CASES)
    def test_each_cripple_lifted_to_one_percent(self, ticker, date_, rec_pct, entry):
        # Seed bars whose ATR (raw) reproduces the recorded near-zero stop %.
        tr = rec_pct * entry / _MULT
        agent = make_agent(bars=make_bars(close=entry, tr=tr))

        # BEFORE: the raw ATR stop distance — near-zero.
        atr = agent._fetch_atr(ticker)
        raw_stop_dist = atr * _MULT
        before_pct = raw_stop_dist / entry
        assert before_pct == pytest.approx(rec_pct, rel=1e-3)
        assert before_pct < 0.003  # all four are < 0.3% (degenerate)

        # AFTER: with the floor, every cripple sits at exactly 1.0%.
        r = agent.calculate_atr_stops(
            ticker, entry, direction="BUY", atr_stop_multiplier=_MULT,
        )
        after_pct = (entry - r["stop_loss"]) / entry
        assert after_pct == pytest.approx(0.01, abs=1e-6)
        assert r["stop_distance"] == pytest.approx(0.01 * entry, abs=1e-4)

    def test_healthy_control_p50_unchanged(self):
        """A healthy P50 stop (~2.28%) is NOT touched by the floor."""
        entry = 100.0
        agent = make_agent(bars=make_bars(close=entry, tr=0.0228 * entry / _MULT))
        r = agent.calculate_atr_stops(
            "CTRL", entry, direction="BUY", atr_stop_multiplier=_MULT,
        )
        after_pct = (entry - r["stop_loss"]) / entry
        assert after_pct == pytest.approx(0.0228, abs=1e-4)  # unchanged

    def test_replay_summary(self, capsys):
        """Emit the before/after evidence for the four Familie-1 cripples."""
        lines = []
        for ticker, date_, rec_pct, entry in _FAMILY1_CASES:
            agent = make_agent(
                bars=make_bars(close=entry, tr=rec_pct * entry / _MULT),
            )
            r = agent.calculate_atr_stops(
                ticker, entry, direction="BUY", atr_stop_multiplier=_MULT,
            )
            before = rec_pct * 100
            after = (entry - r["stop_loss"]) / entry * 100
            lines.append(
                f"  {ticker} {date_}: before -{before:.2f}%  →  after -{after:.2f}% "
                f"(stop ${entry:.2f} → ${r['stop_loss']:.2f})"
            )
        # All four lifted to exactly 1.00%.
        assert all("after -1.00%" in ln for ln in lines)
        print("\nQ-012 Familie-1 ATR-floor replay (near-zero stop → 1.0% floor):")
        for ln in lines:
            print(ln)
        print(f"  Floor lifted {len(lines)}/{len(_FAMILY1_CASES)} degenerate stops "
              f"to exactly 1.0%; healthy stops (>= P5 1.08%) untouched.")
