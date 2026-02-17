"""
Unit tests for RiskAgent — signal parsing, Kelly Criterion, and sizing logic.

All tests are offline: the DB is patched to a no-op so no file I/O occurs.

Run with:
    python3 -m pytest tests/ -v
"""

import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from agents.risk_agent import RiskAgent, _parse_signal, _kelly_fraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent() -> RiskAgent:
    """Return a RiskAgent with a stubbed-out database."""
    db = MagicMock()
    db.log_risk_calculation.return_value = 99   # fake primary key
    return RiskAgent(db=db)


def run(agent, **overrides) -> dict:
    """Call agent.run() with sensible defaults, overriding specific fields."""
    defaults = dict(
        ticker="TEST",
        signal="STRONG BUY",
        confidence=75.0,
        current_price=100.0,
        account_balance=10_000.0,
    )
    defaults.update(overrides)
    return agent.run(**defaults)


# ===========================================================================
# _parse_signal — signal string parsing
# ===========================================================================

class TestParseSignal:
    def test_strong_buy(self):
        assert _parse_signal("STRONG BUY") == ("BUY", "STRONG")

    def test_weak_buy(self):
        assert _parse_signal("WEAK BUY") == ("BUY", "WEAK")

    def test_raw_buy_treated_as_strong(self):
        assert _parse_signal("BUY") == ("BUY", "STRONG")

    def test_strong_sell(self):
        assert _parse_signal("STRONG SELL") == ("SELL", "STRONG")

    def test_weak_sell(self):
        assert _parse_signal("WEAK SELL") == ("SELL", "WEAK")

    def test_raw_sell_treated_as_strong(self):
        assert _parse_signal("SELL") == ("SELL", "STRONG")

    def test_hold(self):
        assert _parse_signal("HOLD") == ("HOLD", "NONE")

    def test_conflicting(self):
        assert _parse_signal("CONFLICTING") == ("HOLD", "NONE")

    def test_case_insensitive(self):
        assert _parse_signal("strong buy") == ("BUY", "STRONG")
        assert _parse_signal("Weak Sell")  == ("SELL", "WEAK")


# ===========================================================================
# _kelly_fraction — Kelly Criterion maths
# ===========================================================================

class TestKellyFraction:
    def test_zero_confidence_gives_positive_fraction(self):
        # p=0.50, b=2: kelly=(1.0-0.5)/2=0.25, half=0.125
        f = _kelly_fraction(0)
        assert f > 0

    def test_100_confidence_gives_larger_fraction(self):
        assert _kelly_fraction(100) > _kelly_fraction(50)

    def test_fraction_always_non_negative(self):
        for conf in range(0, 101, 10):
            assert _kelly_fraction(conf) >= 0

    def test_fraction_never_exceeds_1(self):
        for conf in range(0, 101, 10):
            assert _kelly_fraction(conf) <= 1.0

    def test_monotonically_increasing(self):
        fracs = [_kelly_fraction(c) for c in range(0, 101, 10)]
        assert fracs == sorted(fracs)


# ===========================================================================
# Safety gates
# ===========================================================================

class TestSafetyGates:
    def test_confidence_below_30_skips(self):
        agent = make_agent()
        result = run(agent, confidence=29.9)
        assert result["skipped"] is True
        assert "30" in result["skip_reason"]

    def test_confidence_exactly_30_does_not_skip(self):
        agent = make_agent()
        result = run(agent, confidence=30.0)
        assert result["skipped"] is False

    def test_hold_signal_skips(self):
        agent = make_agent()
        result = run(agent, signal="HOLD")
        assert result["skipped"] is True

    def test_conflicting_signal_skips(self):
        agent = make_agent()
        result = run(agent, signal="CONFLICTING")
        assert result["skipped"] is True

    def test_zero_shares_skips(self):
        """Price so high that no whole share fits in the position budget."""
        agent = make_agent()
        result = run(agent, current_price=999_999.0, account_balance=100.0)
        assert result["skipped"] is True

    def test_skipped_result_has_zero_position(self):
        agent = make_agent()
        result = run(agent, signal="HOLD")
        assert result["position_size_usd"] == 0.0
        assert result["shares"] == 0
        assert result["stop_loss"] is None
        assert result["take_profit"] is None
        assert result["risk_amount"] == 0.0


# ===========================================================================
# Position sizing constraints
# ===========================================================================

class TestPositionSizing:
    def test_position_capped_at_10_pct_of_balance(self):
        agent = make_agent()
        result = run(agent, confidence=100.0, account_balance=10_000.0)
        assert result["position_size_usd"] <= 10_000.0 * 0.10 + 1  # +1 for rounding

    def test_risk_amount_never_exceeds_2_pct_of_balance(self):
        agent = make_agent()
        for conf in (30, 50, 75, 100):
            result = run(agent, confidence=float(conf), account_balance=10_000.0)
            if not result["skipped"]:
                assert result["risk_amount"] <= 10_000.0 * 0.02 + 0.01  # +0.01 float tolerance

    def test_position_size_is_whole_shares_times_price(self):
        agent = make_agent()
        result = run(agent, current_price=50.0)
        assert result["position_size_usd"] == pytest.approx(result["shares"] * 50.0)

    def test_shares_is_integer(self):
        agent = make_agent()
        result = run(agent)
        assert isinstance(result["shares"], int)

    def test_higher_confidence_does_not_decrease_position(self):
        """Kelly grows with confidence → position should be equal or larger (up to cap)."""
        agent = make_agent()
        low  = run(agent, confidence=30.0)["position_size_usd"]
        high = run(agent, confidence=80.0)["position_size_usd"]
        assert high >= low


# ===========================================================================
# Stop-loss and take-profit
# ===========================================================================

class TestStopLossAndTakeProfit:
    def test_strong_buy_stop_loss_is_2pct_below_price(self):
        agent = make_agent()
        result = run(agent, signal="STRONG BUY", current_price=100.0)
        assert result["stop_loss"] == pytest.approx(98.0, rel=1e-4)

    def test_weak_buy_stop_loss_is_1pct_below_price(self):
        agent = make_agent()
        result = run(agent, signal="WEAK BUY", current_price=100.0)
        assert result["stop_loss"] == pytest.approx(99.0, rel=1e-4)

    def test_strong_sell_stop_loss_is_2pct_above_price(self):
        agent = make_agent()
        result = run(agent, signal="STRONG SELL", current_price=100.0)
        assert result["stop_loss"] == pytest.approx(102.0, rel=1e-4)

    def test_weak_sell_stop_loss_is_1pct_above_price(self):
        agent = make_agent()
        result = run(agent, signal="WEAK SELL", current_price=100.0)
        assert result["stop_loss"] == pytest.approx(101.0, rel=1e-4)

    def test_take_profit_is_2x_stop_distance_buy(self):
        agent = make_agent()
        result = run(agent, signal="STRONG BUY", current_price=100.0)
        # stop = 2%, TP = 4% above
        assert result["take_profit"] == pytest.approx(104.0, rel=1e-4)

    def test_take_profit_is_2x_stop_distance_sell(self):
        agent = make_agent()
        result = run(agent, signal="STRONG SELL", current_price=100.0)
        # stop = 2%, TP = 4% below
        assert result["take_profit"] == pytest.approx(96.0, rel=1e-4)

    def test_reward_risk_ratio_is_2_to_1(self):
        """Take-profit distance must be exactly 2× the stop-loss distance."""
        agent = make_agent()
        for sig in ("STRONG BUY", "WEAK BUY", "STRONG SELL", "WEAK SELL"):
            result = run(agent, signal=sig, current_price=200.0)
            if result["skipped"]:
                continue
            price = 200.0
            sl_dist = abs(price - result["stop_loss"])
            tp_dist = abs(price - result["take_profit"])
            assert tp_dist == pytest.approx(sl_dist * 2, rel=1e-4), sig


# ===========================================================================
# Return structure
# ===========================================================================

class TestReturnStructure:
    REQUIRED_KEYS = {
        "ticker", "signal", "direction", "position_size_usd", "shares",
        "stop_loss", "take_profit", "risk_amount", "kelly_fraction",
        "stop_pct", "skipped", "skip_reason", "calc_id",
    }

    def test_active_position_has_all_keys(self):
        agent = make_agent()
        result = run(agent)
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_skipped_position_has_all_keys(self):
        agent = make_agent()
        result = run(agent, signal="HOLD")
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_calc_id_is_set(self):
        agent = make_agent()
        result = run(agent)
        assert result["calc_id"] == 99   # matches mock return value
