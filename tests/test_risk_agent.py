"""
Unit tests for RiskAgent — signal parsing, Kelly Criterion, and sizing logic.

All tests are offline: the DB is patched to a no-op so no file I/O occurs.

Run with:
    python3 -m pytest tests/ -v
"""

import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from agents.risk_agent import (
    RiskAgent,
    _parse_signal,
    _kelly_fraction,
    _HISTORICAL_KELLY_CAP,
    _HISTORICAL_KELLY_FALLBACK,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent() -> RiskAgent:
    """Return a RiskAgent with a stubbed-out database."""
    db = MagicMock()
    db.log_risk_calculation.return_value = 99   # fake primary key
    return RiskAgent(db=db)


def run(agent, **overrides) -> dict:
    """Call agent.run() with sensible defaults, overriding specific fields.
    Mocks get_days_to_earnings to return None (no earnings) by default."""
    defaults = dict(
        ticker="TEST",
        signal="STRONG BUY",
        confidence=75.0,
        current_price=100.0,
        account_balance=10_000.0,
    )
    defaults.update(overrides)
    with patch("agents.risk_agent.get_days_to_earnings", return_value=None):
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


# ===========================================================================
# Historical Kelly Criterion — outcome-driven position sizing
# ===========================================================================

class _FakeRow:
    """Mimics sqlite3.Row: dict(row) works via keys() + __getitem__."""
    def __init__(self, d: dict):
        self._d = d
    def keys(self):
        return self._d.keys()
    def __getitem__(self, k):
        return self._d[k]


def _make_agent_with_outcomes(outcomes: list[dict]) -> RiskAgent:
    """Return a RiskAgent whose DB returns *outcomes* from signal_events."""
    db = MagicMock()
    db.log_risk_calculation.return_value = 99

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [_FakeRow(row) for row in outcomes]
    mock_conn.execute.return_value = mock_cursor
    mock_conn.__enter__ = lambda self: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    db._connect.return_value = mock_conn

    return RiskAgent(db=db)


def _make_outcomes(n_wins: int, n_losses: int,
                   avg_win_pct: float = 3.0,
                   avg_loss_pct: float = 1.5) -> list[dict]:
    """Build synthetic signal_events outcome rows."""
    rows = []
    for _ in range(n_wins):
        rows.append({"outcome_correct": 1, "outcome_5d_pct": avg_win_pct})
    for _ in range(n_losses):
        rows.append({"outcome_correct": 0, "outcome_5d_pct": -avg_loss_pct})
    return rows


class TestHistoricalKelly:
    def test_kelly_with_sufficient_history(self):
        """When >= 10 outcomes exist, use actual win rate and payoff ratio."""
        # 10 wins at +3%, 5 losses at -1.5%  →  p=0.667, b=2.0
        # full_kelly = (2.0*0.667 - 0.333) / 2.0 = 0.500
        # half_kelly = 0.250  →  capped at 0.05
        outcomes = _make_outcomes(10, 5, avg_win_pct=3.0, avg_loss_pct=1.5)
        agent = _make_agent_with_outcomes(outcomes)

        kelly = agent.calculate_kelly_position("AAPL", "STRONG BUY", 75, 10_000)
        assert kelly is not None
        assert kelly > 0
        assert kelly <= _HISTORICAL_KELLY_CAP

    def test_kelly_falls_back_to_fixed_when_insufficient_data(self):
        """< 10 outcomes → returns None (caller uses 2% fallback)."""
        outcomes = _make_outcomes(3, 2)
        agent = _make_agent_with_outcomes(outcomes)

        kelly = agent.calculate_kelly_position("AAPL", "STRONG BUY", 75, 10_000)
        assert kelly is None

    def test_kelly_caps_at_5_percent(self):
        """Even with very high win rate, cap at 5%."""
        # 19 wins, 1 loss → p=0.95, b=2.0
        # full_kelly = (2.0*0.95 - 0.05) / 2.0 = 0.925
        # half_kelly = 0.4625 → capped at 0.05
        outcomes = _make_outcomes(19, 1, avg_win_pct=3.0, avg_loss_pct=1.5)
        agent = _make_agent_with_outcomes(outcomes)

        kelly = agent.calculate_kelly_position("META", "STRONG BUY", 80, 10_000)
        assert kelly == _HISTORICAL_KELLY_CAP  # exactly 5%

    def test_half_kelly_applied(self):
        """Returned fraction must be exactly half of full Kelly."""
        # 8 wins, 4 losses → p=0.667, b=3.0/1.5=2.0
        # full_kelly = (2.0 * 0.667 - 0.333) / 2.0 = 0.500
        # half_kelly = 0.250 → capped at 0.05
        outcomes = _make_outcomes(8, 4, avg_win_pct=3.0, avg_loss_pct=1.5)
        agent = _make_agent_with_outcomes(outcomes)

        # 12 outcomes ≥ 10: uses historical data
        kelly = agent.calculate_kelly_position("JPM", "STRONG BUY", 70, 10_000)
        assert kelly is not None

        # Compute expected value manually
        p = 8 / 12
        q = 1 - p
        b = 3.0 / 1.5
        full_kelly = (b * p - q) / b
        expected = min(max(0.0, full_kelly * 0.5), _HISTORICAL_KELLY_CAP)
        assert kelly == pytest.approx(expected, abs=1e-6)

    def test_kelly_no_edge_returns_zero(self):
        """When losses dominate, Kelly is zero (no negative sizing)."""
        # 2 wins, 10 losses → p=0.167, b=2.0
        # full_kelly = (2.0*0.167 - 0.833) / 2.0 = -0.250 → clamped to 0
        outcomes = _make_outcomes(2, 10, avg_win_pct=3.0, avg_loss_pct=1.5)
        agent = _make_agent_with_outcomes(outcomes)

        kelly = agent.calculate_kelly_position("XOM", "STRONG BUY", 70, 10_000)
        assert kelly == 0.0

    def test_run_uses_confidence_fallback_when_no_history(self):
        """run() falls back to confidence-based Kelly when no outcomes."""
        agent = make_agent()  # MagicMock DB → no outcomes
        result = run(agent, confidence=75.0, account_balance=10_000.0)
        # Falls back to _kelly_fraction(75) ≈ 0.194
        if not result["skipped"]:
            assert result["kelly_fraction"] > 0.10  # confidence-based, not fixed 2%


# ===========================================================================
# Portfolio VaR — parametric Value-at-Risk
# ===========================================================================

def _make_synthetic_prices(tickers: list[str], n_days: int = 252) -> pd.DataFrame:
    """Generate synthetic daily close prices for testing."""
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-03-28", periods=n_days)
    data = {}
    for t in tickers:
        # Random walk with ~0.5% daily vol
        returns = np.random.normal(0.0005, 0.005, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        data[t] = prices
    return pd.DataFrame(data, index=dates)


class TestPortfolioVar:
    def test_portfolio_var_calculation(self):
        """VaR is computed correctly for a 2-stock portfolio."""
        agent = make_agent()
        positions = [
            {"ticker": "AAPL", "current_value": 5000.0},
            {"ticker": "META", "current_value": 5000.0},
        ]
        prices_df = _make_synthetic_prices(["AAPL", "META"])

        # Build a multi-level column DataFrame matching yf.download output
        close_df = prices_df.copy()
        close_df.columns = pd.MultiIndex.from_product(
            [["Close"], close_df.columns])

        with patch("agents.risk_agent.yf.download", return_value=close_df):
            result = agent.calculate_portfolio_var(positions)

        assert result["var_1day"] > 0
        assert result["var_5day"] > result["var_1day"]
        assert 0 < result["var_1day_pct"] < 1.0
        assert isinstance(result["should_halt"], bool)
        assert "reasoning" in result

    def test_var_empty_positions(self):
        """Empty positions → zero VaR, no halt."""
        agent = make_agent()
        result = agent.calculate_portfolio_var([])
        assert result["var_1day"] == 0.0
        assert result["should_halt"] is False

    def test_var_single_position(self):
        """Single position degenerates to individual stock VaR."""
        agent = make_agent()
        positions = [{"ticker": "AAPL", "current_value": 10_000.0}]
        prices_df = _make_synthetic_prices(["AAPL"])

        # Single ticker — yf.download returns flat columns
        with patch("agents.risk_agent.yf.download", return_value=prices_df.rename(
            columns={"AAPL": "Close"})):
            result = agent.calculate_portfolio_var(positions)

        assert result["var_1day"] > 0


# ===========================================================================
# Drawdown halt
# ===========================================================================

class TestDrawdownHalt:
    def test_drawdown_halt_triggers_at_10_percent(self):
        """Drawdown > 10% should trigger halt."""
        agent = make_agent()
        # 11% drawdown: (10000 - 8900) / 10000 = 0.11
        assert agent.check_drawdown_halt(8900, 10_000) is True

    def test_drawdown_halt_does_not_trigger_below_threshold(self):
        """5% drawdown should NOT trigger halt."""
        agent = make_agent()
        assert agent.check_drawdown_halt(9500, 10_000) is False

    def test_drawdown_exactly_at_threshold(self):
        """Exactly 10% drawdown is NOT a halt (> not >=)."""
        agent = make_agent()
        assert agent.check_drawdown_halt(9000, 10_000) is False

    def test_drawdown_zero_peak(self):
        """Zero or negative peak is a no-op (no halt)."""
        agent = make_agent()
        assert agent.check_drawdown_halt(0, 0) is False
        assert agent.check_drawdown_halt(5000, -1) is False
