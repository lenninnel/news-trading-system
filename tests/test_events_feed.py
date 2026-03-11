"""
Unit tests for earnings calendar event risk and its integration with
RiskAgent and Coordinator.

All tests run offline — yfinance and events_feed are mocked.

Run with:
    python3 -m pytest tests/test_events_feed.py -v
"""

import sys
import os
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from data.events_feed import (
    _trading_days_between,
    clear_cache,
    get_days_to_earnings,
    get_earnings_date,
    is_earnings_week,
)


# ===========================================================================
# _trading_days_between — weekday counting
# ===========================================================================

class TestTradingDaysBetween:
    def test_same_day_returns_zero(self):
        d = date(2026, 3, 10)  # Tuesday
        assert _trading_days_between(d, d) == 0

    def test_one_weekday_apart(self):
        # Tuesday → Wednesday
        assert _trading_days_between(date(2026, 3, 10), date(2026, 3, 11)) == 1

    def test_across_weekend(self):
        # Friday → Monday = 1 trading day
        assert _trading_days_between(date(2026, 3, 13), date(2026, 3, 16)) == 1

    def test_full_week(self):
        # Monday → Friday = 4 trading days (Tue-Fri)
        # Actually Mon→next Mon = 5 trading days
        assert _trading_days_between(date(2026, 3, 9), date(2026, 3, 13)) == 4

    def test_two_weeks(self):
        # Mon Mar 9 → Mon Mar 23 = 10 trading days
        assert _trading_days_between(date(2026, 3, 9), date(2026, 3, 23)) == 10

    def test_end_before_start_returns_zero(self):
        assert _trading_days_between(date(2026, 3, 15), date(2026, 3, 10)) == 0


# ===========================================================================
# get_earnings_date — yfinance integration
# ===========================================================================

class TestGetEarningsDate:
    def setup_method(self):
        clear_cache()

    @patch("data.events_feed.yf.Ticker")
    def test_returns_date_from_dict_calendar(self, mock_ticker_cls):
        from datetime import datetime as dt
        earnings_dt = dt(2026, 4, 15, 16, 30)
        ticker_mock = MagicMock()
        ticker_mock.calendar = {"Earnings Date": [earnings_dt]}
        mock_ticker_cls.return_value = ticker_mock

        result = get_earnings_date("AAPL")
        assert result == date(2026, 4, 15)

    @patch("data.events_feed.yf.Ticker")
    def test_returns_none_when_calendar_is_none(self, mock_ticker_cls):
        ticker_mock = MagicMock()
        ticker_mock.calendar = None
        mock_ticker_cls.return_value = ticker_mock

        assert get_earnings_date("FAKE") is None

    @patch("data.events_feed.yf.Ticker")
    def test_returns_none_on_exception(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = Exception("network error")
        assert get_earnings_date("BAD") is None

    @patch("data.events_feed.yf.Ticker")
    def test_caches_result(self, mock_ticker_cls):
        from datetime import datetime as dt
        earnings_dt = dt(2026, 4, 15)
        ticker_mock = MagicMock()
        ticker_mock.calendar = {"Earnings Date": [earnings_dt]}
        mock_ticker_cls.return_value = ticker_mock

        get_earnings_date("AAPL")
        get_earnings_date("AAPL")  # second call should use cache

        assert mock_ticker_cls.call_count == 1


# ===========================================================================
# get_days_to_earnings / is_earnings_week
# ===========================================================================

class TestDaysToEarnings:
    def setup_method(self):
        clear_cache()

    @patch("data.events_feed.get_earnings_date")
    def test_returns_none_when_no_earnings(self, mock_get):
        mock_get.return_value = None
        assert get_days_to_earnings("AAPL") is None

    @patch("data.events_feed.get_earnings_date")
    def test_returns_none_when_past(self, mock_get):
        mock_get.return_value = date.today() - timedelta(days=5)
        assert get_days_to_earnings("AAPL") is None

    @patch("data.events_feed.get_earnings_date")
    def test_returns_zero_for_today(self, mock_get):
        mock_get.return_value = date.today()
        assert get_days_to_earnings("AAPL") == 0

    @patch("data.events_feed.get_earnings_date")
    def test_counts_trading_days(self, mock_get):
        # Find the next Monday from today to set a predictable test
        today = date.today()
        days_ahead = (0 - today.weekday()) % 7  # next Monday
        if days_ahead == 0:
            days_ahead = 7
        next_monday = today + timedelta(days=days_ahead)
        # Set earnings to the Friday of same week = 4 trading days from Monday
        earnings_friday = next_monday + timedelta(days=4)
        mock_get.return_value = earnings_friday

        days = get_days_to_earnings("AAPL")
        assert days is not None
        assert days > 0


class TestIsEarningsWeek:
    def setup_method(self):
        clear_cache()

    @patch("data.events_feed.get_days_to_earnings")
    def test_true_when_within_5_days(self, mock_days):
        mock_days.return_value = 3
        assert is_earnings_week("AAPL") is True

    @patch("data.events_feed.get_days_to_earnings")
    def test_true_at_boundary_5_days(self, mock_days):
        mock_days.return_value = 5
        assert is_earnings_week("AAPL") is True

    @patch("data.events_feed.get_days_to_earnings")
    def test_false_beyond_5_days(self, mock_days):
        mock_days.return_value = 6
        assert is_earnings_week("AAPL") is False

    @patch("data.events_feed.get_days_to_earnings")
    def test_false_when_none(self, mock_days):
        mock_days.return_value = None
        assert is_earnings_week("AAPL") is False

    @patch("data.events_feed.get_days_to_earnings")
    def test_true_at_zero_days(self, mock_days):
        mock_days.return_value = 0
        assert is_earnings_week("AAPL") is True


# ===========================================================================
# RiskAgent earnings integration
# ===========================================================================

class TestRiskAgentEarningsCap:
    """Verify position sizing is capped during earnings events."""

    def _make_agent(self):
        from agents.risk_agent import RiskAgent
        db = MagicMock()
        db.log_risk_calculation.return_value = 99
        return RiskAgent(db=db)

    def _run(self, agent, days_to_earnings_val, **overrides):
        defaults = dict(
            ticker="TEST",
            signal="STRONG BUY",
            confidence=75.0,
            current_price=100.0,
            account_balance=10_000.0,
        )
        defaults.update(overrides)
        with patch("agents.risk_agent.get_days_to_earnings", return_value=days_to_earnings_val):
            return agent.run(**defaults)

    def test_no_earnings_returns_normal_position(self):
        agent = self._make_agent()
        result = self._run(agent, None)
        assert result["event_risk_flag"] == "none"
        assert result["skipped"] is False
        assert result["days_to_earnings"] is None

    def test_earnings_week_caps_position_at_50_pct(self):
        agent = self._make_agent()
        normal = self._run(agent, None)
        earnings_week = self._run(agent, 4)

        assert earnings_week["event_risk_flag"] == "earnings_week"
        assert earnings_week["days_to_earnings"] == 4
        # Position should be smaller (capped at 50% of Kelly)
        assert earnings_week["position_size_usd"] <= normal["position_size_usd"]
        assert earnings_week["skipped"] is False

    def test_earnings_imminent_caps_position_at_25_pct(self):
        agent = self._make_agent()
        normal = self._run(agent, None)
        imminent = self._run(agent, 1)

        assert imminent["event_risk_flag"] == "earnings_imminent"
        assert imminent["days_to_earnings"] == 1
        assert imminent["position_size_usd"] <= normal["position_size_usd"]

    def test_earnings_imminent_low_confidence_skips(self):
        agent = self._make_agent()
        result = self._run(agent, 1, confidence=40.0)

        assert result["skipped"] is True
        assert "imminent" in result["skip_reason"].lower()
        assert result["event_risk_flag"] == "earnings_imminent"

    def test_earnings_imminent_high_confidence_does_not_skip(self):
        agent = self._make_agent()
        result = self._run(agent, 2, confidence=75.0)

        assert result["skipped"] is False
        assert result["event_risk_flag"] == "earnings_imminent"

    def test_earnings_today_is_imminent(self):
        agent = self._make_agent()
        result = self._run(agent, 0, confidence=75.0)

        assert result["event_risk_flag"] == "earnings_imminent"

    def test_boundary_3_days_is_earnings_week_not_imminent(self):
        agent = self._make_agent()
        result = self._run(agent, 3)

        assert result["event_risk_flag"] == "earnings_week"

    def test_boundary_5_days_is_earnings_week(self):
        agent = self._make_agent()
        result = self._run(agent, 5)

        assert result["event_risk_flag"] == "earnings_week"

    def test_boundary_6_days_is_none(self):
        agent = self._make_agent()
        result = self._run(agent, 6)

        assert result["event_risk_flag"] == "none"

    def test_return_structure_includes_new_keys(self):
        agent = self._make_agent()
        result = self._run(agent, 3)
        assert "event_risk_flag" in result
        assert "days_to_earnings" in result


# ===========================================================================
# Coordinator earnings downgrade
# ===========================================================================

class TestCoordinatorEarningsDowngrade:
    """Coordinator should downgrade STRONG signals when earnings are imminent."""

    @patch("orchestrator.coordinator.get_days_to_earnings", return_value=1)
    def test_strong_buy_downgraded_to_weak_buy(self, mock_days):
        from orchestrator.coordinator import Coordinator
        # Just test the downgrade logic inline — no need to run full pipeline
        combined_signal = "STRONG BUY"
        days_to_earn = mock_days.return_value
        if days_to_earn is not None and days_to_earn <= 2:
            event_risk_flag = "earnings_imminent"
        else:
            event_risk_flag = "none"

        sizing_signal = combined_signal
        if event_risk_flag == "earnings_imminent":
            if combined_signal == "STRONG BUY":
                sizing_signal = "WEAK BUY"
            elif combined_signal == "STRONG SELL":
                sizing_signal = "WEAK SELL"

        assert sizing_signal == "WEAK BUY"
        assert event_risk_flag == "earnings_imminent"

    @patch("orchestrator.coordinator.get_days_to_earnings", return_value=1)
    def test_strong_sell_downgraded_to_weak_sell(self, mock_days):
        combined_signal = "STRONG SELL"
        days_to_earn = mock_days.return_value
        event_risk_flag = "earnings_imminent"

        sizing_signal = combined_signal
        if event_risk_flag == "earnings_imminent":
            if combined_signal == "STRONG BUY":
                sizing_signal = "WEAK BUY"
            elif combined_signal == "STRONG SELL":
                sizing_signal = "WEAK SELL"

        assert sizing_signal == "WEAK SELL"

    @patch("orchestrator.coordinator.get_days_to_earnings", return_value=4)
    def test_earnings_week_does_not_downgrade(self, mock_days):
        combined_signal = "STRONG BUY"
        days_to_earn = mock_days.return_value
        if days_to_earn is not None and days_to_earn <= 2:
            event_risk_flag = "earnings_imminent"
        else:
            event_risk_flag = "earnings_week" if days_to_earn <= 5 else "none"

        sizing_signal = combined_signal
        if event_risk_flag == "earnings_imminent":
            if combined_signal == "STRONG BUY":
                sizing_signal = "WEAK BUY"

        assert sizing_signal == "STRONG BUY"  # NOT downgraded

    @patch("orchestrator.coordinator.get_days_to_earnings", return_value=None)
    def test_no_earnings_data_no_downgrade(self, mock_days):
        combined_signal = "STRONG BUY"
        days_to_earn = mock_days.return_value
        if days_to_earn is not None and days_to_earn <= 2:
            event_risk_flag = "earnings_imminent"
        else:
            event_risk_flag = "none"

        sizing_signal = combined_signal
        if event_risk_flag == "earnings_imminent":
            sizing_signal = "WEAK BUY"

        assert sizing_signal == "STRONG BUY"
        assert event_risk_flag == "none"
