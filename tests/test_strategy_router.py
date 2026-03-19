"""
Unit tests for strategies.router — ticker-to-strategy mapping.
"""

import pytest

from strategies.router import get_strategy, strategy_label
from strategies.momentum import MomentumStrategy
from strategies.pullback import PullbackStrategy


class TestGetStrategy:
    """Verify get_strategy returns the correct strategy or None."""

    @pytest.mark.parametrize("ticker", ["MSFT", "NVDA", "DELL", "GOOGL", "META"])
    def test_momentum_tickers(self, ticker):
        strategy = get_strategy(ticker)
        assert isinstance(strategy, MomentumStrategy)

    @pytest.mark.parametrize("ticker", ["AAPL", "CEG", "VST"])
    def test_pullback_tickers(self, ticker):
        strategy = get_strategy(ticker)
        assert isinstance(strategy, PullbackStrategy)

    @pytest.mark.parametrize("ticker", ["SAP.XETRA", "SIE.XETRA", "JPM", "WMT", "TSLA"])
    def test_generic_tickers_return_none(self, ticker):
        assert get_strategy(ticker) is None

    def test_case_insensitive(self):
        assert isinstance(get_strategy("nvda"), MomentumStrategy)
        assert isinstance(get_strategy("aapl"), PullbackStrategy)
        assert get_strategy("sap.xetra") is None

    def test_singleton_instances(self):
        """Same instance returned on repeated calls (no re-creation)."""
        s1 = get_strategy("NVDA")
        s2 = get_strategy("MSFT")
        assert s1 is s2  # same MomentumStrategy singleton

        p1 = get_strategy("AAPL")
        p2 = get_strategy("CEG")
        assert p1 is p2  # same PullbackStrategy singleton


class TestStrategyLabel:
    """Verify strategy_label returns the correct short label."""

    @pytest.mark.parametrize("ticker,expected", [
        ("MSFT", "momentum"),
        ("NVDA", "momentum"),
        ("DELL", "momentum"),
        ("GOOGL", "momentum"),
        ("META", "momentum"),
        ("AAPL", "pullback"),
        ("CEG", "pullback"),
        ("VST", "pullback"),
        ("SAP.XETRA", "generic"),
        ("SIE.XETRA", "generic"),
        ("JPM", "generic"),
        ("TSLA", "generic"),
    ])
    def test_labels(self, ticker, expected):
        assert strategy_label(ticker) == expected

    def test_case_insensitive(self):
        assert strategy_label("nvda") == "momentum"
        assert strategy_label("Aapl") == "pullback"
        assert strategy_label("sap.xetra") == "generic"
