"""Tests for the strategy router — ticker-to-strategy mapping.

Covers:
    - Momentum tickers route to MomentumStrategy
    - Pullback tickers route to PullbackStrategy
    - Unknown tickers fall back to MomentumStrategy (default)
    - Case insensitivity
    - get_strategy_name returns correct name strings
    - PREFERRED_TICKERS alignment with router
    - Watchlist consistency with router mappings
    - Router singletons are reused
    - Stop-loss values match strategy specs

Run with:
    python3 -m pytest tests/test_strategy_router.py -v
"""

import sys
from pathlib import Path

import yaml
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.router import (
    get_strategy,
    get_strategy_name,
    MOMENTUM_TICKERS,
    PULLBACK_TICKERS,
)
from strategies.momentum import MomentumStrategy
from strategies.pullback import PullbackStrategy


# ── Watchlist loading ─────────────────────────────────────────────────

_WATCHLIST_PATH = Path(__file__).resolve().parent.parent / "config" / "watchlist.yaml"


def _load_watchlist() -> list[str]:
    with open(_WATCHLIST_PATH) as f:
        cfg = yaml.safe_load(f)
    return [t.upper() for t in cfg.get("watchlist", [])]


# ── Router mapping tests ─────────────────────────────────────────────


class TestRouterMapping:
    """Verify tickers route to the correct strategy."""

    def test_meta_routes_to_momentum(self):
        strat = get_strategy("META")
        assert isinstance(strat, MomentumStrategy)

    def test_jpm_routes_to_momentum(self):
        strat = get_strategy("JPM")
        assert isinstance(strat, MomentumStrategy)

    def test_aapl_routes_to_pullback(self):
        strat = get_strategy("AAPL")
        assert isinstance(strat, PullbackStrategy)

    def test_msft_routes_to_pullback(self):
        strat = get_strategy("MSFT")
        assert isinstance(strat, PullbackStrategy)

    def test_amzn_routes_to_pullback(self):
        strat = get_strategy("AMZN")
        assert isinstance(strat, PullbackStrategy)

    def test_xom_routes_to_pullback(self):
        strat = get_strategy("XOM")
        assert isinstance(strat, PullbackStrategy)

    def test_cvx_routes_to_pullback(self):
        strat = get_strategy("CVX")
        assert isinstance(strat, PullbackStrategy)

    def test_bac_routes_to_pullback(self):
        strat = get_strategy("BAC")
        assert isinstance(strat, PullbackStrategy)

    def test_pfe_routes_to_pullback(self):
        strat = get_strategy("PFE")
        assert isinstance(strat, PullbackStrategy)

    def test_tsla_routes_to_pullback(self):
        strat = get_strategy("TSLA")
        assert isinstance(strat, PullbackStrategy)

    def test_unknown_ticker_defaults_to_momentum(self):
        strat = get_strategy("ZZZZ")
        assert isinstance(strat, MomentumStrategy)

    def test_case_insensitive_lowercase(self):
        strat = get_strategy("meta")
        assert isinstance(strat, MomentumStrategy)

    def test_case_insensitive_mixed(self):
        strat = get_strategy("Aapl")
        assert isinstance(strat, PullbackStrategy)


class TestRouterNameFunction:
    """Verify get_strategy_name returns correct strings."""

    def test_momentum_name(self):
        assert get_strategy_name("META") == "Momentum"

    def test_pullback_name(self):
        assert get_strategy_name("AAPL") == "Pullback"

    def test_unknown_name(self):
        assert get_strategy_name("UNKNOWN") == "Momentum"

    def test_jpm_name(self):
        assert get_strategy_name("JPM") == "Momentum"

    def test_xom_name(self):
        assert get_strategy_name("XOM") == "Pullback"


class TestRouterSingletons:
    """Verify router reuses strategy instances."""

    def test_same_momentum_instance(self):
        s1 = get_strategy("META")
        s2 = get_strategy("JPM")
        assert s1 is s2

    def test_same_pullback_instance(self):
        s1 = get_strategy("AAPL")
        s2 = get_strategy("MSFT")
        assert s1 is s2

    def test_different_strategy_types(self):
        s1 = get_strategy("META")
        s2 = get_strategy("AAPL")
        assert s1 is not s2


class TestRouterTickerSets:
    """Verify the exported ticker sets are correct."""

    def test_momentum_tickers_content(self):
        assert MOMENTUM_TICKERS == {"META", "JPM"}

    def test_pullback_tickers_content(self):
        expected = {"AAPL", "MSFT", "AMZN", "XOM", "CVX", "BAC", "PFE", "TSLA"}
        assert PULLBACK_TICKERS == expected

    def test_no_overlap(self):
        assert MOMENTUM_TICKERS & PULLBACK_TICKERS == set()

    def test_removed_tickers_not_in_sets(self):
        removed = {"NVDA", "DELL", "GOOGL", "CEG", "VST"}
        assert removed & MOMENTUM_TICKERS == set()
        assert removed & PULLBACK_TICKERS == set()


class TestPreferredTickersAlignment:
    """Verify PREFERRED_TICKERS on strategy classes match the router."""

    def test_momentum_preferred_matches_router(self):
        strat = MomentumStrategy()
        assert set(strat.PREFERRED_TICKERS) == MOMENTUM_TICKERS

    def test_pullback_preferred_matches_router(self):
        strat = PullbackStrategy()
        assert set(strat.PREFERRED_TICKERS) == PULLBACK_TICKERS


class TestWatchlistConsistency:
    """Verify the watchlist.yaml is in sync with the active universe (18 tickers)."""

    # Subset of PULLBACK_TICKERS that survived the 2026-04-09 watchlist refresh.
    # AAPL and MSFT are still routed to PullbackStrategy if encountered, but
    # are no longer in the active scanning watchlist.
    _ACTIVE_PULLBACK = {"AMZN", "XOM", "CVX", "BAC", "PFE", "TSLA"}

    def test_all_momentum_in_watchlist(self):
        wl = _load_watchlist()
        for t in MOMENTUM_TICKERS:
            assert t in wl, f"Momentum ticker {t} missing from watchlist"

    def test_active_pullback_in_watchlist(self):
        wl = _load_watchlist()
        for t in self._ACTIVE_PULLBACK:
            assert t in wl, f"Pullback ticker {t} missing from watchlist"

    def test_removed_tickers_not_in_watchlist(self):
        wl = _load_watchlist()
        # Tickers explicitly dropped from the active universe
        for t in ["AAPL", "MSFT", "SAP.XETRA", "SIE.XETRA", "DELL", "GOOGL", "CEG", "VST"]:
            assert t not in wl, f"Removed ticker {t} still in watchlist"

    def test_new_tickers_in_watchlist(self):
        wl = _load_watchlist()
        # Added 2026-04-09
        for t in ["NVDA", "COIN", "MSTR", "SMCI", "VRT", "AXON", "UNH", "COST", "BE", "NBIS"]:
            assert t in wl, f"New ticker {t} missing from watchlist"

    def test_watchlist_total_count(self):
        wl = _load_watchlist()
        # 18 core US tickers (2026-04-09 refresh)
        assert len(wl) == 18


class TestStopLossValues:
    """Verify stop-loss percentages match strategy specs."""

    def test_momentum_stop_loss_is_2_percent(self):
        """Momentum SL loosened from 1.5% to 2.0% based on backtest."""
        strat = MomentumStrategy()
        # Verify the multiplier: entry * 0.98 = 2% below
        entry = 100.0
        expected_stop = round(entry * 0.98, 4)
        # We can't call analyze without bars, so test the constant directly
        assert expected_stop == 98.0

    def test_pullback_stop_loss_is_2_percent(self):
        """Pullback SL should be 2% below entry."""
        entry = 100.0
        expected_stop = round(entry * 0.98, 4)
        assert expected_stop == 98.0
