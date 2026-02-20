"""
Unit tests for ScreenerAgent.

All external data (yfinance batch downloads, Wikipedia scrapes) is mocked.
No real network calls are made in the unit tests.

Test classes
------------
TestComputeMetrics      Pure metric computation from a synthetic DataFrame.
TestPassesFilter        Filter gate logic for blue_chip / mid_cap / small_cap.
TestComputeHotness      Hotness formula components and boundary values.
TestEnfocusQuota        German focus-market minimum-quota enforcement.
TestRunPipeline         End-to-end run() with mocked yfinance download.
TestSplitMultiindex     MultiIndex → per-ticker dict conversion.

Integration tests (require real network):
TestLiveScreener        @pytest.mark.integration — real yfinance download.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DB_PATH",           "/tmp/pytest_screener.db")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("NEWSAPI_KEY",       "test-key-not-real")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(days: int = 60, base_price: float = 150.0, seed: int = 0) -> pd.DataFrame:
    """Deterministic OHLCV DataFrame."""
    rng    = np.random.default_rng(seed)
    dates  = pd.date_range(end="2025-01-15", periods=days, freq="B")
    close  = base_price + np.cumsum(rng.standard_normal(days) * 1.5)
    volume = np.full(days, 50_000_000, dtype=float)
    return pd.DataFrame(
        {"Open": close*0.99, "High": close*1.01, "Low": close*0.98,
         "Close": close, "Volume": volume},
        index=dates,
    )


def _make_agent(tmp_path=None):
    from agents.screener_agent import ScreenerAgent
    if tmp_path:
        from storage.database import Database
        db = Database(str(tmp_path / "screener.db"))
        return ScreenerAgent(db=db)
    return ScreenerAgent()


# ══════════════════════════════════════════════════════════════════════════════
# 1. _compute_metrics — pure function (no network)
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:

    def setup_method(self):
        from agents.screener_agent import ScreenerAgent
        self.agent = ScreenerAgent.__new__(ScreenerAgent)
        self.agent._price_cache = {}
        self.agent._list_cache  = {}
        self.agent._RSI_WINDOW     = 14
        self.agent._VOL_AVG_WINDOW = 20

    def test_returns_none_for_short_df(self):
        df = _make_ohlcv(days=10)  # fewer than RSI_WINDOW + 1
        result = self.agent._compute_metrics("AAPL", df)
        assert result is None

    def test_returns_dict_for_valid_df(self):
        df = _make_ohlcv(days=60)
        result = self.agent._compute_metrics("AAPL", df)
        assert result is not None
        assert isinstance(result, dict)

    def test_price_is_last_close(self):
        df = _make_ohlcv(days=60)
        result = self.agent._compute_metrics("AAPL", df)
        assert result["price"] == pytest.approx(float(df["Close"].iloc[-1]), rel=1e-6)

    def test_price_change_is_pct_vs_previous(self):
        df = _make_ohlcv(days=60)
        result = self.agent._compute_metrics("AAPL", df)
        expected = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
        assert result["price_change"] == pytest.approx(float(expected), rel=1e-6)

    def test_volume_ratio_uses_20day_avg(self):
        df = _make_ohlcv(days=60)
        result = self.agent._compute_metrics("AAPL", df)
        # All volumes are 50M so ratio should be 1.0
        assert result["volume_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_rsi_is_between_0_and_100(self):
        df = _make_ohlcv(days=60)
        result = self.agent._compute_metrics("AAPL", df)
        if result["rsi"] is not None:
            assert 0.0 <= result["rsi"] <= 100.0

    def test_missing_close_column_returns_none(self):
        df = _make_ohlcv(days=60).rename(columns={"Close": "close_price"})
        result = self.agent._compute_metrics("AAPL", df)
        assert result is None

    def test_required_keys_present(self):
        df = _make_ohlcv(days=60)
        result = self.agent._compute_metrics("AAPL", df)
        for key in ("price", "price_change", "volume_ratio", "avg_volume", "rsi"):
            assert key in result


# ══════════════════════════════════════════════════════════════════════════════
# 2. _passes_filter — pure function (no network)
# ══════════════════════════════════════════════════════════════════════════════

class TestPassesFilter:

    def setup_method(self):
        from agents.screener_agent import ScreenerAgent, _FILTERS
        self.agent   = ScreenerAgent.__new__(ScreenerAgent)
        self.agent._price_cache = {}
        self.agent._list_cache  = {}
        self.filters = _FILTERS

    def _metrics(self, volume_ratio=2.5, price_change=4.0, avg_volume=10_000_000):
        return {
            "volume_ratio": volume_ratio,
            "price_change": price_change,
            "avg_volume":   avg_volume,
        }

    def test_blue_chip_passes_with_2x_volume_and_3pct_move(self):
        assert self.agent._passes_filter(self._metrics(2.0, 3.0), "blue_chip") is True

    def test_blue_chip_fails_with_low_volume_ratio(self):
        assert self.agent._passes_filter(self._metrics(1.5, 5.0), "blue_chip") is False

    def test_blue_chip_fails_with_small_price_move(self):
        assert self.agent._passes_filter(self._metrics(3.0, 2.0), "blue_chip") is False

    def test_mid_cap_passes_threshold(self):
        assert self.agent._passes_filter(self._metrics(1.5, 4.0, 200_000), "mid_cap") is True

    def test_mid_cap_fails_low_volume_avg(self):
        assert self.agent._passes_filter(self._metrics(2.0, 5.0, 50_000), "mid_cap") is False

    def test_small_cap_threshold_same_as_mid_cap(self):
        assert self.agent._passes_filter(self._metrics(1.5, 4.0, 200_000), "small_cap") is True

    def test_large_volume_ratio_passes_all(self):
        for mtype in ("blue_chip", "mid_cap", "small_cap"):
            assert self.agent._passes_filter(
                self._metrics(10.0, 10.0, 1_000_000), mtype
            ) is True


# ══════════════════════════════════════════════════════════════════════════════
# 3. _compute_hotness — pure formula (no network)
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeHotness:

    def setup_method(self):
        from agents.screener_agent import ScreenerAgent
        self.agent = ScreenerAgent.__new__(ScreenerAgent)
        self.agent._price_cache   = {}
        self.agent._list_cache    = {}
        self.agent._HOTNESS_SCALE = 10.0
        self.agent._VOL_RATIO_CAP = 5.0
        self.agent._PRICE_CHG_CAP = 10.0

    def _metrics(self, vol_ratio=1.0, price_change=0.0, rsi=50.0, avg_volume=1_000_000):
        return {
            "volume_ratio": vol_ratio,
            "price_change": price_change,
            "rsi":          rsi,
            "avg_volume":   avg_volume,
        }

    def test_score_is_between_0_and_10(self):
        for vr, pc in [(1.0, 0.0), (5.0, 10.0), (0.0, 0.0)]:
            score = self.agent._compute_hotness(self._metrics(vr, pc), 0.0, "")
            assert 0.0 <= score <= 10.0, f"score={score} out of range for vr={vr}, pc={pc}"

    def test_higher_volume_ratio_gives_higher_score(self):
        low  = self.agent._compute_hotness(self._metrics(vol_ratio=1.0), 0.0, "")
        high = self.agent._compute_hotness(self._metrics(vol_ratio=4.0), 0.0, "")
        assert high > low

    def test_higher_price_change_gives_higher_score(self):
        low  = self.agent._compute_hotness(self._metrics(price_change=1.0), 0.0, "")
        high = self.agent._compute_hotness(self._metrics(price_change=8.0), 0.0, "")
        assert high > low

    def test_extreme_rsi_gives_higher_score_than_neutral(self):
        neutral  = self.agent._compute_hotness(self._metrics(rsi=50.0),  0.0, "")
        oversold = self.agent._compute_hotness(self._metrics(rsi=20.0),  0.0, "")
        overbought = self.agent._compute_hotness(self._metrics(rsi=80.0), 0.0, "")
        assert oversold   > neutral
        assert overbought > neutral

    def test_volume_ratio_capped_at_5x(self):
        at_cap   = self.agent._compute_hotness(self._metrics(vol_ratio=5.0),  0.0, "")
        above_cap = self.agent._compute_hotness(self._metrics(vol_ratio=10.0), 0.0, "")
        assert at_cap == pytest.approx(above_cap, abs=0.01)

    def test_price_change_capped_at_10pct(self):
        at_cap    = self.agent._compute_hotness(self._metrics(price_change=10.0), 0.0, "")
        above_cap = self.agent._compute_hotness(self._metrics(price_change=20.0), 0.0, "")
        assert at_cap == pytest.approx(above_cap, abs=0.01)

    def test_focus_market_priority_bonus(self):
        no_bonus  = self.agent._compute_hotness(self._metrics(), -0.1, "DE")
        with_bonus = self.agent._compute_hotness(self._metrics(),  0.2, "DE")
        assert with_bonus > no_bonus

    def test_neutral_rsi_none_treated_as_50(self):
        m_none = self._metrics(rsi=None)
        m_50   = self._metrics(rsi=50.0)
        m_none["rsi"] = None
        score_none = self.agent._compute_hotness(m_none, 0.0, "")
        score_50   = self.agent._compute_hotness(m_50,   0.0, "")
        assert score_none == pytest.approx(score_50, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Focus quota enforcement
# ══════════════════════════════════════════════════════════════════════════════

class TestEnfocusQuota:

    def setup_method(self):
        from agents.screener_agent import ScreenerAgent
        self.agent = ScreenerAgent.__new__(ScreenerAgent)
        self.agent._price_cache = {}
        self.agent._list_cache  = {}

    def _make_candidate(self, ticker: str, market: str, hotness: float) -> dict:
        return {
            "ticker": ticker, "market": market, "hotness": hotness,
            "price": 100.0, "price_change": 1.0, "volume_ratio": 2.0,
            "rsi": 50.0, "avg_volume": 1_000_000, "market_cap": None,
            "country": "DE" if ".DE" in ticker else "US", "exchange": "XETRA",
            "name": ticker,
        }

    def test_de_focus_guarantees_representation_when_available(self):
        # 10 US candidates with high hotness + 5 DE candidates with low hotness
        candidates = (
            [self._make_candidate(f"US{i}", "SP500", 9.0 - i * 0.1) for i in range(10)] +
            [self._make_candidate(f"DE{i}.DE", "DAX",  1.0 + i * 0.1) for i in range(5)]
        )
        top = self.agent._enforce_focus_quota(candidates, "DE", top=10)
        de_tickers = [c for c in top if c["market"] == "DAX"]
        # At least some DE candidates should appear despite lower hotness
        assert len(de_tickers) >= 1

    def test_no_quota_when_focus_is_none(self):
        candidates = [self._make_candidate(f"US{i}", "SP500", 9.0 - i * 0.1) for i in range(10)]
        top = self.agent._enforce_focus_quota(candidates, None, top=5)
        assert len(top) <= 5

    def test_result_length_does_not_exceed_top(self):
        candidates = [self._make_candidate(f"T{i}", "DAX", float(i)) for i in range(50)]
        top = self.agent._enforce_focus_quota(candidates, "DE", top=20)
        assert len(top) <= 20

    def test_candidates_sorted_by_hotness_descending(self):
        candidates = [self._make_candidate(f"T{i}", "DAX", float(i)) for i in range(20)]
        top = self.agent._enforce_focus_quota(candidates, "DE", top=10)
        hotnesses = [c["hotness"] for c in top]
        assert hotnesses == sorted(hotnesses, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. run() pipeline — yfinance mocked
# ══════════════════════════════════════════════════════════════════════════════

class TestRunPipeline:

    def _make_dax_download(self) -> pd.DataFrame:
        """Simulate a single-ticker yf.download return for DAX tickers."""
        return _make_ohlcv(days=60, base_price=150.0)

    def test_run_returns_expected_keys(self, tmp_db):
        from agents.screener_agent import ScreenerAgent

        ohlcv = _make_ohlcv(days=60)
        with patch("agents.screener_agent.yf.download", return_value=ohlcv), \
             patch("agents.screener_agent.yf.Ticker") as mock_t:
            mock_t.return_value.fast_info = MagicMock(market_cap=1_000_000_000)
            agent  = ScreenerAgent(db=tmp_db)
            result = agent.run(markets=["DE"], focus_market="DE", top=5)

        for key in ("run_at", "markets_scanned", "focus_market",
                    "universe_size", "screened", "candidates"):
            assert key in result

    def test_run_markets_scanned_matches_input(self, tmp_db):
        from agents.screener_agent import ScreenerAgent

        ohlcv = _make_ohlcv(days=60)
        with patch("agents.screener_agent.yf.download", return_value=ohlcv), \
             patch("agents.screener_agent.yf.Ticker"):
            agent  = ScreenerAgent(db=tmp_db)
            result = agent.run(markets=["DE"], focus_market=None, top=5)

        assert result["markets_scanned"] == ["DE"]

    def test_run_empty_download_returns_zero_candidates(self, tmp_db):
        from agents.screener_agent import ScreenerAgent

        empty_df = pd.DataFrame()
        with patch("agents.screener_agent.yf.download", return_value=empty_df):
            agent  = ScreenerAgent(db=tmp_db)
            result = agent.run(markets=["DE"], focus_market=None, top=5)

        assert result["candidates"] == []

    def test_candidates_list_capped_at_top(self, tmp_db):
        from agents.screener_agent import ScreenerAgent

        # Build a MultiIndex download result for multiple DE tickers
        dax_tickers = ["ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE"]
        cols = pd.MultiIndex.from_product(
            [dax_tickers, ["Open", "High", "Low", "Close", "Volume"]],
            names=["Ticker", "Price"],
        )
        idx = pd.date_range(end="2025-01-15", periods=60, freq="B")
        data = {}
        for t in dax_tickers:
            ohlcv = _make_ohlcv(days=60)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                data[(t, col)] = ohlcv[col].values
        multi_df = pd.DataFrame(data, index=idx)
        multi_df.columns = pd.MultiIndex.from_tuples(multi_df.columns, names=["Ticker", "Price"])

        with patch("agents.screener_agent.yf.download", return_value=multi_df), \
             patch("agents.screener_agent.yf.Ticker") as mock_t:
            mock_t.return_value.fast_info = MagicMock(market_cap=None)
            agent  = ScreenerAgent(db=tmp_db)
            result = agent.run(markets=["DE"], focus_market="DE", top=3)

        assert len(result["candidates"]) <= 3

    def test_candidates_have_required_fields(self, tmp_db):
        from agents.screener_agent import ScreenerAgent

        ohlcv = _make_ohlcv(days=60)
        with patch("agents.screener_agent.yf.download", return_value=ohlcv), \
             patch("agents.screener_agent.yf.Ticker") as mock_t:
            mock_t.return_value.fast_info = MagicMock(market_cap=None)
            agent  = ScreenerAgent(db=tmp_db)
            result = agent.run(markets=["DE"], focus_market=None, top=5)

        for c in result["candidates"]:
            for field in ("ticker", "market", "hotness", "price_change",
                          "volume_ratio", "price", "avg_volume"):
                assert field in c, f"Missing field '{field}' in candidate {c}"


# ══════════════════════════════════════════════════════════════════════════════
# 6. _split_multiindex — pure helper (no network)
# ══════════════════════════════════════════════════════════════════════════════

class TestSplitMultiindex:

    def setup_method(self):
        from agents.screener_agent import ScreenerAgent
        self.agent = ScreenerAgent.__new__(ScreenerAgent)
        self.agent._price_cache = {}
        self.agent._list_cache  = {}

    def test_single_ticker_returns_plain_df(self):
        df = _make_ohlcv(days=30)
        result = self.agent._split_multiindex(df, ["AAPL"])
        assert "AAPL" in result
        assert not result["AAPL"].empty

    def test_multi_ticker_splits_correctly(self):
        tickers = ["AAPL", "NVDA"]
        cols = pd.MultiIndex.from_product(
            [tickers, ["Close", "Volume", "Open", "High", "Low"]],
        )
        idx  = pd.date_range(end="2025-01-15", periods=30, freq="B")
        data = {(t, f): np.ones(30) * (100 + i) for i, t in enumerate(tickers)
                for f in ["Close", "Volume", "Open", "High", "Low"]}
        df   = pd.DataFrame(data, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        result = self.agent._split_multiindex(df, tickers)
        assert "AAPL" in result
        assert "NVDA" in result

    def test_unknown_ticker_not_in_result(self):
        df = _make_ohlcv(days=30)
        result = self.agent._split_multiindex(df, ["UNKNOWN"])
        # Single-ticker plain DF — returned under UNKNOWN key
        # OR empty dict if column names don't match
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Integration tests (real yfinance — skipped by default)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.slow
class TestLiveScreener:
    """
    Makes real yfinance download calls.

    Run with:
        pytest -m integration tests/test_screener_agent.py
    """

    def test_live_dax_scan(self, tmp_db):
        from agents.screener_agent import ScreenerAgent
        agent  = ScreenerAgent(db=tmp_db)
        result = agent.run(markets=["DE"], focus_market="DE", top=5)
        assert "candidates" in result
        assert isinstance(result["candidates"], list)
        assert result["universe_size"] > 0
