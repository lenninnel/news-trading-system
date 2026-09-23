"""Data paths + honest logging (2026-09-23).

(1) Every signal_events row says which daily bar the indicators rest on and
    which live price the decision saw (indicator_bar_* / live_price*).
(2) RiskAgent skips persist their earnings context instead of flag=none.
(3) PriceFallback finds the Alpha Vantage key under either env name.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.technical_agent import BAR_SOURCE_ATTR, TechnicalAgent, bar_provenance  # noqa: E402
from analytics.signal_logger import SignalLogger  # noqa: E402
from orchestrator.coordinator import Coordinator, _bar_fields, _live_fields  # noqa: E402
from storage.database import Database  # noqa: E402
from strategies.base import StrategyResult  # noqa: E402


def _frame(n: int = 5, source: str | None = "daily_ohlc") -> pd.DataFrame:
    idx = pd.date_range("2026-09-14", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": [100.0 + i for i in range(n)],
        "High": [101.0 + i for i in range(n)],
        "Low": [99.0 + i for i in range(n)],
        "Close": [100.5 + i for i in range(n)],
        "Volume": [1000] * n,
    }, index=idx)
    if source:
        df.attrs[BAR_SOURCE_ATTR] = source
    return df


# ── bar provenance ─────────────────────────────────────────────────────────

class TestBarProvenance:
    def test_last_bar_date_close_and_source(self):
        prov = bar_provenance(_frame(5))
        assert prov == {"bar_date": "2026-09-18", "bar_close": 104.5, "bar_source": "daily_ohlc"}

    def test_untagged_frame_has_no_source(self):
        prov = bar_provenance(_frame(3, source=None))
        assert prov["bar_date"] == "2026-09-16" and prov["bar_source"] is None

    def test_empty_or_none_is_all_none(self):
        assert bar_provenance(None) == {"bar_date": None, "bar_close": None, "bar_source": None}
        assert bar_provenance(pd.DataFrame()) == {"bar_date": None, "bar_close": None, "bar_source": None}

    def test_fetch_history_tags_the_store_as_source(self):
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        f.close()
        db = Database(db_path=f.name)
        try:
            rows, d = [], date(2026, 3, 2)
            while len(rows) < 80:
                if d.weekday() < 5:
                    rows.append({"ticker": "TST", "date": d.isoformat(), "open": 100, "high": 101,
                                 "low": 99, "close": 100 + len(rows) * 0.5, "volume": 10, "source": "alpaca"})
                d += timedelta(days=1)
            db.upsert_daily_ohlc(rows)
            agent = TechnicalAgent(db=db, alpaca_client=MagicMock())
            df = agent._fetch_history("TST")
            prov = bar_provenance(df)
            assert prov["bar_source"] == "daily_ohlc"
            assert prov["bar_date"] == rows[-1]["date"]
            assert prov["bar_close"] == pytest.approx(rows[-1]["close"])
        finally:
            os.unlink(f.name)

    def test_fetch_history_tags_yfinance_fallback(self):
        db = MagicMock()
        db.get_daily_ohlc.return_value = []          # store empty → fallback
        yf = MagicMock()
        yf.get_bars.return_value = _frame(70, source=None)
        agent = TechnicalAgent(db=db, alpaca_client=yf)
        df = agent._fetch_history("NEW")
        assert df.attrs[BAR_SOURCE_ATTR] == "yfinance"

    def test_indicators_carry_bar_fields(self):
        """run() puts bar_date/bar_close/bar_source next to the indicators."""
        db = MagicMock()
        db.log_technical_signal.return_value = 1
        agent = TechnicalAgent(db=db, alpaca_client=MagicMock())
        frame = _frame(250)
        with patch.object(agent, "_fetch_history", return_value=frame), \
             patch.object(agent, "_fetch_multi_timeframe", return_value={
                 "rsi_1h": None, "macd_15m_hist": None,
                 "timeframe_alignment": 0.5, "intraday_available": False}), \
             patch.object(agent, "_intraday_supplement", return_value=None):
            result = agent.run("AAPL")
        ind = result["indicators"]
        assert ind["bar_date"] == frame.index[-1].strftime("%Y-%m-%d")
        assert ind["bar_close"] == pytest.approx(float(frame["Close"].iloc[-1]))
        assert ind["bar_source"] == "daily_ohlc"
        assert ind["price"] == pytest.approx(ind["bar_close"])   # price_at_signal IS the bar close


# ── coordinator helpers + logging ──────────────────────────────────────────

class TestCoordinatorFields:
    def test_bar_fields_prefer_indicator_keys(self):
        out = _bar_fields(None, {"bar_date": "2026-09-18", "bar_close": 104.5, "bar_source": "daily_ohlc"})
        assert out == {"indicator_bar_date": "2026-09-18", "indicator_bar_close": 104.5,
                       "indicator_bar_source": "daily_ohlc"}

    def test_bar_fields_derive_from_frame(self):
        out = _bar_fields(_frame(4), {})
        assert out["indicator_bar_date"] == "2026-09-17" and out["indicator_bar_source"] == "daily_ohlc"

    def test_live_fields(self):
        assert _live_fields({"price": 181.2, "source": "alpaca", "degraded": False}) == \
            {"live_price": 181.2, "live_price_source": "alpaca"}
        assert _live_fields({"price": None, "source": "none", "degraded": True}) == \
            {"live_price": None, "live_price_source": None}
        assert _live_fields(None) == {"live_price": None, "live_price_source": None}

    def test_combined_row_logs_bar_and_live_price(self):
        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock()
        frame = _frame(5)
        result = {
            "ticker": "AAPL",
            "combined_signal": "WEAK BUY",
            "confidence": 0.35,
            "technical": {
                "indicators": {"price": 104.5, "rsi": 45.0, "bar_date": "2026-09-18",
                               "bar_close": 104.5, "bar_source": "daily_ohlc"},
                "bars": frame,
            },
            "sentiment": {"avg_score": 0.4, "source_breakdown": {},
                          "market": {"price": 105.9, "source": "alpaca", "degraded": False}},
            "execution": {},
        }
        coordinator._log_signal_event(result, session="US_OPEN")
        logged = coordinator.signal_logger.log.call_args[0][0]
        assert logged["price_at_signal"] == 104.5
        assert logged["indicator_bar_date"] == "2026-09-18"
        assert logged["indicator_bar_close"] == 104.5
        assert logged["indicator_bar_source"] == "daily_ohlc"
        assert logged["live_price"] == 105.9
        assert logged["live_price_source"] == "alpaca"

    def test_combined_row_without_live_price_logs_null(self):
        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock()
        result = {
            "ticker": "AAPL", "combined_signal": "HOLD", "confidence": 0.1,
            "technical": {"indicators": {"price": 104.5}, "bars": _frame(2)},
            "sentiment": {"avg_score": 0.0, "source_breakdown": {},
                          "market": {"price": None, "source": "none", "degraded": True}},
            "execution": {},
        }
        coordinator._log_signal_event(result, session="EOD")
        logged = coordinator.signal_logger.log.call_args[0][0]
        assert logged["live_price"] is None and logged["live_price_source"] is None
        assert logged["indicator_bar_date"] == "2026-09-15"

    def test_strategy_row_logs_bar_and_live_price(self):
        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock()
        sr = StrategyResult(strategy_name="Momentum", signal="STRONG BUY", confidence=70.0,
                            reasoning=[], indicators={"price": 104.5})
        coordinator._log_strategy_result(
            "AAPL", sr, session="US_PRE", bars=_frame(5),
            live_market={"price": 105.9, "source": "alpaca", "degraded": False},
        )
        logged = coordinator.signal_logger.log.call_args[0][0]
        assert logged["indicator_bar_date"] == "2026-09-18"
        assert logged["indicator_bar_source"] == "daily_ohlc"
        assert logged["live_price"] == 105.9

    def test_strategy_row_without_bars_logs_null(self):
        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock()
        sr = StrategyResult(strategy_name="PEAD", signal="HOLD", confidence=0.0,
                            reasoning=[], indicators={})
        coordinator._log_strategy_result("AAPL", sr, session="PEAD_OPEN")
        logged = coordinator.signal_logger.log.call_args[0][0]
        assert logged["indicator_bar_date"] is None and logged["live_price"] is None


class TestSignalLoggerColumns:
    def test_columns_persist_and_migrate_idempotently(self, tmp_path):
        db = Database(str(tmp_path / "s.db"))
        SignalLogger(db=db)
        lgr = SignalLogger(db=db)          # second init must not raise on duplicate columns
        lgr.log({"ticker": "AAPL", "signal": "WEAK BUY", "price_at_signal": 104.5,
                 "indicator_bar_date": "2026-09-18", "indicator_bar_close": 104.5,
                 "indicator_bar_source": "daily_ohlc",
                 "live_price": 105.9, "live_price_source": "alpaca"})
        row = lgr.get_signals("AAPL", days=1)[0]
        assert row["indicator_bar_date"] == "2026-09-18"
        assert row["indicator_bar_close"] == 104.5
        assert row["indicator_bar_source"] == "daily_ohlc"
        assert row["live_price"] == 105.9
        assert row["live_price_source"] == "alpaca"
        assert row["outcome_status"] is None      # tracker columns exist, untouched

    def test_legacy_rows_stay_null(self, tmp_path):
        db = Database(str(tmp_path / "s.db"))
        lgr = SignalLogger(db=db)
        lgr.log({"ticker": "AAPL", "signal": "HOLD"})
        row = lgr.get_signals("AAPL", days=1)[0]
        assert row["indicator_bar_date"] is None and row["live_price"] is None


# ── RiskAgent skip logging ─────────────────────────────────────────────────

class TestRiskAgentSkipLogging:
    def _agent(self):
        from agents.risk_agent import RiskAgent
        db = MagicMock()
        db.log_risk_calculation.return_value = 7
        return RiskAgent(db=db), db

    def test_earnings_imminent_skip_persists_flag_and_days(self):
        agent, db = self._agent()
        with patch("agents.risk_agent.get_days_to_earnings", return_value=1), \
             patch("agents.risk_agent.USE_ATR_STOPS", False):
            res = agent.run(ticker="CASY", signal="WEAK BUY", confidence=44.0,
                            current_price=100.0, account_balance=10_000.0)
        assert res["skipped"] and "Earnings imminent" in res["skip_reason"]
        kw = db.log_risk_calculation.call_args.kwargs
        assert kw["skipped"] is True
        assert kw["event_risk_flag"] == "earnings_imminent"
        assert kw["days_to_earnings"] == 1
        assert res["event_risk_flag"] == "earnings_imminent" and res["days_to_earnings"] == 1

    def test_confidence_skip_still_carries_earnings_week(self):
        agent, db = self._agent()
        with patch("agents.risk_agent.get_days_to_earnings", return_value=4), \
             patch("agents.risk_agent.USE_ATR_STOPS", False):
            res = agent.run(ticker="AAPL", signal="WEAK BUY", confidence=10.0,
                            current_price=100.0, account_balance=10_000.0, regime="TRENDING")
        assert res["skipped"]
        kw = db.log_risk_calculation.call_args.kwargs
        assert kw["event_risk_flag"] == "earnings_week"
        assert kw["days_to_earnings"] == 4
        assert kw["regime"] == "TRENDING"

    def test_unknown_earnings_date_is_logged(self, caplog):
        agent, db = self._agent()
        with patch("agents.risk_agent.get_days_to_earnings", return_value=None), \
             patch("agents.risk_agent.USE_ATR_STOPS", False), \
             caplog.at_level(logging.INFO, logger="agents.risk_agent"):
            agent.run(ticker="AAPL", signal="STRONG BUY", confidence=75.0,
                      current_price=100.0, account_balance=10_000.0)
        assert any("earnings date unknown" in r.getMessage() and "filter open" in r.getMessage()
                   for r in caplog.records)
        assert db.log_risk_calculation.call_args.kwargs["event_risk_flag"] == "none"


# ── PriceFallback env key ──────────────────────────────────────────────────

class TestAlphaVantageKey:
    def test_api_key_spelling_is_accepted(self):
        from data.price_fallback import PriceFallback
        with patch.dict(os.environ, {"ALPHA_VANTAGE_KEY": "", "ALPHA_VANTAGE_API_KEY": "abc123"}):
            assert PriceFallback()._alpha_key == "abc123"

    def test_legacy_spelling_wins_when_both_set(self):
        from data.price_fallback import PriceFallback
        with patch.dict(os.environ, {"ALPHA_VANTAGE_KEY": "legacy", "ALPHA_VANTAGE_API_KEY": "abc123"}):
            assert PriceFallback()._alpha_key == "legacy"

    def test_no_key_leaves_level1_inactive(self):
        from data.price_fallback import PriceFallback
        with patch.dict(os.environ, {"ALPHA_VANTAGE_KEY": "", "ALPHA_VANTAGE_API_KEY": ""}):
            assert PriceFallback()._alpha_key == ""
