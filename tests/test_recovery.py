"""
Unit tests for the error-recovery subsystem.

Tests
-----
TestCircuitBreaker       — state transitions, thresholds, half-open probe
TestAPIRecovery          — per-service retry configs, 401 short-circuit,
                           circuit-open fast-fail
TestResponseCache        — TTL expiry, hit/miss, per-service clear
TestNetworkMonitor       — degraded-mode activation and restore
TestStateRecovery        — CheckpointManager save/load/validate/resume/clear
TestSentimentFallback    — rule-based keyword scorer correctness
TestNewsFeedFallback     — cache hit when live fetch fails
TestTechnicalFallback    — indicator cache hit; empty-dict HOLD on cache miss
TestRecoveryLogDB        — log_recovery_event() writes to recovery_log table
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Ensure project root is on sys.path ────────────────────────────────────────
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DB_PATH", "/tmp/test_recovery.db")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("NEWSAPI_KEY", "dummy")


# ══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker(unittest.TestCase):

    def setUp(self):
        from utils.api_recovery import CircuitBreaker
        self.CB = CircuitBreaker

    def _make(self, threshold=5, reset_timeout=300):
        return self.CB("test_svc", failure_threshold=threshold, reset_timeout=reset_timeout)

    def test_initial_state_is_closed(self):
        cb = self._make()
        self.assertEqual(cb.state, self.CB.CLOSED)
        self.assertEqual(cb.failures, 0)

    def test_opens_at_failure_threshold(self):
        cb = self._make(threshold=3)
        for _ in range(2):
            cb.record_failure()
        self.assertEqual(cb.state, self.CB.CLOSED)
        cb.record_failure()
        self.assertEqual(cb.state, self.CB.OPEN)

    def test_allows_calls_when_closed(self):
        cb = self._make()
        self.assertTrue(cb.allow_call())

    def test_blocks_calls_when_open(self):
        cb = self._make(threshold=1)
        cb.record_failure()
        self.assertEqual(cb.state, self.CB.OPEN)
        self.assertFalse(cb.allow_call())

    def test_transitions_to_half_open_after_timeout(self):
        cb = self._make(threshold=1, reset_timeout=0)
        cb.record_failure()
        # Immediately after failure with reset_timeout=0 the circuit may already
        # be HALF_OPEN; accept either OPEN or HALF_OPEN here.
        self.assertIn(cb.state, (self.CB.OPEN, self.CB.HALF_OPEN))
        time.sleep(0.01)
        # After sleep (timeout elapsed) accessing .state must return HALF_OPEN
        self.assertEqual(cb.state, self.CB.HALF_OPEN)

    def test_closes_on_success_from_half_open(self):
        cb = self._make(threshold=1, reset_timeout=0)
        cb.record_failure()
        time.sleep(0.01)
        _ = cb.state  # trigger HALF_OPEN transition
        cb.record_success()
        self.assertEqual(cb.state, self.CB.CLOSED)
        self.assertEqual(cb.failures, 0)

    def test_reopens_on_failure_from_half_open(self):
        # Use a long reset_timeout so the circuit stays OPEN after re-opening
        cb = self._make(threshold=1, reset_timeout=300)
        cb.record_failure()
        # Manually force it into HALF_OPEN for the test
        with cb._lock:
            cb._state = self.CB.HALF_OPEN
        cb.record_failure()
        # After a failure from HALF_OPEN it must go back to OPEN
        with cb._lock:
            self.assertEqual(cb._state, self.CB.OPEN)

    def test_success_resets_failure_count(self):
        cb = self._make(threshold=5)
        for _ in range(3):
            cb.record_failure()
        cb.record_success()
        self.assertEqual(cb.failures, 0)
        self.assertEqual(cb.state, self.CB.CLOSED)

    def test_to_dict_contains_expected_keys(self):
        cb = self._make()
        d = cb.to_dict()
        for key in ("name", "state", "failures", "failure_threshold", "reset_timeout_s"):
            self.assertIn(key, d)


# ══════════════════════════════════════════════════════════════════════════════
# APIRecovery
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIRecovery(unittest.TestCase):

    def setUp(self):
        from utils.api_recovery import APIRecovery
        # Reset all circuits between tests
        APIRecovery._circuits.clear()
        APIRecovery._db = None
        self.AR = APIRecovery

    def test_successful_call_returns_value(self):
        result = self.AR.call("newsapi", lambda: 42)
        self.assertEqual(result, 42)

    def test_call_retries_on_generic_exception(self):
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("transient error")
            return "ok"

        with patch("utils.api_recovery.time.sleep"):  # skip real sleeps
            result = self.AR.call("yfinance", flaky)
        self.assertEqual(result, "ok")
        self.assertEqual(call_count[0], 3)

    def test_401_raises_unauthorized_immediately(self):
        from utils.api_recovery import UnauthorizedError

        def bad_auth():
            exc = Exception("Unauthorized")
            exc.status_code = 401
            raise exc

        with patch("utils.api_recovery.time.sleep"):
            with self.assertRaises(UnauthorizedError):
                self.AR.call("newsapi", bad_auth)

    def test_circuit_opens_after_repeated_failures(self):
        from utils.api_recovery import CircuitOpenError

        def always_fail():
            raise RuntimeError("always fails")

        # Run enough failing calls to open the circuit (threshold = 5)
        for _ in range(6):
            try:
                with patch("utils.api_recovery.time.sleep"):
                    self.AR.call("newsapi", always_fail)
            except Exception:
                pass

        # Circuit should now be open; next call raises CircuitOpenError
        with self.assertRaises(CircuitOpenError):
            self.AR.call("newsapi", lambda: None)

    def test_all_retries_exhausted_raises_original(self):
        def always_fail():
            raise ValueError("boom")

        with patch("utils.api_recovery.time.sleep"):
            with self.assertRaises(ValueError):
                self.AR.call("database", always_fail)

    def test_get_status_returns_dict(self):
        self.AR.call("newsapi", lambda: 1)
        status = self.AR.get_status()
        self.assertIn("newsapi", status)
        self.assertIn("state", status["newsapi"])

    def test_reset_circuit_closes_open_circuit(self):
        from utils.api_recovery import CircuitBreaker
        cb = self.AR.get_circuit("anthropic")
        for _ in range(5):
            cb.record_failure()
        self.assertEqual(cb.state, CircuitBreaker.OPEN)
        self.AR.reset_circuit("anthropic")
        self.assertEqual(cb.state, CircuitBreaker.CLOSED)


# ══════════════════════════════════════════════════════════════════════════════
# Response Cache
# ══════════════════════════════════════════════════════════════════════════════

class TestResponseCache(unittest.TestCase):

    def setUp(self):
        from utils.network_recovery import ResponseCache
        self.cache = ResponseCache(max_age_seconds=1.0)

    def test_set_and_get_fresh_entry(self):
        self.cache.set("newsapi", "AAPL", ["headline 1"])
        value, hit = self.cache.get("newsapi", "AAPL")
        self.assertTrue(hit)
        self.assertEqual(value, ["headline 1"])

    def test_get_missing_key_returns_no_hit(self):
        value, hit = self.cache.get("newsapi", "MISSING")
        self.assertFalse(hit)
        self.assertIsNone(value)

    def test_stale_entry_is_a_miss(self):
        stale_cache = __import__(
            "utils.network_recovery", fromlist=["ResponseCache"]
        ).ResponseCache(max_age_seconds=0.01)
        stale_cache.set("newsapi", "TSLA", ["old"])
        time.sleep(0.05)
        _, hit = stale_cache.get("newsapi", "TSLA")
        self.assertFalse(hit)

    def test_clear_all(self):
        self.cache.set("newsapi", "AAPL", ["h"])
        self.cache.set("yfinance", "AAPL", {"rsi": 50})
        self.cache.clear()
        _, hit1 = self.cache.get("newsapi", "AAPL")
        _, hit2 = self.cache.get("yfinance", "AAPL")
        self.assertFalse(hit1)
        self.assertFalse(hit2)

    def test_clear_by_service(self):
        self.cache.set("newsapi", "AAPL", ["h"])
        self.cache.set("yfinance", "AAPL", {"rsi": 50})
        self.cache.clear("newsapi")
        _, hit_news = self.cache.get("newsapi", "AAPL")
        _, hit_yf   = self.cache.get("yfinance", "AAPL")
        self.assertFalse(hit_news)
        self.assertTrue(hit_yf)

    def test_has_returns_true_for_fresh_entry(self):
        self.cache.set("newsapi", "NVDA", ["x"])
        self.assertTrue(self.cache.has("newsapi", "NVDA"))
        self.assertFalse(self.cache.has("newsapi", "NOPE"))

    def test_stats_counts_fresh_and_stale(self):
        stale_cache = __import__(
            "utils.network_recovery", fromlist=["ResponseCache"]
        ).ResponseCache(max_age_seconds=0.01)
        stale_cache.set("a", "1", "fresh-initially")
        time.sleep(0.05)
        stale_cache.set("b", "2", "still-fresh")
        stats = stale_cache.stats()
        self.assertEqual(stats["total_entries"], 2)
        self.assertEqual(stats["fresh_entries"], 1)
        self.assertEqual(stats["stale_entries"], 1)


# ══════════════════════════════════════════════════════════════════════════════
# NetworkMonitor
# ══════════════════════════════════════════════════════════════════════════════

class TestNetworkMonitor(unittest.TestCase):

    def setUp(self):
        from utils.network_recovery import NetworkMonitor
        self.NM = NetworkMonitor
        # Reset class state
        self.NM._degraded      = False
        self.NM._offline_since = None
        self.NM._last_check_at = None
        self.NM._db            = None

    def tearDown(self):
        self.NM._degraded      = False
        self.NM._offline_since = None
        self.NM._last_check_at = None

    def test_is_degraded_initially_false(self):
        self.assertFalse(self.NM.is_degraded())

    def test_check_and_update_sets_degraded_when_offline(self):
        with patch.object(self.NM, "is_online", return_value=False):
            result = self.NM.check_and_update(force=True)
        self.assertFalse(result)
        self.assertTrue(self.NM.is_degraded())

    def test_check_and_update_clears_degraded_on_restore(self):
        with patch.object(self.NM, "is_online", return_value=False):
            self.NM.check_and_update(force=True)
        self.assertTrue(self.NM.is_degraded())

        with patch.object(self.NM, "is_online", return_value=True):
            result = self.NM.check_and_update(force=True)
        self.assertTrue(result)
        self.assertFalse(self.NM.is_degraded())

    def test_throttle_skips_probe_within_interval(self):
        with patch.object(self.NM, "is_online", return_value=True) as mock_probe:
            self.NM.check_and_update(force=True)
            self.NM.check_and_update(force=False)  # should not re-probe
        # is_online called only once due to throttle
        self.assertEqual(mock_probe.call_count, 1)

    def test_status_dict_has_expected_keys(self):
        status = self.NM.status()
        for key in ("degraded", "offline_since", "last_check", "cache_stats"):
            self.assertIn(key, status)


# ══════════════════════════════════════════════════════════════════════════════
# CheckpointManager
# ══════════════════════════════════════════════════════════════════════════════

class TestStateRecovery(unittest.TestCase):

    def _make(self, tmpdir, **kwargs) -> "CheckpointManager":
        from utils.state_recovery import CheckpointManager
        return CheckpointManager(
            name="test_run",
            checkpoint_dir=tmpdir,
            save_interval=kwargs.get("save_interval", 1),
            max_age_hours=kwargs.get("max_age_hours", 24),
        )

    def test_load_returns_none_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make(tmpdir)
            self.assertIsNone(mgr.load())

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make(tmpdir)
            mgr.update("completed_tickers", ["AAPL", "NVDA"])
            mgr.save()

            mgr2 = self._make(tmpdir)
            state = mgr2.load()
            self.assertIsNotNone(state)
            self.assertEqual(state["completed_tickers"], ["AAPL", "NVDA"])

    def test_auto_save_triggers_at_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make(tmpdir, save_interval=3)
            mgr.update("k", "v1")
            self.assertFalse(mgr.path.exists())
            mgr.update("k", "v2")
            self.assertFalse(mgr.path.exists())
            mgr.update("k", "v3")  # 3rd call → auto-save
            self.assertTrue(mgr.path.exists())

    def test_get_pending_returns_all_when_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make(tmpdir)
            all_t = ["AAPL", "NVDA", "TSLA"]
            self.assertEqual(mgr.get_pending(all_t), all_t)

    def test_get_pending_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make(tmpdir)
            mgr.update("completed_tickers", ["AAPL", "NVDA"])
            mgr.save()

            mgr2 = self._make(tmpdir)
            pending = mgr2.get_pending(["AAPL", "NVDA", "TSLA"])
            self.assertEqual(pending, ["TSLA"])

    def test_validate_rejects_wrong_version(self):
        import json
        from utils.state_recovery import _SCHEMA_VERSION
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make(tmpdir)
            bad = {
                "_version":  _SCHEMA_VERSION + 999,
                "_name":     "test_run",
                "_saved_at": "2099-01-01T00:00:00+00:00",
                "_op_count": 1,
            }
            mgr.path.write_text(json.dumps(bad), encoding="utf-8")
            self.assertIsNone(mgr.load())

    def test_validate_rejects_stale_checkpoint(self):
        import json
        from datetime import datetime, timezone, timedelta
        from utils.state_recovery import _SCHEMA_VERSION
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make(tmpdir, max_age_hours=1)
            old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            stale = {
                "_version":  _SCHEMA_VERSION,
                "_name":     "test_run",
                "_saved_at": old_ts,
                "_op_count": 1,
                "completed_tickers": [],
            }
            mgr.path.write_text(json.dumps(stale), encoding="utf-8")
            self.assertIsNone(mgr.load())

    def test_clear_removes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make(tmpdir)
            mgr.update("k", "v")
            mgr.save()
            self.assertTrue(mgr.path.exists())
            mgr.clear()
            self.assertFalse(mgr.path.exists())

    def test_internal_keys_stripped_on_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make(tmpdir)
            mgr.update("tickers", ["A", "B"])
            mgr.save()
            state = mgr.load()
            for key in state:
                self.assertFalse(key.startswith("_"), f"Internal key leaked: {key}")


# ══════════════════════════════════════════════════════════════════════════════
# Sentiment rule-based fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestSentimentFallback(unittest.TestCase):

    def setUp(self):
        from agents.sentiment_agent import _rule_based_sentiment
        self.rb = _rule_based_sentiment

    def test_bullish_headline(self):
        r = self.rb("Apple beats expectations with record profit and revenue growth")
        self.assertEqual(r["sentiment"], "bullish")
        self.assertEqual(r["score"], 1)
        self.assertTrue(r["degraded"])

    def test_bearish_headline(self):
        r = self.rb("Company faces bankruptcy and massive fraud investigation")
        self.assertEqual(r["sentiment"], "bearish")
        self.assertEqual(r["score"], -1)
        self.assertTrue(r["degraded"])

    def test_neutral_headline(self):
        r = self.rb("Company announces Q3 2024 earnings call schedule")
        self.assertEqual(r["degraded"], True)
        # Score should be in valid range
        self.assertIn(r["score"], (-1, 0, 1))

    def test_always_returns_degraded_true(self):
        for headline in [
            "Tesla surges on strong delivery numbers",
            "Stock crashes amid market turmoil",
            "CEO steps down after board meeting",
        ]:
            r = self.rb(headline)
            self.assertTrue(r["degraded"], f"degraded should be True for: {headline}")

    def test_returns_all_required_keys(self):
        r = self.rb("Any headline here")
        for key in ("sentiment", "score", "reason", "headline", "degraded"):
            self.assertIn(key, r)


# ══════════════════════════════════════════════════════════════════════════════
# NewsFeed cache fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestNewsFeedFallback(unittest.TestCase):

    def test_returns_cached_headlines_when_api_fails(self):
        from data.news_feed import NewsFeed
        from utils.network_recovery import ResponseCache

        fake = ["Cached headline 1", "Cached headline 2"]
        cache = ResponseCache(max_age_seconds=3600)
        cache.set("newsapi", "headlines:AAPL", fake)

        feed = NewsFeed.__new__(NewsFeed)
        feed.api_key       = "dummy"
        feed.max_headlines = 10

        with patch("utils.network_recovery._cache", cache):
            with patch("utils.network_recovery.NetworkMonitor.check_and_update", return_value=True):
                with patch("utils.network_recovery.NetworkMonitor.is_degraded", return_value=False):
                    with patch("utils.api_recovery.APIRecovery.call", side_effect=Exception("down")):
                        result = feed.fetch("AAPL")

        self.assertEqual(result, fake)

    def test_returns_empty_list_when_no_cache_and_api_fails(self):
        from data.news_feed import NewsFeed
        from utils.network_recovery import ResponseCache

        empty_cache = ResponseCache(max_age_seconds=3600)

        feed = NewsFeed.__new__(NewsFeed)
        feed.api_key       = "dummy"
        feed.max_headlines = 10

        with patch("utils.network_recovery._cache", empty_cache):
            with patch("utils.network_recovery.NetworkMonitor.check_and_update", return_value=True):
                with patch("utils.network_recovery.NetworkMonitor.is_degraded", return_value=False):
                    with patch("utils.api_recovery.APIRecovery.call", side_effect=Exception("down")):
                        result = feed.fetch("AAPL")

        self.assertEqual(result, [])


# ══════════════════════════════════════════════════════════════════════════════
# TechnicalAgent cache fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestTechnicalFallback(unittest.TestCase):

    def _make_fake_indicators(self) -> dict:
        return {
            "rsi": 55.0, "macd": 0.1, "macd_signal": 0.05, "macd_hist": 0.05,
            "macd_bull_cross": False, "macd_bear_cross": False,
            "sma_20": 150.0, "sma_50": 148.0,
            "bb_upper": 160.0, "bb_lower": 140.0, "price": 150.0,
        }

    def test_cache_hit_returns_cached_indicators(self):
        from utils.network_recovery import ResponseCache

        fake_ind  = self._make_fake_indicators()
        cache     = ResponseCache(max_age_seconds=3600)
        cache.set("yfinance", "indicators:AAPL", fake_ind)

        cached, hit = cache.get("yfinance", "indicators:AAPL")
        self.assertTrue(hit)
        self.assertEqual(cached["rsi"], 55.0)

    def test_empty_cache_miss(self):
        from utils.network_recovery import ResponseCache

        cache = ResponseCache(max_age_seconds=3600)
        _, hit = cache.get("yfinance", "indicators:MISSING")
        self.assertFalse(hit)

    def test_fallback_indicators_are_none_filled(self):
        """When no cache, _fallback_indicators returns all-None dict."""
        from agents.technical_agent import TechnicalAgent
        from storage.database import Database
        from utils.network_recovery import ResponseCache

        empty_cache = ResponseCache(max_age_seconds=3600)
        db          = MagicMock(spec=Database)
        agent       = TechnicalAgent(db=db)

        with patch("utils.network_recovery._cache", empty_cache):
            ind, degraded = agent._fallback_indicators("FAKE", "indicators:FAKE", "test error")

        self.assertTrue(degraded)
        self.assertIsNone(ind["rsi"])
        self.assertIsNone(ind["price"])


# ══════════════════════════════════════════════════════════════════════════════
# Database recovery_log
# ══════════════════════════════════════════════════════════════════════════════

class TestRecoveryLogDB(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test_recovery.db")
        os.environ["DB_PATH"] = self.db_path
        from storage.database import Database
        self.db = Database(self.db_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_log_recovery_event_returns_id(self):
        row_id = self.db.log_recovery_event(
            service="newsapi",
            event_type="retry",
            ticker="AAPL",
            attempt=2,
            error_msg="HTTP 429",
            recovery_action="rate_limited_backoff_60s",
            duration_ms=1234,
            success=False,
        )
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

    def test_recovery_log_table_exists(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        self.assertIn("recovery_log", tables)

    def test_multiple_events_persist_correctly(self):
        import sqlite3

        services = ["newsapi", "anthropic", "yfinance", "network", "checkpoint"]
        for svc in services:
            self.db.log_recovery_event(
                service=svc,
                event_type="degraded_mode",
                success=False,
            )

        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM recovery_log").fetchone()[0]
        conn.close()
        self.assertEqual(count, len(services))

    def test_optional_fields_can_be_null(self):
        # All optional fields omitted — should not raise
        row_id = self.db.log_recovery_event(
            service="database",
            event_type="circuit_open",
        )
        self.assertGreater(row_id, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
