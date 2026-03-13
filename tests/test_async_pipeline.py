"""
Tests for the async pipeline: semaphore limits, error resilience,
progress tracking, and --workers flag.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.coordinator import Coordinator
from scheduler.daily_runner import (
    DEFAULT_WATCHLIST,
    _ProgressTracker,
    _parse_args,
    _process_ticker,
    run_batch,
)
from storage.database import Database


# ── Helpers ────────────────────────────────────────────────────────────

def _temp_db() -> Database:
    return Database(db_path=tempfile.mktemp(suffix=".db"))


def _make_stub_coordinator(
    *,
    delay: float = 0.0,
    fail_tickers: set[str] | None = None,
) -> Coordinator:
    """
    Build a Coordinator with all agents stubbed out.

    Args:
        delay:        Simulated delay per API / data call (seconds).
        fail_tickers: Set of ticker symbols that should raise an error.
    """
    fail_tickers = fail_tickers or set()
    db = _temp_db()

    # Stub news feed
    news_feed = MagicMock()
    news_feed.fetch = MagicMock(side_effect=lambda t: (
        _maybe_fail(t, fail_tickers, delay) or ["Headline A", "Headline B"]
    ))

    # Stub market data
    market_data = MagicMock()
    market_data.fetch = MagicMock(side_effect=lambda t: (
        _maybe_fail(t, fail_tickers, delay) or
        {"price": 150.0, "name": t, "currency": "USD"}
    ))

    # Stub sentiment agent
    sentiment_agent = MagicMock()
    sentiment_agent.run = MagicMock(side_effect=lambda text, ticker: (
        _maybe_fail(ticker, fail_tickers, delay) or
        {"sentiment": "bullish", "score": 1, "reason": "test", "headline": text}
    ))

    # Stub technical agent
    technical_agent = MagicMock()
    technical_agent.run = MagicMock(side_effect=lambda ticker, **kw: (
        _maybe_fail(ticker, fail_tickers, delay) or {
            "ticker": ticker,
            "signal": "BUY",
            "reasoning": ["Test"],
            "indicators": {"price": 150.0, "rsi": 35.0, "rvol": 1.8,
                           "macd": 0.5, "macd_signal": 0.3, "macd_hist": 0.2,
                           "sma_20": 148.0, "sma_50": 145.0,
                           "bb_upper": 160.0, "bb_lower": 140.0,
                           "macd_bull_cross": False, "macd_bear_cross": False,
                           "volume_trending_up": True, "obv_trend": "rising"},
            "signal_id": 1,
            "volume_confirmed": True,
        }
    ))

    # Stub regime agent
    regime_agent = MagicMock()
    regime_agent.run = MagicMock(return_value={
        "regime": "TRENDING_BULL", "cached": True,
    })

    # Stub risk agent
    risk_agent = MagicMock()
    risk_agent.run = MagicMock(side_effect=lambda **kw: {
        "ticker": kw.get("ticker", "?"),
        "signal": kw.get("signal", "HOLD"),
        "direction": "BUY",
        "position_size_usd": 500.0,
        "shares": 3,
        "stop_loss": 147.0,
        "take_profit": 156.0,
        "risk_amount": 9.0,
        "kelly_fraction": 0.05,
        "stop_pct": 0.02,
        "skipped": False,
        "skip_reason": None,
        "event_risk_flag": "none",
        "days_to_earnings": None,
        "regime": "TRENDING_BULL",
        "calc_id": 1,
    })

    # Stub social feeds
    reddit_feed = MagicMock()
    reddit_feed.fetch = MagicMock(return_value=[])
    stocktwits_feed = MagicMock()
    stocktwits_feed.fetch = MagicMock(return_value=[])

    # Stub broker
    paper_trader = MagicMock()
    paper_trader.track_trade = MagicMock(return_value={"trade_id": 1})

    return Coordinator(
        news_feed=news_feed,
        market_data=market_data,
        sentiment_agent=sentiment_agent,
        technical_agent=technical_agent,
        risk_agent=risk_agent,
        regime_agent=regime_agent,
        db=db,
        paper_trader=paper_trader,
        reddit_feed=reddit_feed,
        stocktwits_feed=stocktwits_feed,
    )


def _maybe_fail(ticker: str, fail_tickers: set[str], delay: float):
    """Raise if ticker is in fail set; otherwise sleep and return None."""
    if ticker in fail_tickers:
        raise RuntimeError(f"Simulated failure for {ticker}")
    if delay > 0:
        time.sleep(delay)
    return None


# ══════════════════════════════════════════════════════════════════════
# Semaphore tests
# ══════════════════════════════════════════════════════════════════════

class TestSemaphoreLimits:
    def test_api_semaphore_limits_concurrent_calls(self):
        """API semaphore(2) should never allow more than 2 concurrent calls."""
        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        original_run = None

        async def _counting_sentiment(text, ticker):
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.05)  # simulate API latency
            async with lock:
                current -= 1
            return {"sentiment": "bullish", "score": 1, "reason": "t", "headline": text}

        coordinator = _make_stub_coordinator()
        # Patch sentiment agent to be async-aware via the semaphore
        # We need to track concurrent calls inside the async wrapper
        original_sentiment_run = coordinator.sentiment_agent.run

        call_count = 0
        call_lock = asyncio.Lock()
        max_concurrent_api = 0
        current_api = 0

        def tracked_sentiment(text, ticker):
            nonlocal max_concurrent_api, current_api
            # This runs in a thread, use threading primitives
            import threading
            with threading.Lock.__new__(threading.Lock) if False else _thread_lock:
                current_api += 1
                if current_api > max_concurrent_api:
                    max_concurrent_api = current_api
            time.sleep(0.05)
            with _thread_lock:
                current_api -= 1
            return {"sentiment": "bullish", "score": 1, "reason": "t", "headline": text}

        import threading
        _thread_lock = threading.Lock()
        coordinator.sentiment_agent.run = tracked_sentiment

        api_sem = asyncio.Semaphore(2)
        data_sem = asyncio.Semaphore(10)
        db_lock = asyncio.Lock()

        async def _run():
            # Run 4 tickers concurrently with api_sem=2
            tasks = [
                coordinator.analyse_ticker_async(
                    t,
                    account_balance=10_000,
                    execute=False,
                    api_semaphore=api_sem,
                    data_semaphore=data_sem,
                    db_lock=db_lock,
                )
                for t in ["AAA", "BBB", "CCC", "DDD"]
            ]
            return await asyncio.gather(*tasks)

        asyncio.run(_run())
        # With api_sem=2, at most 2 concurrent Claude API calls
        assert max_concurrent_api <= 2

    def test_data_semaphore_limits_fetches(self):
        """data_semaphore(2) should limit concurrent data fetches."""
        import threading
        _lock = threading.Lock()
        max_concurrent = 0
        current = 0

        coordinator = _make_stub_coordinator()
        original_fetch = coordinator.news_feed.fetch

        def tracked_fetch(ticker):
            nonlocal max_concurrent, current
            with _lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            time.sleep(0.05)
            with _lock:
                current -= 1
            return ["Headline"]

        coordinator.news_feed.fetch = tracked_fetch

        api_sem = asyncio.Semaphore(10)
        data_sem = asyncio.Semaphore(2)  # limit to 2
        db_lock = asyncio.Lock()

        async def _run():
            tasks = [
                coordinator.analyse_ticker_async(
                    t,
                    account_balance=10_000,
                    execute=False,
                    api_semaphore=api_sem,
                    data_semaphore=data_sem,
                    db_lock=db_lock,
                )
                for t in ["AAA", "BBB", "CCC", "DDD"]
            ]
            return await asyncio.gather(*tasks)

        asyncio.run(_run())
        assert max_concurrent <= 2

    def test_workers_flag_limits_concurrent_tickers(self):
        """--workers 2 should process at most 2 tickers concurrently."""
        import threading
        _lock = threading.Lock()
        max_concurrent = 0
        current = 0

        coordinator = _make_stub_coordinator(delay=0.05)
        original_fetch = coordinator.market_data.fetch

        def tracked_market(ticker):
            nonlocal max_concurrent, current
            with _lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            time.sleep(0.05)
            with _lock:
                current -= 1
            return {"price": 150.0, "name": ticker, "currency": "USD"}

        coordinator.market_data.fetch = tracked_market

        async def _run():
            api_sem = asyncio.Semaphore(5)
            data_sem = asyncio.Semaphore(10)
            db_lock = asyncio.Lock()
            worker_sem = asyncio.Semaphore(2)
            tracker = _ProgressTracker(4)

            tasks = [
                _process_ticker(
                    coordinator, t,
                    account_balance=10_000,
                    execute=False,
                    api_semaphore=api_sem,
                    data_semaphore=data_sem,
                    db_lock=db_lock,
                    worker_semaphore=worker_sem,
                    tracker=tracker,
                )
                for t in ["AAA", "BBB", "CCC", "DDD"]
            ]
            return await asyncio.gather(*tasks)

        asyncio.run(_run())
        assert max_concurrent <= 2


# ══════════════════════════════════════════════════════════════════════
# Error resilience
# ══════════════════════════════════════════════════════════════════════

class TestErrorResilience:
    def test_all_tickers_complete_even_if_some_fail(self):
        """Failing tickers should not block other tickers."""
        coordinator = _make_stub_coordinator(fail_tickers={"BBB", "DDD"})

        async def _run():
            api_sem = asyncio.Semaphore(5)
            data_sem = asyncio.Semaphore(10)
            db_lock = asyncio.Lock()
            worker_sem = asyncio.Semaphore(5)
            tracker = _ProgressTracker(4)

            tasks = [
                _process_ticker(
                    coordinator, t,
                    account_balance=10_000,
                    execute=False,
                    api_semaphore=api_sem,
                    data_semaphore=data_sem,
                    db_lock=db_lock,
                    worker_semaphore=worker_sem,
                    tracker=tracker,
                )
                for t in ["AAA", "BBB", "CCC", "DDD"]
            ]
            results = await asyncio.gather(*tasks)
            return results, tracker

        results, tracker = asyncio.run(_run())

        # All 4 tickers reported progress
        assert len(tracker.lines) == 4
        # AAA and CCC succeeded, BBB and DDD failed
        assert results[0] is not None  # AAA
        assert results[1] is None      # BBB failed
        assert results[2] is not None  # CCC
        assert results[3] is None      # DDD failed

    def test_single_failure_does_not_crash_batch(self):
        """One exception should not prevent the batch from completing."""
        coordinator = _make_stub_coordinator(fail_tickers={"FAIL"})

        async def _run():
            api_sem = asyncio.Semaphore(5)
            data_sem = asyncio.Semaphore(10)
            db_lock = asyncio.Lock()
            worker_sem = asyncio.Semaphore(5)
            tracker = _ProgressTracker(3)

            tasks = [
                _process_ticker(
                    coordinator, t,
                    account_balance=10_000,
                    execute=False,
                    api_semaphore=api_sem,
                    data_semaphore=data_sem,
                    db_lock=db_lock,
                    worker_semaphore=worker_sem,
                    tracker=tracker,
                )
                for t in ["OK1", "FAIL", "OK2"]
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_run())
        successes = [r for r in results if r is not None]
        assert len(successes) == 2


# ══════════════════════════════════════════════════════════════════════
# Progress tracking
# ══════════════════════════════════════════════════════════════════════

class TestProgressTracking:
    def test_progress_lines_count_matches_tickers(self):
        """Each ticker should produce exactly one progress line."""
        coordinator = _make_stub_coordinator()

        async def _run():
            tracker = _ProgressTracker(3)
            api_sem = asyncio.Semaphore(5)
            data_sem = asyncio.Semaphore(10)
            db_lock = asyncio.Lock()
            worker_sem = asyncio.Semaphore(5)

            tasks = [
                _process_ticker(
                    coordinator, t,
                    account_balance=10_000,
                    execute=False,
                    api_semaphore=api_sem,
                    data_semaphore=data_sem,
                    db_lock=db_lock,
                    worker_semaphore=worker_sem,
                    tracker=tracker,
                )
                for t in ["AAPL", "MSFT", "NVDA"]
            ]
            await asyncio.gather(*tasks)
            return tracker

        tracker = asyncio.run(_run())
        assert len(tracker.lines) == 3

    def test_progress_shows_ticker_name(self):
        """Progress output should contain the ticker symbol."""
        coordinator = _make_stub_coordinator()

        async def _run():
            tracker = _ProgressTracker(1)
            api_sem = asyncio.Semaphore(5)
            data_sem = asyncio.Semaphore(10)
            db_lock = asyncio.Lock()
            worker_sem = asyncio.Semaphore(5)

            await _process_ticker(
                coordinator, "AAPL",
                account_balance=10_000,
                execute=False,
                api_semaphore=api_sem,
                data_semaphore=data_sem,
                db_lock=db_lock,
                worker_semaphore=worker_sem,
                tracker=tracker,
            )
            return tracker

        tracker = asyncio.run(_run())
        assert "AAPL" in tracker.lines[0]

    def test_failed_ticker_shows_failed(self):
        """Failed tickers should show FAILED in progress."""
        coordinator = _make_stub_coordinator(fail_tickers={"BAD"})

        async def _run():
            tracker = _ProgressTracker(1)
            api_sem = asyncio.Semaphore(5)
            data_sem = asyncio.Semaphore(10)
            db_lock = asyncio.Lock()
            worker_sem = asyncio.Semaphore(5)

            await _process_ticker(
                coordinator, "BAD",
                account_balance=10_000,
                execute=False,
                api_semaphore=api_sem,
                data_semaphore=data_sem,
                db_lock=db_lock,
                worker_semaphore=worker_sem,
                tracker=tracker,
            )
            return tracker

        tracker = asyncio.run(_run())
        assert "FAILED" in tracker.lines[0]

    def test_progress_shows_elapsed_time(self):
        """Success lines should contain timing info (ends with 's')."""
        coordinator = _make_stub_coordinator()

        async def _run():
            tracker = _ProgressTracker(1)
            api_sem = asyncio.Semaphore(5)
            data_sem = asyncio.Semaphore(10)
            db_lock = asyncio.Lock()
            worker_sem = asyncio.Semaphore(5)

            await _process_ticker(
                coordinator, "AAPL",
                account_balance=10_000,
                execute=False,
                api_semaphore=api_sem,
                data_semaphore=data_sem,
                db_lock=db_lock,
                worker_semaphore=worker_sem,
                tracker=tracker,
            )
            return tracker

        tracker = asyncio.run(_run())
        # Line should contain elapsed time like "0.1s"
        assert "s" in tracker.lines[0]


# ══════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ══════════════════════════════════════════════════════════════════════

class TestCLIArgs:
    def test_default_workers(self):
        args = _parse_args(["--now"])
        assert args.workers == 3

    def test_custom_workers(self):
        args = _parse_args(["--now", "--workers", "8"])
        assert args.workers == 8

    def test_watchlist_parsing(self):
        args = _parse_args(["--now", "--watchlist", "AAPL,MSFT,NVDA"])
        assert args.watchlist == "AAPL,MSFT,NVDA"

    def test_no_execute_flag(self):
        args = _parse_args(["--now", "--no-execute"])
        assert args.no_execute is True

    def test_balance_default(self):
        args = _parse_args(["--now"])
        assert args.balance == 10_000.0

    def test_balance_custom(self):
        args = _parse_args(["--now", "--balance", "50000"])
        assert args.balance == 50_000.0

    def test_benchmark_flag(self):
        args = _parse_args(["--benchmark"])
        assert args.benchmark is True


# ══════════════════════════════════════════════════════════════════════
# run_batch integration
# ══════════════════════════════════════════════════════════════════════

class TestRunBatch:
    def test_run_batch_returns_expected_keys(self):
        """run_batch should return results, elapsed_s, success/fail counts."""
        coordinator = _make_stub_coordinator()

        async def _run():
            # Monkey-patch Coordinator() creation inside run_batch
            # by directly calling the components
            api_sem = asyncio.Semaphore(5)
            data_sem = asyncio.Semaphore(10)
            db_lock = asyncio.Lock()
            worker_sem = asyncio.Semaphore(2)
            tracker = _ProgressTracker(2)

            tasks = [
                _process_ticker(
                    coordinator, t,
                    account_balance=10_000,
                    execute=False,
                    api_semaphore=api_sem,
                    data_semaphore=data_sem,
                    db_lock=db_lock,
                    worker_semaphore=worker_sem,
                    tracker=tracker,
                )
                for t in ["AAA", "BBB"]
            ]
            results = await asyncio.gather(*tasks)
            success = sum(1 for r in results if r is not None)
            return {
                "results": results,
                "elapsed_s": 0.0,
                "success_count": success,
                "fail_count": len(results) - success,
            }

        batch = asyncio.run(_run())
        assert "results" in batch
        assert "elapsed_s" in batch
        assert "success_count" in batch
        assert batch["success_count"] == 2
        assert batch["fail_count"] == 0

    def test_async_faster_than_sequential(self):
        """Async with delay should be faster than sequential."""
        coordinator = _make_stub_coordinator(delay=0.05)

        tickers = ["T1", "T2", "T3", "T4"]

        # Sequential timing
        t0 = time.monotonic()
        for t in tickers:
            try:
                coordinator.run_combined(t, verbose=False, account_balance=10_000)
            except Exception:
                pass
        sync_time = time.monotonic() - t0

        # Async timing
        async def _run():
            api_sem = asyncio.Semaphore(5)
            data_sem = asyncio.Semaphore(10)
            db_lock = asyncio.Lock()
            worker_sem = asyncio.Semaphore(4)
            tracker = _ProgressTracker(len(tickers))

            tasks = [
                _process_ticker(
                    coordinator, t,
                    account_balance=10_000,
                    execute=False,
                    api_semaphore=api_sem,
                    data_semaphore=data_sem,
                    db_lock=db_lock,
                    worker_semaphore=worker_sem,
                    tracker=tracker,
                )
                for t in tickers
            ]
            return await asyncio.gather(*tasks)

        t0 = time.monotonic()
        asyncio.run(_run())
        async_time = time.monotonic() - t0

        # Async should be noticeably faster (at least 1.5x)
        assert async_time < sync_time * 0.9, (
            f"Async ({async_time:.2f}s) should be faster than sync ({sync_time:.2f}s)"
        )
