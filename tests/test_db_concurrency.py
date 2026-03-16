"""
Tests for concurrent SQLite access in the Database class.

Verifies that the WAL mode, busy_timeout, threading lock, and retry
decorator together prevent "database is locked" errors when multiple
threads perform writes simultaneously.
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from storage.database import Database


@pytest.fixture
def shared_db(tmp_path):
    """A single Database instance shared across threads."""
    return Database(str(tmp_path / "concurrent_test.db"))


class TestConcurrentWrites:
    """Verify zero lock errors under concurrent write pressure."""

    def test_ten_concurrent_writes_zero_lock_errors(self, shared_db: Database):
        """
        Spawn 10 threads that each insert a run row concurrently.
        All 10 must succeed without any 'database is locked' error.
        """
        num_threads = 10
        errors: list[Exception] = []
        results: list[int] = []
        lock = threading.Lock()

        def _writer(thread_id: int) -> int:
            try:
                row_id = shared_db.log_run(
                    ticker=f"T{thread_id:02d}",
                    headlines_fetched=thread_id,
                    headlines_analysed=thread_id,
                    avg_score=0.5,
                    signal="HOLD",
                )
                return row_id
            except Exception as exc:
                with lock:
                    errors.append(exc)
                raise

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(_writer, i) for i in range(num_threads)]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception:
                    pass  # already captured in errors list

        assert errors == [], f"Expected zero errors, got: {errors}"
        assert len(results) == num_threads
        # Each row ID should be unique
        assert len(set(results)) == num_threads

        # Verify all rows exist in the database
        rows = shared_db.get_recent_runs(limit=num_threads)
        assert len(rows) == num_threads

    def test_mixed_read_write_concurrency(self, shared_db: Database):
        """
        Mix reads and writes from 10 threads concurrently.
        Ensures readers don't block writers and vice versa.
        """
        num_threads = 10
        errors: list[Exception] = []
        lock = threading.Lock()

        def _mixed_worker(thread_id: int) -> None:
            try:
                # Write
                shared_db.log_run(
                    ticker=f"MIX{thread_id:02d}",
                    headlines_fetched=1,
                    headlines_analysed=1,
                    avg_score=0.1 * thread_id,
                    signal="BUY" if thread_id % 2 == 0 else "SELL",
                )
                # Read immediately after
                shared_db.get_recent_runs(limit=5)
            except Exception as exc:
                with lock:
                    errors.append(exc)
                raise

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(_mixed_worker, i) for i in range(num_threads)]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception:
                    pass

        assert errors == [], f"Expected zero errors, got: {errors}"
        rows = shared_db.get_recent_runs(limit=num_threads)
        assert len(rows) == num_threads

    def test_concurrent_different_write_methods(self, shared_db: Database):
        """
        Hit multiple different write methods concurrently to ensure
        the class-level lock serializes across all of them.
        """
        num_threads = 10
        errors: list[Exception] = []
        lock = threading.Lock()

        def _diverse_writer(thread_id: int) -> None:
            try:
                if thread_id % 3 == 0:
                    shared_db.log_run(
                        ticker=f"DIV{thread_id}",
                        headlines_fetched=1,
                        headlines_analysed=1,
                        avg_score=0.0,
                        signal="HOLD",
                    )
                elif thread_id % 3 == 1:
                    shared_db.log_technical_signal(
                        ticker=f"DIV{thread_id}",
                        signal="BUY",
                        reasoning="test concurrent access",
                    )
                else:
                    shared_db.log_emergency_stop(
                        action="test",
                        reason=f"concurrency test thread {thread_id}",
                    )
            except Exception as exc:
                with lock:
                    errors.append(exc)
                raise

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(_diverse_writer, i) for i in range(num_threads)]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception:
                    pass

        assert errors == [], f"Expected zero errors, got: {errors}"

    def test_wal_mode_enabled(self, shared_db: Database):
        """Verify that WAL journal mode is active on new connections."""
        conn = shared_db._connect()
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal", f"Expected WAL mode, got {mode!r}"
        finally:
            conn.close()

    def test_busy_timeout_set(self, shared_db: Database):
        """Verify that busy_timeout is set to 5000ms on new connections."""
        conn = shared_db._connect()
        try:
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert timeout == 5000, f"Expected 5000ms timeout, got {timeout}"
        finally:
            conn.close()
