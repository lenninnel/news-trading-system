"""
Dashboard Load Test — simulates concurrent user sessions against the data layer.

Tests
-----
1. Data seeding    — inserts 1000+ rows across all key tables
2. Concurrent load — 10 threads make dashboard-style queries simultaneously
3. Response time   — all queries must complete in <2s (P95)
4. Memory usage    — tracks RSS before/after under load
5. Bottleneck report — slowest queries identified

The test is self-contained: it creates a temp SQLite database, seeds it,
runs the load test, then deletes everything.  No network calls required.

Usage
-----
    python3 tests/load_test_dashboard.py                   # default 10 users
    python3 tests/load_test_dashboard.py --users 20        # 20 concurrent users
    python3 tests/load_test_dashboard.py --rows 5000       # seed more data
    python3 tests/load_test_dashboard.py --target-url http://localhost:8501  # HTTP too
    pytest  tests/load_test_dashboard.py -v
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Thresholds ────────────────────────────────────────────────────────────────
TARGET_P95_MS     = 2_000   # 2 s
CONCURRENT_USERS  = 10
SEED_ROWS         = 1_000   # signals / headlines

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL",
           "AMZN", "META", "NFLX", "AMD",  "INTC"]
SIGNALS = ["BUY", "SELL", "HOLD", "STRONG BUY", "STRONG SELL",
           "WEAK BUY", "WEAK SELL", "CONFLICTING"]
STRATEGIES = ["momentum", "mean_reversion", "swing"]


# ══════════════════════════════════════════════════════════════════════════════
# Data seeder
# ══════════════════════════════════════════════════════════════════════════════

def seed_database(db_path: str, n_rows: int = SEED_ROWS) -> dict[str, int]:
    """
    Insert n_rows synthetic rows across all dashboard-queried tables.
    Returns dict of {table: row_count}.
    """
    from storage.database import Database
    db = Database(db_path)

    counts: dict[str, int] = {}

    # ── runs + headline_scores ────────────────────────────────────────────────
    run_ids = []
    for i in range(n_rows):
        ticker  = TICKERS[i % len(TICKERS)]
        score   = random.uniform(-1.0, 1.0)
        signal  = random.choice(["BUY", "SELL", "HOLD"])
        ts      = (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat()
        run_id  = db.log_run(
            ticker=ticker,
            headlines_fetched=10,
            headlines_analysed=8,
            avg_score=score,
            signal=signal,
        )
        run_ids.append(run_id)
        for j in range(3):
            db.log_headline_score(
                run_id=run_id,
                headline=f"Headline {i}-{j} for {ticker}",
                sentiment=random.choice(["bullish", "bearish", "neutral"]),
                score=random.choice([1, 0, -1]),
                reason="Auto-generated",
            )
    counts["runs"]            = n_rows
    counts["headline_scores"] = n_rows * 3

    # ── technical_signals ─────────────────────────────────────────────────────
    ts_ids = []
    for i in range(n_rows):
        ticker = TICKERS[i % len(TICKERS)]
        ts_id  = db.log_technical_signal(
            ticker=ticker,
            signal=random.choice(["BUY", "SELL", "HOLD"]),
            reasoning="RSI oversold; MACD crossover",
            rsi=random.uniform(20, 80),
            macd=random.uniform(-2, 2),
            macd_signal=random.uniform(-2, 2),
            macd_hist=random.uniform(-1, 1),
            sma_20=random.uniform(140, 160),
            sma_50=random.uniform(135, 155),
            bb_upper=random.uniform(165, 175),
            bb_lower=random.uniform(135, 145),
            price=random.uniform(140, 170),
        )
        ts_ids.append(ts_id)
    counts["technical_signals"] = n_rows

    # ── combined_signals ──────────────────────────────────────────────────────
    for i in range(n_rows):
        db.log_combined_signal(
            ticker=TICKERS[i % len(TICKERS)],
            combined_signal=random.choice(SIGNALS),
            sentiment_signal=random.choice(["BUY", "SELL", "HOLD"]),
            technical_signal=random.choice(["BUY", "SELL", "HOLD"]),
            sentiment_score=random.uniform(-1, 1),
            confidence=random.uniform(0, 1),
            run_id=run_ids[i],
            technical_id=ts_ids[i],
        )
    counts["combined_signals"] = n_rows

    # ── strategy_signals + strategy_performance ───────────────────────────────
    for i in range(n_rows):
        run_at = (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat()
        db.log_strategy_signal(
            ticker=TICKERS[i % len(TICKERS)],
            strategy=random.choice(STRATEGIES),
            signal=random.choice(["BUY", "SELL", "HOLD"]),
            confidence=random.uniform(40, 95),
            timeframe=random.choice(["1-3 days", "3-5 days", "1-2 weeks"]),
        )

        db.log_strategy_performance(
            ticker=TICKERS[i % len(TICKERS)],
            run_at=run_at,
            combined_signal=random.choice(["BUY", "SELL", "HOLD"]),
            ensemble_confidence=random.uniform(50, 90),
            voting_method=random.choice(["unanimous", "majority", "conflicting"]),
            momentum_signal=random.choice(["BUY", "SELL", "HOLD"]),
            momentum_confidence=random.uniform(40, 90),
            mean_rev_signal=random.choice(["BUY", "SELL", "HOLD"]),
            mean_rev_confidence=random.uniform(40, 90),
            swing_signal=random.choice(["BUY", "SELL", "HOLD"]),
            swing_confidence=random.uniform(40, 90),
        )
    counts["strategy_signals"]     = n_rows
    counts["strategy_performance"] = n_rows

    # ── scheduler_logs ────────────────────────────────────────────────────────
    for i in range(min(n_rows, 100)):
        run_at = (datetime.now(timezone.utc) - timedelta(days=i)).isoformat()
        db.log_scheduler_run(
            run_at=run_at,
            tickers=TICKERS,
            success_count=random.randint(3, 10),
            fail_count=random.randint(0, 2),
            elapsed_s=random.uniform(30, 120),
            total_elapsed_s=random.uniform(30, 120),
            errors=[],
            status="success",
            notes="Daily run summary",
        )
    counts["scheduler_logs"] = min(n_rows, 100)

    return counts


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard query workload
# ══════════════════════════════════════════════════════════════════════════════

# Queries mirroring what the dashboard executes per page
_DASHBOARD_QUERIES = {
    "overview_signals":
        "SELECT * FROM combined_signals ORDER BY id DESC LIMIT 50",
    "overview_runs":
        "SELECT * FROM runs ORDER BY id DESC LIMIT 20",
    "strategy_recent":
        "SELECT * FROM strategy_performance ORDER BY id DESC LIMIT 30",
    "strategy_signals_detail":
        "SELECT s.*, p.combined_signal, p.ensemble_confidence "
        "FROM strategy_signals s "
        "LEFT JOIN strategy_performance p ON s.ticker = p.ticker "
        "ORDER BY s.id DESC LIMIT 50",
    "technical_latest":
        "SELECT * FROM technical_signals ORDER BY id DESC LIMIT 20",
    "scheduler_history":
        "SELECT * FROM scheduler_logs ORDER BY id DESC LIMIT 30",
    "ticker_history_AAPL":
        "SELECT * FROM runs WHERE ticker='AAPL' ORDER BY id DESC LIMIT 50",
    "headline_scores_recent":
        "SELECT h.*, r.ticker FROM headline_scores h "
        "JOIN runs r ON h.run_id = r.id ORDER BY h.id DESC LIMIT 100",
    "signal_distribution":
        "SELECT signal, COUNT(*) as cnt FROM runs GROUP BY signal",
    "strategy_performance_summary":
        "SELECT ticker, AVG(ensemble_confidence) as avg_conf, COUNT(*) as runs "
        "FROM strategy_performance GROUP BY ticker ORDER BY avg_conf DESC",
}


def _run_query(db_path: str, query_name: str, sql: str) -> tuple[str, float, int, str | None]:
    """Execute one dashboard query. Returns (name, elapsed_ms, row_count, error|None)."""
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    t0 = time.perf_counter()
    error = None
    row_count = 0
    try:
        rows = conn.execute(sql).fetchall()
        row_count = len(rows)
    except Exception as exc:
        error = str(exc)
    finally:
        conn.close()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return query_name, elapsed_ms, row_count, error


def _user_session(
    db_path: str,
    user_id: int,
    results: list,
    lock: threading.Lock,
) -> None:
    """Simulate one dashboard user running the full query workload."""
    session_times: list[float] = []
    for name, sql in _DASHBOARD_QUERIES.items():
        qname, ms, rows, err = _run_query(db_path, name, sql)
        with lock:
            results.append({"user": user_id, "query": qname,
                             "ms": ms, "rows": rows, "error": err})
        session_times.append(ms)
    # Small random think time between pages (10–100ms)
    time.sleep(random.uniform(0.01, 0.1))


# ══════════════════════════════════════════════════════════════════════════════
# HTTP load test (optional, if a live server URL is provided)
# ══════════════════════════════════════════════════════════════════════════════

def _http_load_test(target_url: str, n_users: int = 5) -> dict:
    """Simple HTTP GET load test against the dashboard URL."""
    import urllib.request
    import urllib.error

    results: list[dict] = []
    errors  = 0

    def _fetch(user_id: int) -> None:
        nonlocal errors
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(target_url, headers={"User-Agent": "LoadTest/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
                ms = (time.perf_counter() - t0) * 1000
                results.append({"user": user_id, "ms": ms, "status": resp.status})
        except Exception as exc:
            errors += 1
            results.append({"user": user_id, "ms": 9999, "status": 0, "error": str(exc)})

    threads = [threading.Thread(target=_fetch, args=(i,)) for i in range(n_users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    times = [r["ms"] for r in results]
    times.sort()
    p95 = times[int(len(times) * 0.95)] if times else 0
    return {"total": len(results), "errors": errors,
            "p50_ms": times[len(times) // 2] if times else 0,
            "p95_ms": p95, "max_ms": max(times) if times else 0}


# ══════════════════════════════════════════════════════════════════════════════
# Load test runner
# ══════════════════════════════════════════════════════════════════════════════

def run_load_test(
    db_path: str,
    n_users: int = CONCURRENT_USERS,
    target_url: str | None = None,
) -> dict:
    """
    Run the full load test. Returns a report dict.
    """
    results: list[dict] = []
    lock    = threading.Lock()

    threads = [
        threading.Thread(
            target=_user_session,
            args=(db_path, i, results, lock),
            name=f"user-{i}",
        )
        for i in range(n_users)
    ]

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_elapsed = (time.perf_counter() - t0) * 1000

    # Compute per-query stats
    from collections import defaultdict
    query_times: dict[str, list[float]] = defaultdict(list)
    query_errors: dict[str, int]        = defaultdict(int)
    for r in results:
        query_times[r["query"]].append(r["ms"])
        if r["error"]:
            query_errors[r["query"]] += 1

    all_times = sorted([r["ms"] for r in results])
    p50 = all_times[len(all_times) // 2]
    p95 = all_times[int(len(all_times) * 0.95)]
    p99 = all_times[int(len(all_times) * 0.99)]

    per_query = {}
    for name, times in query_times.items():
        times.sort()
        per_query[name] = {
            "p50_ms": times[len(times) // 2],
            "p95_ms": times[int(len(times) * 0.95)],
            "max_ms": max(times),
            "errors": query_errors.get(name, 0),
        }

    http_report = None
    if target_url:
        http_report = _http_load_test(target_url, n_users=min(n_users, 5))

    return {
        "users":          n_users,
        "total_queries":  len(results),
        "total_elapsed_ms": total_elapsed,
        "p50_ms":         p50,
        "p95_ms":         p95,
        "p99_ms":         p99,
        "max_ms":         max(all_times),
        "errors":         sum(query_errors.values()),
        "per_query":      per_query,
        "http":           http_report,
        "p95_pass":       p95 <= TARGET_P95_MS,
    }


def print_report(report: dict, n_seed_rows: int) -> None:
    W = 70
    print(f"\n{'═' * W}")
    print(f"  Dashboard Load Test Report")
    print(f"{'═' * W}")
    print(f"  Seed rows       : {n_seed_rows:,}")
    print(f"  Concurrent users: {report['users']}")
    print(f"  Total queries   : {report['total_queries']:,}")
    print(f"  Total wall time : {report['total_elapsed_ms']:.0f} ms")
    print(f"  Errors          : {report['errors']}")
    print()
    print(f"  Response Time Distribution")
    print(f"  {'Metric':<10}  {'Value':>10}")
    print(f"  {'─' * 25}")
    print(f"  {'P50':<10}  {report['p50_ms']:>8.1f} ms")
    print(f"  {'P95':<10}  {report['p95_ms']:>8.1f} ms  "
          f"{'✓ PASS' if report['p95_pass'] else f'✗ FAIL (>{TARGET_P95_MS}ms)'}")
    print(f"  {'P99':<10}  {report['p99_ms']:>8.1f} ms")
    print(f"  {'Max':<10}  {report['max_ms']:>8.1f} ms")

    print(f"\n  Per-Query Breakdown  (sorted by P95 desc)")
    print(f"  {'Query':<40}  {'P50':>7}  {'P95':>7}  {'Max':>7}  Err")
    print(f"  {'─' * 68}")
    for name, stats in sorted(report["per_query"].items(),
                               key=lambda x: x[1]["p95_ms"], reverse=True):
        flag = "⚠" if stats["p95_ms"] > TARGET_P95_MS else " "
        print(
            f"  {flag} {name:<38}  {stats['p50_ms']:>6.0f}ms"
            f"  {stats['p95_ms']:>6.0f}ms  {stats['max_ms']:>6.0f}ms"
            f"  {stats['errors']}"
        )

    if report.get("http"):
        h = report["http"]
        print(f"\n  HTTP Test ({h['total']} requests)")
        print(f"  P50={h['p50_ms']:.0f}ms  P95={h['p95_ms']:.0f}ms  "
              f"Errors={h['errors']}")

    print(f"\n{'═' * W}")
    verdict = "✓ PASS" if report["p95_pass"] and report["errors"] == 0 else "✗ FAIL"
    print(f"  Result: {verdict}")
    print(f"{'═' * W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Pytest integration
# ══════════════════════════════════════════════════════════════════════════════

class TestDashboardLoad(unittest.TestCase):
    """pytest-compatible wrapper around the load test."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir  = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, "load_test.db")
        os.environ["DB_PATH"] = cls.db_path
        cls.n_rows  = 1_000
        print(f"\nSeeding {cls.n_rows} rows into {cls.db_path} …", flush=True)
        cls.seed_counts = seed_database(cls.db_path, cls.n_rows)
        print(f"Seed complete: {sum(cls.seed_counts.values()):,} total rows")

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)
        os.environ.pop("DB_PATH", None)

    def test_seed_counts(self):
        """Verify 1000+ rows exist across key tables."""
        self.assertGreaterEqual(self.seed_counts["runs"], 1_000)
        self.assertGreaterEqual(self.seed_counts["headline_scores"], 3_000)
        self.assertGreaterEqual(self.seed_counts["technical_signals"], 1_000)

    def test_all_queries_complete(self):
        """All dashboard queries should run without error."""
        report = run_load_test(self.db_path, n_users=CONCURRENT_USERS)
        self.assertEqual(report["errors"], 0,
                         f"Query errors: {report['errors']}")

    def test_p95_response_time(self):
        """P95 response time across all queries must be <2s."""
        report = run_load_test(self.db_path, n_users=CONCURRENT_USERS)
        self.assertLessEqual(
            report["p95_ms"], TARGET_P95_MS,
            f"P95={report['p95_ms']:.0f}ms exceeds target {TARGET_P95_MS}ms",
        )

    def test_concurrent_read_safety(self):
        """10 threads reading simultaneously should not corrupt results."""
        errors = []
        lock   = threading.Lock()

        def _read():
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                conn.row_factory = sqlite3.Row
                conn.execute("SELECT COUNT(*) FROM runs").fetchone()
                conn.close()
            except Exception as exc:
                with lock:
                    errors.append(str(exc))

        threads = [threading.Thread(target=_read) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"Concurrent read errors: {errors}")

    def test_memory_usage(self):
        """Memory increase under load should be <100 MB."""
        try:
            import psutil
            proc       = psutil.Process()
            mem_before = proc.memory_info().rss / 1024 / 1024  # MB
            run_load_test(self.db_path, n_users=CONCURRENT_USERS)
            mem_after  = proc.memory_info().rss / 1024 / 1024
            delta      = mem_after - mem_before
            print(f"\n  Memory: {mem_before:.0f} MB → {mem_after:.0f} MB (Δ {delta:+.0f} MB)")
            self.assertLess(delta, 100, f"Memory increase {delta:.0f} MB exceeds 100 MB limit")
        except ImportError:
            self.skipTest("psutil not installed — skipping memory test")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard Load Test")
    parser.add_argument("--users",  type=int, default=CONCURRENT_USERS,
                        help=f"Concurrent users (default: {CONCURRENT_USERS})")
    parser.add_argument("--rows",   type=int, default=SEED_ROWS,
                        help=f"Seed row count (default: {SEED_ROWS})")
    parser.add_argument("--target-url", default=None,
                        help="Optional Streamlit URL for HTTP load test (e.g. http://localhost:8501)")
    parser.add_argument("--keep-db", action="store_true",
                        help="Keep temp DB after test (for manual inspection)")
    args = parser.parse_args()

    tmpdir  = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "load_test.db")
    os.environ["DB_PATH"] = db_path

    try:
        print(f"Seeding {args.rows:,} rows …")
        t0 = time.perf_counter()
        seed_counts = seed_database(db_path, args.rows)
        seed_ms     = (time.perf_counter() - t0) * 1000
        print(f"Seed complete in {seed_ms:.0f}ms: {sum(seed_counts.values()):,} total rows")

        print(f"\nRunning load test with {args.users} concurrent user(s) …")
        report = run_load_test(db_path, args.users, args.target_url)
        print_report(report, args.rows)

        sys.exit(0 if report["p95_pass"] and report["errors"] == 0 else 1)

    finally:
        if not args.keep_db:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
