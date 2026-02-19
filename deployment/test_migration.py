"""
PostgreSQL Migration Dry-Run & Verification Script.

Steps
-----
1. Backup the local SQLite DB
2. Spin up a temporary PostgreSQL container via Docker
3. Run the full migration (SQLite → PostgreSQL)
4. Verify every table exists and row counts match
5. Run representative SELECT queries from each table
6. Tear down the test container
7. Print a detailed migration report

Usage
-----
    python3 deployment/test_migration.py
    python3 deployment/test_migration.py --sqlite-path /path/to/other.db
    python3 deployment/test_migration.py --skip-docker  # use DATABASE_URL directly
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ──────────────────────────────────────────────────────────────────
PG_CONTAINER  = "nt_migration_test_pg"
PG_IMAGE      = "postgres:15-alpine"
PG_USER       = "trader"
PG_PASSWORD   = "trader_test"
PG_DB         = "news_trading_test"
PG_PORT       = "5435"   # avoid colliding with existing local pg on 5432
PG_DSN        = f"postgresql://{PG_USER}:{PG_PASSWORD}@localhost:{PG_PORT}/{PG_DB}"

BACKUP_DIR    = PROJECT_ROOT / "backups"
REPORT_LINES: list[str] = []


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    print(msg)
    REPORT_LINES.append(msg)


def _ok(label: str, detail: str = "") -> None:
    line = f"  ✓  {label:<45} {detail}"
    _log(line)


def _fail(label: str, detail: str = "") -> None:
    line = f"  ✗  {label:<45} {detail}"
    _log(line)


def _header(title: str) -> None:
    _log(f"\n{'═' * 60}")
    _log(f"  {title}")
    _log('═' * 60)


def _sqlite_tables(path: str) -> dict[str, int]:
    """Return {table_name: row_count} for all tables in SQLite."""
    conn = sqlite3.connect(path)
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    result = {}
    for t in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            result[t] = count
        except Exception:
            result[t] = -1
    conn.close()
    return result


def _pg_table_count(conn, table: str) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return cur.fetchone()["count"] if hasattr(cur.fetchone, "__getitem__") else cur.fetchone()[0]  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# Docker helpers
# ══════════════════════════════════════════════════════════════════════════════

def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _start_pg_container() -> None:
    """Start a temporary PostgreSQL container."""
    # Remove any stale container
    subprocess.run(
        ["docker", "rm", "-f", PG_CONTAINER],
        capture_output=True,
    )
    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", PG_CONTAINER,
            "-e", f"POSTGRES_USER={PG_USER}",
            "-e", f"POSTGRES_PASSWORD={PG_PASSWORD}",
            "-e", f"POSTGRES_DB={PG_DB}",
            "-p", f"{PG_PORT}:5432",
            PG_IMAGE,
        ],
        check=True,
        capture_output=True,
    )
    _log(f"  Container '{PG_CONTAINER}' started on port {PG_PORT}.")

    # Wait for readiness
    _log("  Waiting for PostgreSQL to be ready …")
    for i in range(30):
        result = subprocess.run(
            [
                "docker", "exec", PG_CONTAINER,
                "pg_isready", "-U", PG_USER, "-d", PG_DB,
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            _log(f"  PostgreSQL ready after {i * 2}s.")
            return
        time.sleep(2)
    raise RuntimeError("PostgreSQL container did not become ready within 60s.")


def _stop_pg_container() -> None:
    subprocess.run(
        ["docker", "rm", "-f", PG_CONTAINER],
        capture_output=True,
    )
    _log(f"  Container '{PG_CONTAINER}' removed.")


# ══════════════════════════════════════════════════════════════════════════════
# Verification queries
# ══════════════════════════════════════════════════════════════════════════════

_SPOT_QUERIES: list[tuple[str, str]] = [
    ("runs",               "SELECT id, ticker, signal FROM runs LIMIT 5"),
    ("headline_scores",    "SELECT id, run_id, sentiment FROM headline_scores LIMIT 5"),
    ("technical_signals",  "SELECT id, ticker, signal, rsi FROM technical_signals LIMIT 5"),
    ("combined_signals",   "SELECT id, ticker, combined_signal FROM combined_signals LIMIT 5"),
    ("risk_calculations",  "SELECT id, ticker, position_size_usd FROM risk_calculations LIMIT 5"),
    ("strategy_signals",   "SELECT id, ticker, strategy, signal FROM strategy_signals LIMIT 5"),
    ("strategy_performance","SELECT id, ticker, combined_signal FROM strategy_performance LIMIT 5"),
    ("scheduler_logs",     "SELECT id, status, signals_generated FROM scheduler_logs LIMIT 5"),
    ("screener_results",   "SELECT id, ticker, hotness FROM screener_results LIMIT 5"),
    ("backtest_results",   "SELECT id, ticker, sharpe_ratio FROM backtest_results LIMIT 5"),
    ("backtest_strategy_comparison",
                           "SELECT id, ticker, strategy FROM backtest_strategy_comparison LIMIT 5"),
    ("portfolio_snapshots","SELECT id, open_positions, total_value FROM portfolio_snapshots LIMIT 5"),
    ("portfolio_violations","SELECT id, ticker, violation_type FROM portfolio_violations LIMIT 5"),
    ("price_alerts",       "SELECT id, ticker, alert_type FROM price_alerts LIMIT 5"),
    ("optimization_results","SELECT id, ticker, best_sharpe FROM optimization_results LIMIT 5"),
    ("health_checks",      "SELECT id, overall_ok FROM health_checks LIMIT 5"),
    ("emergency_stops",    "SELECT id, action FROM emergency_stops LIMIT 5"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="PostgreSQL Migration Test")
    parser.add_argument(
        "--sqlite-path",
        default=os.environ.get("DB_PATH", "news_trading.db"),
        help="SQLite source file (default: news_trading.db)",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Use DATABASE_URL directly instead of spinning up a container.",
    )
    args = parser.parse_args()

    sqlite_path = args.sqlite_path
    started_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_pass  = 0
    total_fail  = 0
    use_docker  = not args.skip_docker

    _header(f"PostgreSQL Migration Test  —  {started_at}")

    # ── 1. Check SQLite source ────────────────────────────────────────────────
    _header("Step 1 — SQLite source")
    if not Path(sqlite_path).exists():
        _fail("SQLite file exists", sqlite_path)
        _log("\nNo SQLite database found. Run the system once to generate data first.")
        _log("Example: python3 scheduler/daily_runner.py --now")
        sys.exit(1)

    sqlite_counts = _sqlite_tables(sqlite_path)
    _ok("SQLite file found", sqlite_path)
    total_rows_sqlite = sum(v for v in sqlite_counts.values() if v >= 0)
    _ok("Total tables", str(len(sqlite_counts)))
    _ok("Total rows", str(total_rows_sqlite))
    for table, count in sqlite_counts.items():
        _log(f"    {table:<45} {count:>6} rows")
    total_pass += 1

    # ── 2. Backup ─────────────────────────────────────────────────────────────
    _header("Step 2 — Backup")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = BACKUP_DIR / f"migration_test_backup_{ts}.db"
    shutil.copy2(sqlite_path, backup)
    _ok("Backup created", str(backup))
    total_pass += 1

    # ── 3. Start PostgreSQL ───────────────────────────────────────────────────
    _header("Step 3 — PostgreSQL")
    pg_dsn = PG_DSN

    if args.skip_docker:
        pg_dsn = os.environ.get("DATABASE_URL", "")
        if not pg_dsn:
            _fail("DATABASE_URL not set (required when --skip-docker)")
            sys.exit(1)
        _ok("Using DATABASE_URL", pg_dsn[:40] + "…")
        total_pass += 1
    elif not _docker_available():
        _fail("Docker not available", "install Docker or use --skip-docker with DATABASE_URL")
        sys.exit(1)
    else:
        try:
            _start_pg_container()
            _ok("Test PostgreSQL container started")
            total_pass += 1
        except Exception as exc:
            _fail("Container start failed", str(exc))
            sys.exit(1)

    # ── 4. Run migration ──────────────────────────────────────────────────────
    _header("Step 4 — Migration")
    env = {**os.environ, "DATABASE_URL": pg_dsn, "DB_PATH": sqlite_path}
    result = subprocess.run(
        [sys.executable, "deployment/migrate_to_postgres.py"],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        _ok("Migration completed successfully")
        total_pass += 1
        for line in result.stdout.splitlines():
            _log(f"    {line}")
    else:
        _fail("Migration failed")
        total_fail += 1
        _log(result.stdout)
        _log(result.stderr)

    # ── 5. Row count verification ─────────────────────────────────────────────
    _header("Step 5 — Row Count Verification")
    try:
        import psycopg2
        import psycopg2.extras
        pg_conn = psycopg2.connect(pg_dsn, cursor_factory=psycopg2.extras.RealDictCursor)
        pg_cur  = pg_conn.cursor()

        for table, sq_count in sqlite_counts.items():
            try:
                pg_cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
                pg_count = pg_cur.fetchone()["count"]
                if sq_count == pg_count:
                    _ok(f"{table}", f"SQLite={sq_count}  PG={pg_count}  ✓")
                    total_pass += 1
                else:
                    _fail(f"{table}", f"SQLite={sq_count}  PG={pg_count}  MISMATCH")
                    total_fail += 1
            except Exception as exc:
                _fail(f"{table}", f"query failed: {exc}")
                total_fail += 1

    except ImportError:
        _fail("psycopg2 not installed — skipping row count check")
        total_fail += 1
    except Exception as exc:
        _fail("PostgreSQL connection failed", str(exc))
        total_fail += 1

    # ── 6. Query verification ─────────────────────────────────────────────────
    _header("Step 6 — Query Verification")
    try:
        pg_cur  # already connected from step 5
        for table, sql in _SPOT_QUERIES:
            try:
                pg_cur.execute(sql)
                rows = pg_cur.fetchall()
                _ok(f"SELECT on {table}", f"{len(rows)} row(s) returned")
                total_pass += 1
            except Exception as exc:
                _fail(f"SELECT on {table}", str(exc))
                total_fail += 1
        pg_conn.close()
    except NameError:
        _log("  Skipped (no PG connection from step 5)")

    # ── 7. Tear down ──────────────────────────────────────────────────────────
    if use_docker and not args.skip_docker:
        _header("Step 7 — Teardown")
        try:
            _stop_pg_container()
            _ok("Test container removed")
            total_pass += 1
        except Exception as exc:
            _fail("Container teardown failed", str(exc))

    # ── Report ────────────────────────────────────────────────────────────────
    _header("Migration Report")
    _log(f"  Started at   : {started_at}")
    _log(f"  Finished at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log(f"  SQLite source: {sqlite_path}")
    _log(f"  Tables       : {len(sqlite_counts)}")
    _log(f"  SQLite rows  : {total_rows_sqlite}")
    _log(f"  Checks passed: {total_pass}")
    _log(f"  Checks failed: {total_fail}")

    report_path = PROJECT_ROOT / "backups" / f"migration_report_{ts}.txt"
    report_path.write_text("\n".join(REPORT_LINES), encoding="utf-8")
    _log(f"\n  Report saved: {report_path}")

    if total_fail > 0:
        print(f"\n\033[31m✗ Migration test FAILED ({total_fail} failures)\033[0m")
        sys.exit(1)
    else:
        print(f"\n\033[32m✓ Migration test PASSED (all {total_pass} checks)\033[0m")


if __name__ == "__main__":
    main()
