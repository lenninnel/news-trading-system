"""
SQLite → PostgreSQL migration script.

Copies all data from the local SQLite database to the target PostgreSQL
database (configured via DATABASE_URL).

Usage
-----
    # Dry-run (shows row counts, makes no changes)
    python3 deployment/migrate_to_postgres.py --dry-run

    # Full migration
    python3 deployment/migrate_to_postgres.py

    # Migrate from a specific SQLite file
    python3 deployment/migrate_to_postgres.py --sqlite-path /path/to/db.db

The script is idempotent: it will not duplicate rows that already exist in
the target database (it truncates each table before inserting).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pg_conn(dsn: str):
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("ERROR: psycopg2-binary is required. Install with: pip install psycopg2-binary")
        sys.exit(1)
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


def _sqlite_tables(sqlite_path: str) -> list[str]:
    conn = sqlite3.connect(sqlite_path)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _copy_table(
    sqlite_path: str,
    pg_dsn: str,
    table: str,
    dry_run: bool,
) -> int:
    """Copy all rows from one SQLite table to PostgreSQL. Returns row count."""
    sq_conn = sqlite3.connect(sqlite_path)
    sq_conn.row_factory = sqlite3.Row
    rows = sq_conn.execute(f"SELECT * FROM {table}").fetchall()
    sq_conn.close()

    if not rows:
        return 0

    columns = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(columns))
    col_list = ", ".join(columns)
    values = [tuple(row[c] for c in columns) for row in rows]

    if dry_run:
        return len(values)

    pg_conn = _get_pg_conn(pg_dsn)
    try:
        cur = pg_conn.cursor()
        # Truncate first so the migration is idempotent
        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
        cur.executemany(
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
            values,
        )
        pg_conn.commit()
    except Exception:
        pg_conn.rollback()
        raise
    finally:
        pg_conn.close()

    return len(values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate SQLite → PostgreSQL")
    parser.add_argument(
        "--sqlite-path",
        default=os.environ.get("DB_PATH", "news_trading.db"),
        help="Path to the SQLite database file (default: news_trading.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show row counts without writing to PostgreSQL",
    )
    args = parser.parse_args()

    sqlite_path = args.sqlite_path
    if not Path(sqlite_path).exists():
        print(f"ERROR: SQLite file not found: {sqlite_path}")
        sys.exit(1)

    pg_dsn = os.environ.get("DATABASE_URL", "")
    if not pg_dsn:
        print("ERROR: DATABASE_URL environment variable is not set.")
        sys.exit(1)

    # Ensure the PostgreSQL schema is up-to-date before copying data
    if not args.dry_run:
        print("Initialising PostgreSQL schema …")
        from storage.database import Database
        Database()  # __init__ calls _init_schema()
        print("Schema ready.")

    tables = _sqlite_tables(sqlite_path)
    print(f"\nSQLite source : {sqlite_path}")
    print(f"PostgreSQL DSN: {pg_dsn[:30]}…")
    print(f"Dry-run       : {args.dry_run}")
    print(f"\nTables found  : {', '.join(tables)}\n")

    total = 0
    for table in tables:
        try:
            count = _copy_table(sqlite_path, pg_dsn, table, dry_run=args.dry_run)
            status = "would copy" if args.dry_run else "copied"
            print(f"  {table:<40} {status} {count:>6} rows")
            total += count
        except Exception as exc:
            print(f"  {table:<40} ERROR: {exc}")

    print(f"\n{'[DRY-RUN] Total rows' if args.dry_run else 'Total rows migrated'}: {total}")
    if not args.dry_run:
        print("\nMigration complete. Verify the data in your PostgreSQL console.")


if __name__ == "__main__":
    main()
