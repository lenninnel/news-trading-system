"""
SQLite persistence layer for the News Trading System.

Provides a Database class that initialises the schema on first use and
exposes methods for logging analysis runs and individual headline scores.
All writes use context managers so connections are always cleanly closed.

Schema
------
runs
    id              INTEGER  Primary key
    ticker          TEXT     Stock ticker symbol
    headlines_fetched INTEGER Total headlines retrieved from NewsAPI
    headlines_analysed INTEGER Headlines successfully scored
    avg_score       REAL     Average sentiment score (-1.0 to +1.0)
    signal          TEXT     BUY / SELL / HOLD
    created_at      TEXT     ISO-8601 UTC timestamp

headline_scores
    id              INTEGER  Primary key
    run_id          INTEGER  Foreign key → runs.id
    headline        TEXT     Original headline text
    sentiment       TEXT     bullish / bearish / neutral
    score           INTEGER  Numeric score (+1 / 0 / -1)
    reason          TEXT     One-sentence explanation from Claude
"""

import sqlite3
from datetime import datetime, timezone

from config.settings import DB_PATH


class Database:
    """Thin wrapper around a SQLite database for logging trading analysis."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        """
        Initialise the database and create tables if they don't exist.

        Args:
            db_path: Path to the SQLite file. Defaults to settings.DB_PATH.
        """
        self.db_path = db_path
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker              TEXT    NOT NULL,
                    headlines_fetched   INTEGER NOT NULL DEFAULT 0,
                    headlines_analysed  INTEGER NOT NULL DEFAULT 0,
                    avg_score           REAL,
                    signal              TEXT,
                    created_at          TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS headline_scores (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      INTEGER NOT NULL REFERENCES runs(id),
                    headline    TEXT    NOT NULL,
                    sentiment   TEXT    NOT NULL,
                    score       INTEGER NOT NULL,
                    reason      TEXT
                );
                """
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_run(
        self,
        ticker: str,
        headlines_fetched: int,
        headlines_analysed: int,
        avg_score: float,
        signal: str,
    ) -> int:
        """
        Persist a completed analysis run and return its auto-generated ID.

        Args:
            ticker:               Stock ticker symbol.
            headlines_fetched:    Number of headlines retrieved.
            headlines_analysed:   Number of headlines successfully scored.
            avg_score:            Aggregated sentiment score.
            signal:               Trading signal (BUY / SELL / HOLD).

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs
                    (ticker, headlines_fetched, headlines_analysed, avg_score, signal, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticker, headlines_fetched, headlines_analysed, avg_score, signal, now),
            )
            return cur.lastrowid

    def log_headline_score(
        self,
        run_id: int,
        headline: str,
        sentiment: str,
        score: int,
        reason: str,
    ) -> None:
        """
        Persist a single headline score linked to a run.

        Args:
            run_id:    ID of the parent run (from log_run).
            headline:  Original headline text.
            sentiment: bullish / bearish / neutral.
            score:     Numeric score (+1 / 0 / -1).
            reason:    Claude's one-sentence explanation.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO headline_scores (run_id, headline, sentiment, score, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, headline, sentiment, score, reason),
            )

    def get_recent_runs(self, limit: int = 10) -> list[dict]:
        """
        Return the most recent analysis runs, newest first.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of dicts, each representing one row from the runs table.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_scores_for_run(self, run_id: int) -> list[dict]:
        """
        Return all headline scores associated with a given run.

        Args:
            run_id: Primary key of the run.

        Returns:
            List of dicts, each representing one headline_scores row.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM headline_scores WHERE run_id = ?", (run_id,)
            ).fetchall()
            return [dict(row) for row in rows]
