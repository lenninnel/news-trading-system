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

technical_signals
    id              INTEGER  Primary key
    ticker          TEXT     Stock ticker symbol
    signal          TEXT     BUY / SELL / HOLD
    rsi             REAL     RSI-14 value
    macd            REAL     MACD line value
    macd_signal     REAL     MACD signal line value
    macd_hist       REAL     MACD histogram value
    sma_20          REAL     20-period simple moving average
    sma_50          REAL     50-period simple moving average
    bb_upper        REAL     Upper Bollinger Band (20, 2σ)
    bb_lower        REAL     Lower Bollinger Band (20, 2σ)
    price           REAL     Latest close price
    reasoning       TEXT     Human-readable list of triggered conditions
    created_at      TEXT     ISO-8601 UTC timestamp

combined_signals
    id               INTEGER  Primary key
    ticker           TEXT     Stock ticker symbol
    combined_signal  TEXT     STRONG BUY / STRONG SELL / WEAK BUY / WEAK SELL / CONFLICTING / HOLD
    sentiment_signal TEXT     BUY / SELL / HOLD from SentimentAgent
    technical_signal TEXT     BUY / SELL / HOLD from TechnicalAgent
    sentiment_score  REAL     Avg sentiment score (-1.0 to +1.0)
    confidence       REAL     Confidence score (0.0 to 1.0)
    run_id           INTEGER  Foreign key → runs.id
    technical_id     INTEGER  Foreign key → technical_signals.id
    created_at       TEXT     ISO-8601 UTC timestamp

risk_calculations
    id                INTEGER  Primary key
    ticker            TEXT     Stock ticker symbol
    signal            TEXT     Combined signal that triggered sizing
    confidence        REAL     Confidence score 0–100
    current_price     REAL     Price per share at time of calculation
    account_balance   REAL     Total account value used for sizing
    position_size_usd REAL     Dollar value to deploy (0 if skipped)
    shares            INTEGER  Whole shares to trade (0 if skipped)
    stop_loss         REAL     Stop-loss price (NULL if skipped)
    take_profit       REAL     Take-profit price (NULL if skipped)
    risk_amount       REAL     Max loss if stop-loss is hit
    kelly_fraction    REAL     Half-Kelly fraction used
    stop_pct          REAL     Stop-loss percentage (NULL if skipped)
    skipped           INTEGER  1 = no position taken, 0 = position taken
    skip_reason       TEXT     Explanation when skipped (NULL otherwise)
    created_at        TEXT     ISO-8601 UTC timestamp
"""

from __future__ import annotations

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

                CREATE TABLE IF NOT EXISTS technical_signals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT    NOT NULL,
                    signal      TEXT    NOT NULL,
                    rsi         REAL,
                    macd        REAL,
                    macd_signal REAL,
                    macd_hist   REAL,
                    sma_20      REAL,
                    sma_50      REAL,
                    bb_upper    REAL,
                    bb_lower    REAL,
                    price       REAL,
                    reasoning   TEXT,
                    created_at  TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS combined_signals (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker           TEXT    NOT NULL,
                    combined_signal  TEXT    NOT NULL,
                    sentiment_signal TEXT    NOT NULL,
                    technical_signal TEXT    NOT NULL,
                    sentiment_score  REAL,
                    confidence       REAL,
                    run_id           INTEGER REFERENCES runs(id),
                    technical_id     INTEGER REFERENCES technical_signals(id),
                    created_at       TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS risk_calculations (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker            TEXT    NOT NULL,
                    signal            TEXT    NOT NULL,
                    confidence        REAL    NOT NULL,
                    current_price     REAL    NOT NULL,
                    account_balance   REAL    NOT NULL,
                    position_size_usd REAL    NOT NULL DEFAULT 0,
                    shares            INTEGER NOT NULL DEFAULT 0,
                    stop_loss         REAL,
                    take_profit       REAL,
                    risk_amount       REAL    NOT NULL DEFAULT 0,
                    kelly_fraction    REAL,
                    stop_pct          REAL,
                    skipped           INTEGER NOT NULL DEFAULT 0,
                    skip_reason       TEXT,
                    created_at        TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scheduler_logs (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at            TEXT    NOT NULL,
                    tickers           TEXT    NOT NULL,
                    signals_generated INTEGER NOT NULL DEFAULT 0,
                    trades_executed   INTEGER NOT NULL DEFAULT 0,
                    portfolio_value   REAL    NOT NULL DEFAULT 0,
                    duration_seconds  REAL,
                    errors            TEXT,
                    status            TEXT    NOT NULL,
                    summary           TEXT,
                    created_at        TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtest_results (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker                  TEXT    NOT NULL,
                    start_date              TEXT    NOT NULL,
                    end_date                TEXT    NOT NULL,
                    initial_balance         REAL    NOT NULL,
                    final_balance           REAL    NOT NULL,
                    total_return_pct        REAL,
                    buy_and_hold_return_pct REAL,
                    sharpe_ratio            REAL,
                    max_drawdown_pct        REAL,
                    win_rate_pct            REAL,
                    avg_win                 REAL,
                    avg_loss                REAL,
                    total_trades            INTEGER NOT NULL DEFAULT 0,
                    sentiment_mode          TEXT,
                    trades_json             TEXT,
                    created_at              TEXT    NOT NULL
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

    def log_technical_signal(
        self,
        ticker: str,
        signal: str,
        reasoning: str,
        rsi: "float | None" = None,
        macd: "float | None" = None,
        macd_signal: "float | None" = None,
        macd_hist: "float | None" = None,
        sma_20: "float | None" = None,
        sma_50: "float | None" = None,
        bb_upper: "float | None" = None,
        bb_lower: "float | None" = None,
        price: "float | None" = None,
    ) -> int:
        """
        Persist a technical analysis signal and return its auto-generated ID.

        Args:
            ticker:      Stock ticker symbol.
            signal:      BUY / SELL / HOLD.
            reasoning:   Semicolon-separated list of triggered conditions.
            rsi:         RSI-14 value.
            macd:        MACD line value.
            macd_signal: MACD signal line value.
            macd_hist:   MACD histogram value.
            sma_20:      20-period SMA.
            sma_50:      50-period SMA.
            bb_upper:    Upper Bollinger Band.
            bb_lower:    Lower Bollinger Band.
            price:       Latest close price.

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO technical_signals
                    (ticker, signal, rsi, macd, macd_signal, macd_hist,
                     sma_20, sma_50, bb_upper, bb_lower, price, reasoning, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker, signal, rsi, macd, macd_signal, macd_hist,
                    sma_20, sma_50, bb_upper, bb_lower, price, reasoning, now,
                ),
            )
            return cur.lastrowid

    def log_combined_signal(
        self,
        ticker: str,
        combined_signal: str,
        sentiment_signal: str,
        technical_signal: str,
        sentiment_score: float,
        confidence: float,
        run_id: int,
        technical_id: int,
    ) -> int:
        """
        Persist a combined analysis result and return its auto-generated ID.

        Args:
            ticker:           Stock ticker symbol.
            combined_signal:  STRONG BUY / STRONG SELL / WEAK BUY / WEAK SELL /
                              CONFLICTING / HOLD.
            sentiment_signal: BUY / SELL / HOLD from SentimentAgent.
            technical_signal: BUY / SELL / HOLD from TechnicalAgent.
            sentiment_score:  Averaged sentiment score (−1.0 to +1.0).
            confidence:       Confidence score (0.0 to 1.0).
            run_id:           FK to the sentiment run in runs table.
            technical_id:     FK to the technical result in technical_signals table.

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO combined_signals
                    (ticker, combined_signal, sentiment_signal, technical_signal,
                     sentiment_score, confidence, run_id, technical_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker, combined_signal, sentiment_signal, technical_signal,
                    sentiment_score, confidence, run_id, technical_id, now,
                ),
            )
            return cur.lastrowid

    def log_risk_calculation(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        current_price: float,
        account_balance: float,
        position_size_usd: float = 0.0,
        shares: int = 0,
        stop_loss: "float | None" = None,
        take_profit: "float | None" = None,
        risk_amount: float = 0.0,
        kelly_fraction: float = 0.0,
        stop_pct: "float | None" = None,
        skipped: bool = False,
        skip_reason: "str | None" = None,
    ) -> int:
        """
        Persist a risk calculation result and return its auto-generated ID.

        Args:
            ticker:            Stock ticker symbol.
            signal:            Combined signal string.
            confidence:        Signal confidence 0–100.
            current_price:     Price per share at calculation time.
            account_balance:   Total account value used for sizing.
            position_size_usd: Dollar value to deploy.
            shares:            Whole shares to trade.
            stop_loss:         Stop-loss price (None when skipped).
            take_profit:       Take-profit price (None when skipped).
            risk_amount:       Max dollar loss if stop-loss is hit.
            kelly_fraction:    Half-Kelly fraction used.
            stop_pct:          Stop-loss percentage (None when skipped).
            skipped:           True when no position is recommended.
            skip_reason:       Explanation when skipped.

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO risk_calculations
                    (ticker, signal, confidence, current_price, account_balance,
                     position_size_usd, shares, stop_loss, take_profit,
                     risk_amount, kelly_fraction, stop_pct,
                     skipped, skip_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker, signal, confidence, current_price, account_balance,
                    position_size_usd, shares, stop_loss, take_profit,
                    risk_amount, kelly_fraction, stop_pct,
                    int(skipped), skip_reason, now,
                ),
            )
            return cur.lastrowid

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

    def log_scheduler_run(
        self,
        run_at: str,
        tickers: list[str],
        signals_generated: int,
        trades_executed: int,
        portfolio_value: float,
        duration_seconds: float,
        errors: list[str],
        status: str,
        summary: str,
    ) -> int:
        """
        Persist a scheduler run summary and return its auto-generated ID.

        Args:
            run_at:            ISO-8601 timestamp when the run started.
            tickers:           List of tickers that were analysed.
            signals_generated: Number of combined signals produced.
            trades_executed:   Number of paper trades logged.
            portfolio_value:   Total open portfolio value at end of run.
            duration_seconds:  Wall-clock time for the full run.
            errors:            List of error strings (empty when clean).
            status:            "success" | "partial" | "failed".
            summary:           Human-readable one-paragraph summary.

        Returns:
            The integer primary key of the newly inserted row.
        """
        import json
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO scheduler_logs
                    (run_at, tickers, signals_generated, trades_executed,
                     portfolio_value, duration_seconds, errors, status,
                     summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_at,
                    json.dumps(tickers),
                    signals_generated,
                    trades_executed,
                    portfolio_value,
                    duration_seconds,
                    json.dumps(errors),
                    status,
                    summary,
                    now,
                ),
            )
            return cur.lastrowid

    def log_backtest_result(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        initial_balance: float,
        final_balance: float,
        total_return_pct: float,
        buy_and_hold_return_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
        win_rate_pct: float,
        avg_win: float,
        avg_loss: float,
        total_trades: int,
        sentiment_mode: str,
        trades_json: str,
    ) -> int:
        """
        Persist a backtest run summary and return its auto-generated ID.

        Args:
            ticker:                  Stock ticker symbol.
            start_date:              Backtest start date string "YYYY-MM-DD".
            end_date:                Backtest end date string "YYYY-MM-DD".
            initial_balance:         Starting capital in USD.
            final_balance:           Ending capital in USD.
            total_return_pct:        Strategy total return %.
            buy_and_hold_return_pct: Buy-and-hold return % over same period.
            sharpe_ratio:            Annualised Sharpe ratio.
            max_drawdown_pct:        Maximum drawdown % (negative value).
            win_rate_pct:            Percentage of winning closed trades.
            avg_win:                 Average profit on winning trades.
            avg_loss:                Average loss on losing trades.
            total_trades:            Number of closed trades.
            sentiment_mode:          Sentiment simulation mode used.
            trades_json:             JSON-serialised list of individual trades.

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO backtest_results
                    (ticker, start_date, end_date, initial_balance, final_balance,
                     total_return_pct, buy_and_hold_return_pct, sharpe_ratio,
                     max_drawdown_pct, win_rate_pct, avg_win, avg_loss,
                     total_trades, sentiment_mode, trades_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker, start_date, end_date, initial_balance, final_balance,
                    total_return_pct, buy_and_hold_return_pct, sharpe_ratio,
                    max_drawdown_pct, win_rate_pct, avg_win, avg_loss,
                    total_trades, sentiment_mode, trades_json, now,
                ),
            )
            return cur.lastrowid
