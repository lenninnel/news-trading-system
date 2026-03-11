"""
SQLite persistence layer for the News Trading System.

Provides a Database class that initialises the schema on first use and
exposes methods for logging analysis runs and individual headline scores.
All writes use context managers so connections are always cleanly closed.

Schema
------
runs
    id               INTEGER  Primary key
    ticker           TEXT     Stock ticker symbol
    headlines_fetched INTEGER Total items retrieved from all sources
    headlines_analysed INTEGER Items successfully scored
    avg_score        REAL     Weighted average sentiment score (-1.0 to +1.0)
    signal           TEXT     BUY / SELL / HOLD
    source_breakdown TEXT     JSON: per-source {count, avg}
    created_at       TEXT     ISO-8601 UTC timestamp

headline_scores
    id              INTEGER  Primary key
    run_id          INTEGER  Foreign key → runs.id
    headline        TEXT     Original headline text
    sentiment       TEXT     bullish / bearish / neutral
    score           INTEGER  Numeric score (+1 / 0 / -1)
    reason          TEXT     One-sentence explanation from Claude
    source          TEXT     newsapi / stocktwits / reddit

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
    rvol            REAL     Relative volume (current / 20-day avg)
    obv_trend       TEXT     OBV direction: "rising" or "falling"
    volume_confirmed INTEGER 1 = volume confirms signal, 0 = not confirmed
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
    event_risk_flag   TEXT     none / earnings_week / earnings_imminent
    days_to_earnings  INTEGER  Trading days until earnings (NULL if unknown)
    created_at        TEXT     ISO-8601 UTC timestamp

portfolio_positions
    id              INTEGER  Primary key
    ticker          TEXT     Stock ticker symbol (UNIQUE)
    shares          INTEGER  Number of shares held
    avg_price       REAL     Weighted-average cost basis per share
    current_value   REAL     shares × latest price
    updated_at      TEXT     ISO-8601 UTC timestamp

trade_history
    id              INTEGER  Primary key
    ticker          TEXT     Stock ticker symbol
    action          TEXT     BUY or SELL
    shares          INTEGER  Number of shares traded
    price           REAL     Fill price per share
    stop_loss       REAL     Stop-loss price (NULL if not set)
    take_profit     REAL     Take-profit price (NULL if not set)
    pnl             REAL     Realised PnL (0.0 for buys)
    created_at      TEXT     ISO-8601 UTC timestamp

optimization_runs
    id                    INTEGER  Primary key
    ticker                TEXT     Stock ticker symbol
    start_date            TEXT     ISO date string
    end_date              TEXT     ISO date string
    best_params           TEXT     JSON-encoded parameter dict
    in_sample_sharpe      REAL     Average in-sample Sharpe
    out_of_sample_sharpe  REAL     Average out-of-sample Sharpe
    total_windows         INTEGER  Number of walk-forward windows
    equity_curve          TEXT     JSON-encoded OOS equity curve
    created_at            TEXT     ISO-8601 UTC timestamp
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

from config.settings import DB_PATH


def _resolve_db_path(default: str = DB_PATH) -> str:
    """Use a Railway persistent volume if /data exists, else fall back to local."""
    railway_dir = "/data"
    if os.path.isdir(railway_dir):
        return os.path.join(railway_dir, "news_trading.db")
    return default


class Database:
    """Thin wrapper around a SQLite database for logging trading analysis."""

    _write_lock = threading.Lock()

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialise the database and create tables if they don't exist.

        Args:
            db_path: Path to the SQLite file. Defaults to settings.DB_PATH.
        """
        self.db_path = db_path or _resolve_db_path()
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
                    source_breakdown    TEXT,
                    created_at          TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS headline_scores (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      INTEGER NOT NULL REFERENCES runs(id),
                    headline    TEXT    NOT NULL,
                    sentiment   TEXT    NOT NULL,
                    score       INTEGER NOT NULL,
                    reason      TEXT,
                    source      TEXT    NOT NULL DEFAULT 'newsapi'
                );

                CREATE TABLE IF NOT EXISTS technical_signals (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker           TEXT    NOT NULL,
                    signal           TEXT    NOT NULL,
                    rsi              REAL,
                    macd             REAL,
                    macd_signal      REAL,
                    macd_hist        REAL,
                    sma_20           REAL,
                    sma_50           REAL,
                    bb_upper         REAL,
                    bb_lower         REAL,
                    price            REAL,
                    reasoning        TEXT,
                    rvol             REAL,
                    obv_trend        TEXT,
                    volume_confirmed INTEGER NOT NULL DEFAULT 0,
                    created_at       TEXT    NOT NULL
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
                    event_risk_flag   TEXT    NOT NULL DEFAULT 'none',
                    days_to_earnings  INTEGER,
                    created_at        TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker        TEXT    NOT NULL UNIQUE,
                    shares        INTEGER NOT NULL,
                    avg_price     REAL    NOT NULL,
                    current_value REAL    NOT NULL DEFAULT 0,
                    updated_at    TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT    NOT NULL,
                    action      TEXT    NOT NULL,
                    shares      INTEGER NOT NULL,
                    price       REAL    NOT NULL,
                    stop_loss   REAL,
                    take_profit REAL,
                    pnl         REAL    NOT NULL DEFAULT 0,
                    created_at  TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS optimization_runs (
                    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker                TEXT    NOT NULL,
                    start_date            TEXT    NOT NULL,
                    end_date              TEXT    NOT NULL,
                    best_params           TEXT    NOT NULL,
                    in_sample_sharpe      REAL    NOT NULL DEFAULT 0,
                    out_of_sample_sharpe  REAL    NOT NULL DEFAULT 0,
                    total_windows         INTEGER NOT NULL DEFAULT 0,
                    equity_curve          TEXT,
                    created_at            TEXT    NOT NULL
                );
                """
            )
            # Migrate existing DBs: add volume columns to technical_signals
            for col, typedef in [
                ("rvol", "REAL"),
                ("obv_trend", "TEXT"),
                ("volume_confirmed", "INTEGER NOT NULL DEFAULT 0"),
            ]:
                try:
                    conn.execute(
                        f"ALTER TABLE technical_signals ADD COLUMN {col} {typedef}"
                    )
                except sqlite3.OperationalError:
                    pass  # column already exists

            # Migrate existing DBs: add multi-source columns
            for table, col, typedef in [
                ("runs", "source_breakdown", "TEXT"),
                ("headline_scores", "source", "TEXT NOT NULL DEFAULT 'newsapi'"),
            ]:
                try:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} {typedef}"
                    )
                except sqlite3.OperationalError:
                    pass  # column already exists

            # Migrate existing DBs: add event risk + regime columns to risk_calculations
            for col, typedef in [
                ("event_risk_flag", "TEXT NOT NULL DEFAULT 'none'"),
                ("days_to_earnings", "INTEGER"),
                ("regime", "TEXT"),
            ]:
                try:
                    conn.execute(
                        f"ALTER TABLE risk_calculations ADD COLUMN {col} {typedef}"
                    )
                except sqlite3.OperationalError:
                    pass  # column already exists

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
        source_breakdown: "dict | None" = None,
    ) -> int:
        """
        Persist a completed analysis run and return its auto-generated ID.

        Args:
            ticker:               Stock ticker symbol.
            headlines_fetched:    Number of items retrieved from all sources.
            headlines_analysed:   Number of items successfully scored.
            avg_score:            Weighted sentiment score.
            signal:               Trading signal (BUY / SELL / HOLD).
            source_breakdown:     Per-source {count, avg} dict (stored as JSON).

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        breakdown_json = json.dumps(source_breakdown) if source_breakdown else None
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs
                    (ticker, headlines_fetched, headlines_analysed, avg_score,
                     signal, source_breakdown, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, headlines_fetched, headlines_analysed, avg_score,
                 signal, breakdown_json, now),
            )
            return cur.lastrowid

    def log_headline_score(
        self,
        run_id: int,
        headline: str,
        sentiment: str,
        score: int,
        reason: str,
        source: str = "newsapi",
    ) -> None:
        """
        Persist a single headline score linked to a run.

        Args:
            run_id:    ID of the parent run (from log_run).
            headline:  Original headline text.
            sentiment: bullish / bearish / neutral.
            score:     Numeric score (+1 / 0 / -1).
            reason:    Claude's one-sentence explanation.
            source:    Data source: newsapi / stocktwits / reddit.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO headline_scores
                    (run_id, headline, sentiment, score, reason, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, headline, sentiment, score, reason, source),
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
        rvol: "float | None" = None,
        obv_trend: "str | None" = None,
        volume_confirmed: bool = False,
    ) -> int:
        """
        Persist a technical analysis signal and return its auto-generated ID.

        Args:
            ticker:           Stock ticker symbol.
            signal:           BUY / SELL / HOLD.
            reasoning:        Semicolon-separated list of triggered conditions.
            rsi:              RSI-14 value.
            macd:             MACD line value.
            macd_signal:      MACD signal line value.
            macd_hist:        MACD histogram value.
            sma_20:           20-period SMA.
            sma_50:           50-period SMA.
            bb_upper:         Upper Bollinger Band.
            bb_lower:         Lower Bollinger Band.
            price:            Latest close price.
            rvol:             Relative volume (current / 20-day avg).
            obv_trend:        OBV direction: "rising" or "falling".
            volume_confirmed: True when volume confirms the signal direction.

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO technical_signals
                    (ticker, signal, rsi, macd, macd_signal, macd_hist,
                     sma_20, sma_50, bb_upper, bb_lower, price, reasoning,
                     rvol, obv_trend, volume_confirmed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker, signal, rsi, macd, macd_signal, macd_hist,
                    sma_20, sma_50, bb_upper, bb_lower, price, reasoning,
                    rvol, obv_trend, int(volume_confirmed), now,
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
        event_risk_flag: str = "none",
        days_to_earnings: "int | None" = None,
        regime: "str | None" = None,
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
            event_risk_flag:   none / earnings_week / earnings_imminent.
            days_to_earnings:  Trading days until earnings (None if unknown).
            regime:            Market regime (None if not detected).

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
                     skipped, skip_reason,
                     event_risk_flag, days_to_earnings, regime, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker, signal, confidence, current_price, account_balance,
                    position_size_usd, shares, stop_loss, take_profit,
                    risk_amount, kelly_fraction, stop_pct,
                    int(skipped), skip_reason,
                    event_risk_flag, days_to_earnings, regime, now,
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

    # ------------------------------------------------------------------
    # Portfolio & trade history (used by PaperTrader)
    # ------------------------------------------------------------------

    def get_portfolio_position(self, ticker: str) -> "dict | None":
        """
        Return the current position for *ticker*, or None if no position exists.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict with keys: id, ticker, shares, avg_price, current_value,
            updated_at — or None.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio_positions WHERE ticker = ?", (ticker,)
            ).fetchone()
            return dict(row) if row else None

    def set_portfolio_position(
        self,
        ticker: str,
        shares: int,
        avg_price: float,
        current_value: float,
    ) -> None:
        """
        Insert or update a portfolio position (upsert).

        Args:
            ticker:        Stock ticker symbol.
            shares:        Total shares held after the trade.
            avg_price:     Weighted-average cost basis per share.
            current_value: shares x latest price.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_positions
                    (ticker, shares, avg_price, current_value, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    shares        = excluded.shares,
                    avg_price     = excluded.avg_price,
                    current_value = excluded.current_value,
                    updated_at    = excluded.updated_at
                """,
                (ticker, shares, avg_price, current_value, now),
            )

    def delete_portfolio_position(self, ticker: str) -> None:
        """
        Remove a position from the portfolio (called when shares reach zero).

        Args:
            ticker: Stock ticker symbol.
        """
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM portfolio_positions WHERE ticker = ?", (ticker,)
            )

    def log_trade_history(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        stop_loss: "float | None" = None,
        take_profit: "float | None" = None,
        pnl: float = 0.0,
    ) -> int:
        """
        Append an immutable trade record and return its auto-generated ID.

        Args:
            ticker:      Stock ticker symbol.
            action:      "BUY" or "SELL".
            shares:      Number of shares traded.
            price:       Fill price per share.
            stop_loss:   Stop-loss price (None if not set).
            take_profit: Take-profit price (None if not set).
            pnl:         Realised PnL (0.0 for buys).

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trade_history
                    (ticker, action, shares, price, stop_loss, take_profit, pnl, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, action, shares, price, stop_loss, take_profit, pnl, now),
            )
            return cur.lastrowid

    def get_portfolio(self) -> list[dict]:
        """
        Return all open positions, ordered by ticker.

        Returns:
            List of dicts, each with: id, ticker, shares, avg_price,
            current_value, updated_at.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM portfolio_positions ORDER BY ticker"
            ).fetchall()
            return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Optimisation runs
    # ------------------------------------------------------------------

    def log_optimization_run(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        best_params: dict,
        in_sample_sharpe: float = 0.0,
        out_of_sample_sharpe: float = 0.0,
        total_windows: int = 0,
        equity_curve: "list[float] | None" = None,
    ) -> int:
        """
        Persist a completed optimisation run and return its ID.

        Args:
            ticker:               Stock ticker symbol.
            start_date:           ISO date string for the overall start.
            end_date:             ISO date string for the overall end.
            best_params:          Best parameter combo (stored as JSON).
            in_sample_sharpe:     Average in-sample Sharpe ratio.
            out_of_sample_sharpe: Average out-of-sample Sharpe ratio.
            total_windows:        Number of walk-forward windows.
            equity_curve:         Concatenated OOS equity curve (stored as JSON).

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        params_json = json.dumps(best_params)
        curve_json = json.dumps(equity_curve) if equity_curve else None
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO optimization_runs
                    (ticker, start_date, end_date, best_params,
                     in_sample_sharpe, out_of_sample_sharpe,
                     total_windows, equity_curve, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker, start_date, end_date, params_json,
                    in_sample_sharpe, out_of_sample_sharpe,
                    total_windows, curve_json, now,
                ),
            )
            return cur.lastrowid

    def get_optimization_runs(
        self,
        ticker: "str | None" = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Return past optimisation runs, newest first.

        Args:
            ticker: Filter by ticker symbol (None = all tickers).
            limit:  Maximum number of rows to return.

        Returns:
            List of dicts, each representing one optimization_runs row.
        """
        with self._connect() as conn:
            if ticker:
                rows = conn.execute(
                    "SELECT * FROM optimization_runs WHERE ticker = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (ticker, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM optimization_runs ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]

    def get_trade_history(
        self,
        ticker: "str | None" = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Return past trades, newest first.

        Args:
            ticker: Filter by ticker symbol (None = all tickers).
            limit:  Maximum number of rows to return.

        Returns:
            List of dicts, each representing one trade_history row.
        """
        with self._connect() as conn:
            if ticker:
                rows = conn.execute(
                    "SELECT * FROM trade_history WHERE ticker = ? ORDER BY id DESC LIMIT ?",
                    (ticker, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trade_history ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
