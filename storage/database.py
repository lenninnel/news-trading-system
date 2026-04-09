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
    sma_200         REAL     200-period simple moving average
    ma200_distance_pct REAL  Distance from SMA-200 as a percentage
    golden_cross_recent INTEGER 1 = golden cross detected recently
    death_cross_recent  INTEGER 1 = death cross detected recently
    adx             REAL     Average Directional Index value
    trend_strength  TEXT     Trend strength label (e.g. "strong", "weak")
    bull_flag_detected INTEGER 1 = bull flag pattern detected
    wedge_type      TEXT     Wedge pattern type (e.g. "rising", "falling")
    wedge_breakout  INTEGER 1 = wedge breakout detected
    nearest_support REAL     Nearest support level price
    nearest_resistance REAL  Nearest resistance level price
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

import functools
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

from filelock import FileLock

from config.settings import DB_PATH

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BASE_BACKOFF = 1.0  # seconds


def _retry_on_locked(func):
    """Decorator that retries on 'database is locked' OperationalError.

    Retries up to _MAX_RETRIES times with exponential back-off
    (1s, 2s, 4s, 8s, 16s).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exc: sqlite3.OperationalError | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                if "database is locked" not in str(exc):
                    raise
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    wait = _BASE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "database is locked (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
        raise last_exc  # type: ignore[misc]
    return wrapper


def _resolve_db_path(default: str = DB_PATH) -> str:
    """Use a Railway persistent volume if /data exists and is writable, else fall back."""
    railway_dir = "/data"
    if os.path.isdir(railway_dir) and os.access(railway_dir, os.W_OK):
        return os.path.join(railway_dir, "news_trading.db")
    return default


class Database:
    """Thin wrapper around a SQLite database for logging trading analysis.

    SQLite is used intentionally — it is the right fit for a single-process
    trading system that runs one daemon per host.  On Railway the DB file
    lives on a persistent volume (/data).  PostgreSQL is NOT required;
    DATABASE_URL is only used by the legacy migration script and the
    docker-compose Postgres service (neither is used in production).
    """

    _write_lock = threading.Lock()

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialise the database and create tables if they don't exist.

        Args:
            db_path: Path to the SQLite file. Defaults to settings.DB_PATH.
        """
        self.db_path = db_path or _resolve_db_path()
        self._file_lock = FileLock(self.db_path + ".lock", timeout=30)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @_retry_on_locked
    def _init_schema(self) -> None:
        with self._file_lock, self._write_lock, self._connect() as conn:
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
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS recovery_log (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    service          TEXT    NOT NULL,
                    event_type       TEXT    NOT NULL,
                    ticker           TEXT,
                    attempt          INTEGER,
                    error_msg        TEXT,
                    recovery_action  TEXT,
                    duration_ms      INTEGER,
                    success          INTEGER NOT NULL DEFAULT 0,
                    created_at       TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS screener_results (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at      TEXT    NOT NULL,
                    ticker      TEXT    NOT NULL,
                    market      TEXT,
                    hotness     REAL,
                    price_change REAL,
                    volume_ratio REAL,
                    rsi         REAL,
                    price       REAL,
                    created_at  TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS strategy_signals (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker           TEXT    NOT NULL,
                    strategy         TEXT    NOT NULL,
                    signal           TEXT    NOT NULL,
                    confidence       REAL,
                    timeframe        TEXT,
                    risk_calc_id     INTEGER,
                    created_at       TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker                TEXT    NOT NULL,
                    run_at                TEXT    NOT NULL,
                    combined_signal       TEXT,
                    ensemble_confidence   REAL,
                    voting_method         TEXT,
                    momentum_signal       TEXT,
                    momentum_confidence   REAL,
                    mean_rev_signal       TEXT,
                    mean_rev_confidence   REAL,
                    swing_signal          TEXT,
                    swing_confidence      REAL,
                    created_at            TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scheduler_logs (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at           TEXT    NOT NULL,
                    tickers          TEXT,
                    success_count    INTEGER NOT NULL DEFAULT 0,
                    fail_count       INTEGER NOT NULL DEFAULT 0,
                    elapsed_s        REAL,
                    total_elapsed_s  REAL,
                    errors           TEXT,
                    status           TEXT,
                    notes            TEXT,
                    created_at       TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS health_checks (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at           TEXT    NOT NULL,
                    newsapi_ok       INTEGER NOT NULL DEFAULT 0,
                    yfinance_ok      INTEGER NOT NULL DEFAULT 0,
                    anthropic_ok     INTEGER NOT NULL DEFAULT 0,
                    database_ok      INTEGER NOT NULL DEFAULT 0,
                    network_ok       INTEGER NOT NULL DEFAULT 0,
                    scheduler_ok     INTEGER NOT NULL DEFAULT 0,
                    created_at       TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS emergency_stops (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    action           TEXT    NOT NULL,
                    reason           TEXT,
                    activated_by     TEXT,
                    created_at       TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_at      TEXT    NOT NULL,
                    total_value      REAL,
                    positions        TEXT,
                    created_at       TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_violations (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    violation_type   TEXT    NOT NULL,
                    ticker           TEXT,
                    details          TEXT,
                    created_at       TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS price_alerts (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker           TEXT    NOT NULL,
                    alert_type       TEXT    NOT NULL,
                    price            REAL,
                    message          TEXT,
                    created_at       TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS optimization_results (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker           TEXT    NOT NULL,
                    strategy         TEXT,
                    params           TEXT,
                    score            REAL,
                    created_at       TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtest_results (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker           TEXT    NOT NULL,
                    strategy         TEXT,
                    start_date       TEXT,
                    end_date         TEXT,
                    total_return     REAL,
                    sharpe_ratio     REAL,
                    max_drawdown     REAL,
                    params           TEXT,
                    created_at       TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtest_strategy_comparison (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at           TEXT    NOT NULL,
                    ticker           TEXT    NOT NULL,
                    comparison       TEXT,
                    created_at       TEXT    NOT NULL
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

            # Migrate existing DBs: add advanced TA columns to technical_signals
            for col, typedef in [
                ("sma_200", "REAL"),
                ("ma200_distance_pct", "REAL"),
                ("golden_cross_recent", "INTEGER DEFAULT 0"),
                ("death_cross_recent", "INTEGER DEFAULT 0"),
                ("adx", "REAL"),
                ("trend_strength", "TEXT"),
                ("bull_flag_detected", "INTEGER DEFAULT 0"),
                ("wedge_type", "TEXT"),
                ("wedge_breakout", "INTEGER DEFAULT 0"),
                ("nearest_support", "REAL"),
                ("nearest_resistance", "REAL"),
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

    @_retry_on_locked
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
        with self._file_lock, self._write_lock, self._connect() as conn:
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

    @_retry_on_locked
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
        with self._file_lock, self._write_lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO headline_scores
                    (run_id, headline, sentiment, score, reason, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, headline, sentiment, score, reason, source),
            )

    @_retry_on_locked
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
        sma_200: "float | None" = None,
        ma200_distance_pct: "float | None" = None,
        golden_cross_recent: bool = False,
        death_cross_recent: bool = False,
        adx: "float | None" = None,
        trend_strength: "str | None" = None,
        bull_flag_detected: bool = False,
        wedge_type: "str | None" = None,
        wedge_breakout: bool = False,
        nearest_support: "float | None" = None,
        nearest_resistance: "float | None" = None,
    ) -> int:
        """
        Persist a technical analysis signal and return its auto-generated ID.

        Args:
            ticker:              Stock ticker symbol.
            signal:              BUY / SELL / HOLD.
            reasoning:           Semicolon-separated list of triggered conditions.
            rsi:                 RSI-14 value.
            macd:                MACD line value.
            macd_signal:         MACD signal line value.
            macd_hist:           MACD histogram value.
            sma_20:              20-period SMA.
            sma_50:              50-period SMA.
            bb_upper:            Upper Bollinger Band.
            bb_lower:            Lower Bollinger Band.
            price:               Latest close price.
            rvol:                Relative volume (current / 20-day avg).
            obv_trend:           OBV direction: "rising" or "falling".
            volume_confirmed:    True when volume confirms the signal direction.
            sma_200:             200-period SMA.
            ma200_distance_pct:  Distance from SMA-200 as a percentage.
            golden_cross_recent: True if golden cross detected recently.
            death_cross_recent:  True if death cross detected recently.
            adx:                 Average Directional Index value.
            trend_strength:      Trend strength label (e.g. "strong", "weak").
            bull_flag_detected:  True if bull flag pattern detected.
            wedge_type:          Wedge pattern type (e.g. "rising", "falling").
            wedge_breakout:      True if wedge breakout detected.
            nearest_support:     Nearest support level price.
            nearest_resistance:  Nearest resistance level price.

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._file_lock, self._write_lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO technical_signals
                    (ticker, signal, rsi, macd, macd_signal, macd_hist,
                     sma_20, sma_50, bb_upper, bb_lower, price, reasoning,
                     rvol, obv_trend, volume_confirmed,
                     sma_200, ma200_distance_pct, golden_cross_recent,
                     death_cross_recent, adx, trend_strength,
                     bull_flag_detected, wedge_type, wedge_breakout,
                     nearest_support, nearest_resistance,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker, signal, rsi, macd, macd_signal, macd_hist,
                    sma_20, sma_50, bb_upper, bb_lower, price, reasoning,
                    rvol, obv_trend, int(volume_confirmed),
                    sma_200, ma200_distance_pct,
                    int(golden_cross_recent), int(death_cross_recent),
                    adx, trend_strength,
                    int(bull_flag_detected), wedge_type, int(wedge_breakout),
                    nearest_support, nearest_resistance,
                    now,
                ),
            )
            return cur.lastrowid

    @_retry_on_locked
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
        with self._file_lock, self._write_lock, self._connect() as conn:
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

    @_retry_on_locked
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
        with self._file_lock, self._write_lock, self._connect() as conn:
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

    @_retry_on_locked
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
        with self._file_lock, self._write_lock, self._connect() as conn:
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

    @_retry_on_locked
    def delete_portfolio_position(self, ticker: str) -> None:
        """
        Remove a position from the portfolio (called when shares reach zero).

        Args:
            ticker: Stock ticker symbol.
        """
        with self._file_lock, self._write_lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM portfolio_positions WHERE ticker = ?", (ticker,)
            )

    @_retry_on_locked
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
        with self._file_lock, self._write_lock, self._connect() as conn:
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

    @_retry_on_locked
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
        with self._file_lock, self._write_lock, self._connect() as conn:
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

    # ------------------------------------------------------------------
    # Recovery log
    # ------------------------------------------------------------------

    @_retry_on_locked
    def log_recovery_event(
        self,
        service: str,
        event_type: str,
        *,
        ticker: "str | None" = None,
        attempt: "int | None" = None,
        error_msg: "str | None" = None,
        recovery_action: "str | None" = None,
        duration_ms: "int | None" = None,
        success: bool = False,
    ) -> int:
        """Persist a recovery event and return its row ID."""
        now = datetime.now(timezone.utc).isoformat()
        with self._file_lock, self._write_lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO recovery_log
                    (service, event_type, ticker, attempt, error_msg,
                     recovery_action, duration_ms, success, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (service, event_type, ticker, attempt, error_msg,
                 recovery_action, duration_ms, int(success), now),
            )
            return cur.lastrowid

    # ------------------------------------------------------------------
    # Screener results
    # ------------------------------------------------------------------

    @_retry_on_locked
    def log_screener_results(self, run_at: str, candidates: list[dict]) -> None:
        """Persist screener results."""
        now = datetime.now(timezone.utc).isoformat()
        with self._file_lock, self._write_lock, self._connect() as conn:
            for c in candidates:
                conn.execute(
                    """
                    INSERT INTO screener_results
                        (run_at, ticker, market, hotness, price_change,
                         volume_ratio, rsi, price, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (run_at, c.get("ticker"), c.get("market"),
                     c.get("hotness"), c.get("price_change"),
                     c.get("volume_ratio"), c.get("rsi"),
                     c.get("price"), now),
                )

    # ------------------------------------------------------------------
    # Strategy signals + performance
    # ------------------------------------------------------------------

    @_retry_on_locked
    def log_strategy_signal(
        self,
        ticker: str,
        strategy: str,
        signal: str,
        confidence: float,
        *,
        timeframe: "str | None" = None,
        risk_calc_id: "int | None" = None,
    ) -> int:
        """Persist a strategy signal and return its row ID."""
        now = datetime.now(timezone.utc).isoformat()
        with self._file_lock, self._write_lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO strategy_signals
                    (ticker, strategy, signal, confidence, timeframe,
                     risk_calc_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, strategy, signal, confidence, timeframe,
                 risk_calc_id, now),
            )
            return cur.lastrowid

    @_retry_on_locked
    def log_strategy_performance(
        self,
        ticker: str,
        run_at: str,
        combined_signal: str,
        ensemble_confidence: float,
        voting_method: str,
        *,
        momentum_signal: "str | None" = None,
        momentum_confidence: "float | None" = None,
        mean_rev_signal: "str | None" = None,
        mean_rev_confidence: "float | None" = None,
        swing_signal: "str | None" = None,
        swing_confidence: "float | None" = None,
    ) -> int:
        """Persist a strategy performance record."""
        now = datetime.now(timezone.utc).isoformat()
        with self._file_lock, self._write_lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO strategy_performance
                    (ticker, run_at, combined_signal, ensemble_confidence,
                     voting_method, momentum_signal, momentum_confidence,
                     mean_rev_signal, mean_rev_confidence,
                     swing_signal, swing_confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, run_at, combined_signal, ensemble_confidence,
                 voting_method, momentum_signal, momentum_confidence,
                 mean_rev_signal, mean_rev_confidence,
                 swing_signal, swing_confidence, now),
            )
            return cur.lastrowid

    # ------------------------------------------------------------------
    # Health checks, emergency stops, scheduler logs
    # ------------------------------------------------------------------

    @_retry_on_locked
    def log_health_check(
        self, run_at: str,
        newsapi_ok: bool, yfinance_ok: bool, anthropic_ok: bool,
        database_ok: bool, network_ok: bool, scheduler_ok: bool,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._file_lock, self._write_lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO health_checks
                    (run_at, newsapi_ok, yfinance_ok, anthropic_ok,
                     database_ok, network_ok, scheduler_ok, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_at, int(newsapi_ok), int(yfinance_ok), int(anthropic_ok),
                 int(database_ok), int(network_ok), int(scheduler_ok), now),
            )
            return cur.lastrowid

    @_retry_on_locked
    def log_emergency_stop(
        self, action: str, reason: "str | None" = None,
        activated_by: "str | None" = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._file_lock, self._write_lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO emergency_stops (action, reason, activated_by, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (action, reason, activated_by, now),
            )
            return cur.lastrowid

    @_retry_on_locked
    def log_scheduler_run(
        self, run_at: str, tickers: list, success_count: int,
        fail_count: int, elapsed_s: float, total_elapsed_s: float,
        errors: list, status: str, notes: str = "",
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._file_lock, self._write_lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO scheduler_logs
                    (run_at, tickers, success_count, fail_count, elapsed_s,
                     total_elapsed_s, errors, status, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_at, json.dumps(tickers), success_count, fail_count,
                 elapsed_s, total_elapsed_s, json.dumps(errors), status,
                 notes, now),
            )
            return cur.lastrowid

    # ------------------------------------------------------------------
    # Generic SQL helper
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Signal cache (US_PRE → US_OPEN fast path)
    # ------------------------------------------------------------------

    def get_cached_signal(
        self,
        ticker: str,
        *,
        max_age_minutes: int = 90,
        valid_sessions: tuple[str, ...] = ("US_PRE", "XETRA_PRE"),
    ) -> "dict | None":
        """Return the latest signal_events row for *ticker* if fresh enough.

        A cached signal is only considered valid when **all** of the
        following hold:

        * its ``session`` is in *valid_sessions* (defaults to the pre-signal
          sessions — never EOD/MIDDAY/US_OPEN);
        * its ``timestamp`` falls on today's UTC date (no overnight reuse);
        * its ``timestamp`` is within *max_age_minutes* of now.

        Returns ``None`` when no row qualifies. When a row exists but is
        rejected, the reason is logged at INFO level so callers can see
        why the cache missed.
        """
        from datetime import datetime as _dt, timedelta, timezone as _tz

        ticker = ticker.upper()
        now_utc = _dt.now(_tz.utc)
        today_str = now_utc.date().isoformat()  # e.g. "2026-04-08"
        cutoff = (now_utc - timedelta(minutes=max_age_minutes)).isoformat()

        try:
            with self._connect() as conn:
                # Diagnostic: latest row for this ticker, any session/date.
                # Used purely for the rejection-reason log below.
                latest = conn.execute(
                    """
                    SELECT id, timestamp, session, signal
                    FROM signal_events
                    WHERE ticker = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (ticker,),
                ).fetchone()

                placeholders = ",".join("?" * len(valid_sessions))
                row = conn.execute(
                    f"""
                    SELECT * FROM signal_events
                    WHERE ticker = ?
                      AND session IN ({placeholders})
                      AND substr(timestamp, 1, 10) = ?
                      AND timestamp >= ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (ticker, *valid_sessions, today_str, cutoff),
                ).fetchone()

            if row:
                return dict(row)

            # No valid cache — log why if there's anything at all to diagnose.
            if latest:
                lat = dict(latest)
                ts = lat.get("timestamp") or ""
                sess = lat.get("session") or "<none>"
                row_date = ts[:10] if len(ts) >= 10 else "unknown"
                if row_date != today_str:
                    logger.info(
                        "[%s] Cached signal rejected: from previous day (%s)",
                        ticker, row_date,
                    )
                elif sess not in valid_sessions:
                    logger.info(
                        "[%s] Cached signal rejected: session=%s not in %s",
                        ticker, sess, list(valid_sessions),
                    )
                elif ts < cutoff:
                    logger.info(
                        "[%s] Cached signal rejected: older than %d min (ts=%s)",
                        ticker, max_age_minutes, ts,
                    )
            return None
        except Exception as exc:
            logger.warning("get_cached_signal(%s) failed: %s", ticker, exc)
            return None

    @staticmethod
    def is_price_stale(
        current_price: float,
        cached_price: float,
        threshold: float = 0.02,
    ) -> bool:
        """Return True if *current_price* moved more than *threshold* from *cached_price*.

        Both prices must be positive; returns True (stale) on bad input so
        the caller falls through to a full re-computation.
        """
        if not cached_price or cached_price <= 0 or not current_price or current_price <= 0:
            return True
        return abs(current_price - cached_price) / cached_price > threshold

    def _select(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run a raw SELECT and return rows as dicts."""
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Sector / peer lookups (sourced from config/sector_map.json)
# ─────────────────────────────────────────────────────────────────────────────

_SECTOR_MAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "sector_map.json",
)
_SECTOR_MAP_CACHE: dict | None = None
_SECTOR_MAP_MTIME: float | None = None


def _load_sector_map() -> dict:
    """Load (and memoise) the sector map. Reloads when the file changes on disk."""
    global _SECTOR_MAP_CACHE, _SECTOR_MAP_MTIME
    try:
        mtime = os.path.getmtime(_SECTOR_MAP_PATH)
    except OSError:
        _SECTOR_MAP_CACHE = {}
        _SECTOR_MAP_MTIME = None
        return _SECTOR_MAP_CACHE
    if _SECTOR_MAP_CACHE is None or mtime != _SECTOR_MAP_MTIME:
        try:
            with open(_SECTOR_MAP_PATH) as fh:
                _SECTOR_MAP_CACHE = json.load(fh) or {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read sector_map.json: %s", exc)
            _SECTOR_MAP_CACHE = {}
        _SECTOR_MAP_MTIME = mtime
    return _SECTOR_MAP_CACHE


def get_peers(ticker: str) -> list[str]:
    """Return the list of correlated peers for *ticker* (empty if none)."""
    entry = _load_sector_map().get(ticker.upper(), {})
    peers = entry.get("peers") or []
    return list(peers)


def get_sector(ticker: str) -> str | None:
    """Return the sector slug for *ticker*, or ``None`` if unknown."""
    entry = _load_sector_map().get(ticker.upper(), {})
    return entry.get("sector")
