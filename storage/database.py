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

strategy_signals
    id                  INTEGER  Primary key
    ticker              TEXT     Stock ticker symbol
    strategy            TEXT     "momentum" | "mean_reversion" | "swing"
    signal              TEXT     "BUY" | "SELL" | "HOLD"
    confidence          REAL     Agent confidence 0–100
    timeframe           TEXT     Intended holding period
    reasoning           TEXT     Semicolon-joined triggered conditions
    indicators_json     TEXT     JSON blob of numeric indicator values
    ensemble_confidence REAL     Coordinator ensemble confidence 0–100
    combined_signal     TEXT     Coordinator combined direction
    consensus           TEXT     "unanimous" | "majority" | "conflicting" | "none"
    account_balance     REAL     Account value used for sizing
    risk_calc_id        INTEGER  FK → risk_calculations.id
    created_at          TEXT     ISO-8601 UTC timestamp

backtest_strategy_comparison
    id            INTEGER  Primary key
    ticker        TEXT     Stock ticker symbol
    start_date    TEXT     Backtest start date "YYYY-MM-DD"
    end_date      TEXT     Backtest end date "YYYY-MM-DD"
    strategy      TEXT     "momentum" | "mean_reversion" | "swing" | "buy_and_hold"
    total_return  REAL     Total return percent
    sharpe        REAL     Annualised Sharpe ratio
    max_dd        REAL     Maximum drawdown percent (negative)
    win_rate      REAL     Win rate percent 0–100
    trade_count   INTEGER  Number of completed trades
    avg_hold_days REAL     Average holding period in calendar days
    created_at    TEXT     ISO-8601 UTC timestamp

strategy_performance
    id                        INTEGER  Primary key
    ticker                    TEXT     Stock ticker symbol
    run_at                    TEXT     ISO-8601 coordinator run timestamp
    momentum_signal           TEXT     MomentumAgent signal (NULL if failed)
    momentum_confidence       REAL     MomentumAgent confidence
    mean_reversion_signal     TEXT     MeanReversionAgent signal
    mean_reversion_confidence REAL     MeanReversionAgent confidence
    swing_signal              TEXT     SwingAgent signal
    swing_confidence          REAL     SwingAgent confidence
    combined_signal           TEXT     Ensemble direction
    ensemble_confidence       REAL     Ensemble confidence 0–100
    consensus                 TEXT     "unanimous" | "majority" | "conflicting" | "none"
    risk_calc_id              INTEGER  FK → risk_calculations.id
    account_balance           REAL     Account value used for sizing
    errors_json               TEXT     JSON list of per-agent error strings
    created_at                TEXT     ISO-8601 UTC timestamp

portfolio_snapshots
    id                INTEGER  Primary key
    snapshot_at       TEXT     ISO-8601 UTC timestamp of the snapshot
    open_positions    INTEGER  Number of open positions at snapshot time
    total_value       REAL     Total portfolio value (entry-based)
    deployed_pct      REAL     Fraction of account balance deployed (0.0–1.0)
    cash_reserve      REAL     Remaining buying power in USD
    portfolio_beta    REAL     Portfolio weighted beta vs SPY (NULL if unavailable)
    portfolio_vol     REAL     30-day annualised portfolio volatility (NULL if unavailable)
    avg_correlation   REAL     Average pairwise 30-day price correlation (NULL if < 2 positions)
    max_concentration REAL     Largest single-position weight (NULL if empty)
    sector_json       TEXT     JSON {"Tech": 2, "Finance": 1, ...} sector position counts
    strategy_json     TEXT     JSON {"momentum": 2, "swing": 1, ...} strategy position counts
    violations_today  INTEGER  Number of blocked trades logged today
    created_at        TEXT     ISO-8601 UTC timestamp

portfolio_violations
    id             INTEGER  Primary key
    ticker         TEXT     Ticker that was blocked
    strategy       TEXT     Strategy that requested the trade
    amount_usd     REAL     Requested position size in USD
    violation_type TEXT     "duplicate" | "max_positions" | "max_per_strategy" | "max_per_sector" |
                            "max_deployed" | "correlation"
    reason         TEXT     Human-readable explanation
    created_at     TEXT     ISO-8601 UTC timestamp

price_alerts
    id           INTEGER  Primary key
    ticker       TEXT     Stock ticker symbol
    alert_type   TEXT     "stop_loss" | "take_profit" | "price_move" | "volume_spike"
    price        REAL     Current price when alert fired
    stop_loss    REAL     Stop-loss level (NULL for non-SL alerts)
    take_profit  REAL     Take-profit level (NULL for non-TP alerts)
    change_pct   REAL     Intraday price change % (NULL for volume alerts)
    volume_ratio REAL     Volume as multiple of average (NULL for price alerts)
    message      TEXT     Full human-readable alert message
    created_at   TEXT     ISO-8601 UTC timestamp

optimization_results
    id                  INTEGER  Primary key
    ticker              TEXT     Stock ticker symbol
    strategy            TEXT     "momentum" | "mean_reversion" | "swing"
    start_date          TEXT     Optimization range start "YYYY-MM-DD"
    end_date            TEXT     Optimization range end   "YYYY-MM-DD"
    best_params_json    TEXT     JSON dict of optimized parameters
    default_params_json TEXT     JSON dict of default parameters
    best_sharpe         REAL     Average test-window Sharpe with best params
    default_sharpe      REAL     Average test-window Sharpe with default params
    best_return         REAL     Average test-window return % with best params
    default_return      REAL     Average test-window return % with default params
    best_max_dd         REAL     Average test-window max drawdown % with best params
    default_max_dd      REAL     Average test-window max drawdown % with default params
    best_win_rate       REAL     Average win rate % with best params
    default_win_rate    REAL     Average win rate % with default params
    best_trade_count    REAL     Average trade count per test window (best params)
    stability_score     REAL     Std-dev of Sharpe across test windows (lower = more stable)
    windows_tested      INTEGER  Number of walk-forward windows used
    combos_tested       INTEGER  Total parameter combinations evaluated
    window_results_json TEXT     JSON list of per-window metrics for best params
    created_at          TEXT     ISO-8601 UTC timestamp
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

from config.settings import DB_PATH

# ---------------------------------------------------------------------------
# PostgreSQL support
# ---------------------------------------------------------------------------
# When the DATABASE_URL environment variable is set (Railway / Render inject it
# automatically when a PostgreSQL plugin is attached), the Database class
# transparently uses psycopg2 instead of SQLite.  All public method signatures
# are identical in both modes so no calling code needs to change.
#
# Local development (no DATABASE_URL)  → SQLite, file at DB_PATH
# Production (DATABASE_URL set)        → PostgreSQL via psycopg2-binary
# ---------------------------------------------------------------------------


class Database:
    """
    Persistence layer for the News Trading System.

    Supports SQLite (local dev) and PostgreSQL (production).
    Backend is chosen automatically from the DATABASE_URL environment variable.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        """
        Initialise the database and create tables if they don't exist.

        Args:
            db_path: Path to the SQLite file (ignored when DATABASE_URL is set).
        """
        pg_url = os.environ.get("DATABASE_URL", "")
        self._use_pg: bool = bool(pg_url)
        if self._use_pg:
            self._pg_dsn: str = pg_url
        self.db_path = db_path          # kept as public attr for backward compat
        self._db_path = db_path
        self._init_schema()
        if not self._use_pg:
            self._migrate_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        """Return an open database connection for the active backend."""
        if self._use_pg:
            try:
                import psycopg2
                import psycopg2.extras
            except ImportError as exc:
                raise ImportError(
                    "psycopg2-binary is required for PostgreSQL. "
                    "Install it with: pip install psycopg2-binary"
                ) from exc
            return psycopg2.connect(
                self._pg_dsn,
                cursor_factory=psycopg2.extras.RealDictCursor,
            )
        conn = sqlite3.connect(self._db_path, timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _pg_sql(self, sql: str) -> str:
        """Adapt SQLite SQL to PostgreSQL: replace ? placeholders and DDL keywords."""
        return (
            sql
            .replace("?", "%s")
            .replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
            .replace("INSERT OR IGNORE INTO", "INSERT INTO")
        )

    def _insert(self, sql: str, params: tuple) -> int:
        """
        Execute an INSERT statement and return the generated row id.

        For PostgreSQL, appends RETURNING id and fetches the value.
        For SQLite, uses cursor.lastrowid.
        """
        if self._use_pg:
            pg_sql = self._pg_sql(sql).rstrip().rstrip(";") + " RETURNING id"
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(pg_sql, params)
                row_id = cur.fetchone()["id"]
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            return row_id
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            return cur.lastrowid

    def _exec_write(self, sql: str, params: tuple = ()) -> None:
        """Execute an INSERT/UPDATE/DELETE with no return value."""
        if self._use_pg:
            conn = self._connect()
            try:
                conn.cursor().execute(self._pg_sql(sql), params)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            return
        with self._connect() as conn:
            conn.execute(sql, params)

    def _select(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a SELECT and return all rows as a list of dicts."""
        if self._use_pg:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(self._pg_sql(sql), params)
                return [dict(row) for row in cur.fetchall()]
            finally:
                conn.close()
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def _exec_many(self, sql: str, rows: list[tuple]) -> None:
        """Execute an INSERT for many rows (executemany)."""
        if self._use_pg:
            pg_sql = self._pg_sql(sql)
            if "OR IGNORE" in sql:
                pg_sql = pg_sql.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
            conn = self._connect()
            try:
                conn.cursor().executemany(pg_sql, rows)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            return
        with self._connect() as conn:
            conn.executemany(sql, rows)

    def _exec_script(self, sql: str) -> None:
        """Execute a multi-statement DDL/DML script."""
        if self._use_pg:
            pg_sql = self._pg_sql(sql)
            conn = self._connect()
            try:
                cur = conn.cursor()
                for stmt in pg_sql.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        cur.execute(stmt)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            return
        with self._connect() as conn:
            conn.executescript(sql)

    def _init_schema(self) -> None:
        self._exec_script(
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

                CREATE TABLE IF NOT EXISTS screener_results (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at       TEXT    NOT NULL,
                    ticker       TEXT    NOT NULL,
                    name         TEXT,
                    market       TEXT    NOT NULL,
                    exchange     TEXT,
                    country      TEXT,
                    hotness      REAL,
                    price        REAL,
                    price_change REAL,
                    volume_ratio REAL,
                    volume       REAL,
                    rsi          REAL,
                    market_cap   REAL,
                    avg_volume   REAL,
                    metrics      TEXT,
                    created_at   TEXT    NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_screener_run_ticker
                    ON screener_results (run_at, ticker);

                CREATE TABLE IF NOT EXISTS backtest_strategy_comparison (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker        TEXT    NOT NULL,
                    start_date    TEXT    NOT NULL,
                    end_date      TEXT    NOT NULL,
                    strategy      TEXT    NOT NULL,
                    total_return  REAL,
                    sharpe        REAL,
                    max_dd        REAL,
                    win_rate      REAL,
                    trade_count   INTEGER,
                    avg_hold_days REAL,
                    created_at    TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS strategy_signals (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker              TEXT    NOT NULL,
                    strategy            TEXT    NOT NULL,
                    signal              TEXT    NOT NULL,
                    confidence          REAL    NOT NULL,
                    timeframe           TEXT,
                    reasoning           TEXT,
                    indicators_json     TEXT,
                    ensemble_confidence REAL,
                    combined_signal     TEXT,
                    consensus           TEXT,
                    account_balance     REAL,
                    risk_calc_id        INTEGER REFERENCES risk_calculations(id),
                    created_at          TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker                    TEXT    NOT NULL,
                    run_at                    TEXT    NOT NULL,
                    momentum_signal           TEXT,
                    momentum_confidence       REAL,
                    mean_reversion_signal     TEXT,
                    mean_reversion_confidence REAL,
                    swing_signal              TEXT,
                    swing_confidence          REAL,
                    combined_signal           TEXT    NOT NULL,
                    ensemble_confidence       REAL    NOT NULL,
                    consensus                 TEXT    NOT NULL,
                    risk_calc_id              INTEGER REFERENCES risk_calculations(id),
                    account_balance           REAL,
                    errors_json               TEXT,
                    created_at                TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_at       TEXT    NOT NULL,
                    open_positions    INTEGER NOT NULL DEFAULT 0,
                    total_value       REAL    NOT NULL DEFAULT 0,
                    deployed_pct      REAL    NOT NULL DEFAULT 0,
                    cash_reserve      REAL    NOT NULL DEFAULT 0,
                    portfolio_beta    REAL,
                    portfolio_vol     REAL,
                    avg_correlation   REAL,
                    max_concentration REAL,
                    sector_json       TEXT,
                    strategy_json     TEXT,
                    violations_today  INTEGER NOT NULL DEFAULT 0,
                    created_at        TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_violations (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker         TEXT    NOT NULL,
                    strategy       TEXT,
                    amount_usd     REAL,
                    violation_type TEXT    NOT NULL,
                    reason         TEXT    NOT NULL,
                    created_at     TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS price_alerts (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker       TEXT    NOT NULL,
                    alert_type   TEXT    NOT NULL,
                    price        REAL    NOT NULL,
                    stop_loss    REAL,
                    take_profit  REAL,
                    change_pct   REAL,
                    volume_ratio REAL,
                    message      TEXT    NOT NULL,
                    created_at   TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS optimization_results (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker              TEXT    NOT NULL,
                    strategy            TEXT    NOT NULL,
                    start_date          TEXT    NOT NULL,
                    end_date            TEXT    NOT NULL,
                    best_params_json    TEXT    NOT NULL,
                    default_params_json TEXT    NOT NULL,
                    best_sharpe         REAL,
                    default_sharpe      REAL,
                    best_return         REAL,
                    default_return      REAL,
                    best_max_dd         REAL,
                    default_max_dd      REAL,
                    best_win_rate       REAL,
                    default_win_rate    REAL,
                    best_trade_count    REAL,
                    stability_score     REAL,
                    windows_tested      INTEGER NOT NULL DEFAULT 0,
                    combos_tested       INTEGER NOT NULL DEFAULT 0,
                    window_results_json TEXT,
                    created_at          TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS health_checks (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    checked_at      TEXT    NOT NULL,
                    database_ok     INTEGER NOT NULL DEFAULT 0,
                    api_keys_ok     INTEGER NOT NULL DEFAULT 0,
                    scheduler_ok    INTEGER NOT NULL DEFAULT 0,
                    disk_ok         INTEGER NOT NULL DEFAULT 0,
                    memory_ok       INTEGER NOT NULL DEFAULT 0,
                    overall_ok      INTEGER NOT NULL DEFAULT 0,
                    details_json    TEXT,
                    created_at      TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS emergency_stops (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    action          TEXT    NOT NULL,
                    reason          TEXT,
                    activated_by    TEXT,
                    created_at      TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recovery_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    service         TEXT    NOT NULL,
                    event_type      TEXT    NOT NULL,
                    ticker          TEXT,
                    attempt         INTEGER,
                    error_msg       TEXT,
                    recovery_action TEXT,
                    duration_ms     INTEGER,
                    success         INTEGER NOT NULL DEFAULT 0,
                    created_at      TEXT    NOT NULL
                );
                """
        )

    def _migrate_schema(self) -> None:
        """Idempotently add columns/indices introduced after the initial schema."""
        # PostgreSQL always starts with the complete current schema; no migration needed.
        if self._use_pg:
            return
        with self._connect() as conn:
            # Add price and volume columns to screener_results if they don't exist yet
            for col_def in ("price REAL", "volume REAL"):
                try:
                    conn.execute(f"ALTER TABLE screener_results ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass  # column already exists — safe to skip
            # Unique index (safe no-op if the table was just created above)
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_screener_run_ticker "
                "ON screener_results (run_at, ticker)"
            )
            # recovery_log table added in a later release — create if absent
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recovery_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    service         TEXT    NOT NULL,
                    event_type      TEXT    NOT NULL,
                    ticker          TEXT,
                    attempt         INTEGER,
                    error_msg       TEXT,
                    recovery_action TEXT,
                    duration_ms     INTEGER,
                    success         INTEGER NOT NULL DEFAULT 0,
                    created_at      TEXT    NOT NULL
                )
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
        return self._insert(
            "INSERT INTO runs "
            "(ticker, headlines_fetched, headlines_analysed, avg_score, signal, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ticker, headlines_fetched, headlines_analysed, avg_score, signal, now),
        )

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
        self._exec_write(
            "INSERT INTO headline_scores (run_id, headline, sentiment, score, reason) "
            "VALUES (?, ?, ?, ?, ?)",
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
        return self._insert(
            "INSERT INTO technical_signals "
            "(ticker, signal, rsi, macd, macd_signal, macd_hist, "
            " sma_20, sma_50, bb_upper, bb_lower, price, reasoning, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ticker, signal, rsi, macd, macd_signal, macd_hist,
             sma_20, sma_50, bb_upper, bb_lower, price, reasoning, now),
        )

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
        return self._insert(
            "INSERT INTO combined_signals"
            " (ticker, combined_signal, sentiment_signal, technical_signal,"
            "  sentiment_score, confidence, run_id, technical_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker, combined_signal, sentiment_signal, technical_signal,
                sentiment_score, confidence, run_id, technical_id, now,
            ),
        )

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
        return self._insert(
            "INSERT INTO risk_calculations"
            " (ticker, signal, confidence, current_price, account_balance,"
            "  position_size_usd, shares, stop_loss, take_profit,"
            "  risk_amount, kelly_fraction, stop_pct,"
            "  skipped, skip_reason, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker, signal, confidence, current_price, account_balance,
                position_size_usd, shares, stop_loss, take_profit,
                risk_amount, kelly_fraction, stop_pct,
                int(skipped), skip_reason, now,
            ),
        )

    def log_strategy_comparison(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        strategy: str,
        total_return: float,
        sharpe: float,
        max_dd: float,
        win_rate: float,
        trade_count: int,
        avg_hold_days: float = 0.0,
    ) -> int:
        """
        Persist one strategy's result from a compare_strategies() run.

        Args:
            ticker:        Stock ticker symbol.
            start_date:    Backtest start date "YYYY-MM-DD".
            end_date:      Backtest end date "YYYY-MM-DD".
            strategy:      "momentum" | "mean_reversion" | "swing" | "buy_and_hold".
            total_return:  Total return percent.
            sharpe:        Annualised Sharpe ratio.
            max_dd:        Maximum drawdown percent (negative value).
            win_rate:      Win rate percent (0–100).
            trade_count:   Number of completed trades.
            avg_hold_days: Average holding period in calendar days.

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        return self._insert(
            "INSERT INTO backtest_strategy_comparison"
            " (ticker, start_date, end_date, strategy,"
            "  total_return, sharpe, max_dd, win_rate,"
            "  trade_count, avg_hold_days, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker, start_date, end_date, strategy,
                total_return, sharpe, max_dd, win_rate,
                trade_count, avg_hold_days, now,
            ),
        )

    def log_strategy_signal(
        self,
        ticker: str,
        strategy: str,
        signal: str,
        confidence: float,
        timeframe: "str | None" = None,
        reasoning: "str | None" = None,
        indicators_json: "str | None" = None,
        ensemble_confidence: "float | None" = None,
        combined_signal: "str | None" = None,
        consensus: "str | None" = None,
        account_balance: "float | None" = None,
        risk_calc_id: "int | None" = None,
    ) -> int:
        """
        Persist one strategy agent's signal and return its auto-generated ID.

        Args:
            ticker:              Stock ticker symbol.
            strategy:            Agent name: "momentum" | "mean_reversion" | "swing".
            signal:              "BUY" | "SELL" | "HOLD".
            confidence:          Agent confidence 0–100.
            timeframe:           Intended holding period.
            reasoning:           Semicolon-joined list of triggered conditions.
            indicators_json:     JSON-serialised indicator dict.
            ensemble_confidence: Coordinator ensemble confidence 0–100.
            combined_signal:     Coordinator combined direction.
            consensus:           "unanimous" | "majority" | "conflicting" | "none".
            account_balance:     Account value used for sizing.
            risk_calc_id:        FK to risk_calculations.id (None when skipped).

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        return self._insert(
            "INSERT INTO strategy_signals"
            " (ticker, strategy, signal, confidence, timeframe, reasoning,"
            "  indicators_json, ensemble_confidence, combined_signal, consensus,"
            "  account_balance, risk_calc_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker, strategy, signal, confidence, timeframe, reasoning,
                indicators_json, ensemble_confidence, combined_signal, consensus,
                account_balance, risk_calc_id, now,
            ),
        )

    def log_strategy_performance(
        self,
        ticker: str,
        run_at: str,
        combined_signal: str,
        ensemble_confidence: float,
        consensus: str,
        momentum_signal: "str | None" = None,
        momentum_confidence: "float | None" = None,
        mean_reversion_signal: "str | None" = None,
        mean_reversion_confidence: "float | None" = None,
        swing_signal: "str | None" = None,
        swing_confidence: "float | None" = None,
        risk_calc_id: "int | None" = None,
        account_balance: "float | None" = None,
        errors_json: "str | None" = None,
    ) -> int:
        """
        Persist one StrategyCoordinator run summary and return its auto-generated ID.

        Args:
            ticker:                    Stock ticker symbol.
            run_at:                    ISO-8601 timestamp when the coordinator ran.
            combined_signal:           Ensemble direction "BUY" | "SELL" | "HOLD".
            ensemble_confidence:       Weighted ensemble confidence 0–100.
            consensus:                 "unanimous" | "majority" | "conflicting" | "none".
            momentum_signal:           MomentumAgent signal (None if agent failed).
            momentum_confidence:       MomentumAgent confidence (None if agent failed).
            mean_reversion_signal:     MeanReversionAgent signal.
            mean_reversion_confidence: MeanReversionAgent confidence.
            swing_signal:              SwingAgent signal.
            swing_confidence:          SwingAgent confidence.
            risk_calc_id:              FK to risk_calculations.id (None when skipped).
            account_balance:           Account value used for sizing.
            errors_json:               JSON-serialised list of per-agent errors.

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        return self._insert(
            "INSERT INTO strategy_performance"
            " (ticker, run_at, momentum_signal, momentum_confidence,"
            "  mean_reversion_signal, mean_reversion_confidence,"
            "  swing_signal, swing_confidence,"
            "  combined_signal, ensemble_confidence, consensus,"
            "  risk_calc_id, account_balance, errors_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker, run_at,
                momentum_signal, momentum_confidence,
                mean_reversion_signal, mean_reversion_confidence,
                swing_signal, swing_confidence,
                combined_signal, ensemble_confidence, consensus,
                risk_calc_id, account_balance, errors_json, now,
            ),
        )

    def log_price_alert(
        self,
        ticker: str,
        alert_type: str,
        price: float,
        message: str,
        stop_loss: "float | None" = None,
        take_profit: "float | None" = None,
        change_pct: "float | None" = None,
        volume_ratio: "float | None" = None,
    ) -> int:
        """
        Persist a price alert event and return its auto-generated ID.

        Args:
            ticker:       Stock ticker symbol.
            alert_type:   "stop_loss" | "take_profit" | "price_move" | "volume_spike".
            price:        Current price when the alert fired.
            message:      Full human-readable alert message.
            stop_loss:    Stop-loss level (None for non-SL alerts).
            take_profit:  Take-profit level (None for non-TP alerts).
            change_pct:   Intraday price change % (None for volume alerts).
            volume_ratio: Volume multiple of average (None for price alerts).

        Returns:
            The integer primary key of the newly inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        return self._insert(
            "INSERT INTO price_alerts"
            " (ticker, alert_type, price, stop_loss, take_profit,"
            "  change_pct, volume_ratio, message, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ticker, alert_type, price, stop_loss, take_profit,
             change_pct, volume_ratio, message, now),
        )

    def log_portfolio_snapshot(
        self,
        snapshot_at: str,
        open_positions: int,
        total_value: float,
        deployed_pct: float,
        cash_reserve: float,
        portfolio_beta: "float | None" = None,
        portfolio_vol: "float | None" = None,
        avg_correlation: "float | None" = None,
        max_concentration: "float | None" = None,
        sector_json: "str | None" = None,
        strategy_json: "str | None" = None,
        violations_today: int = 0,
    ) -> int:
        """Persist a portfolio state snapshot and return its auto-generated ID."""
        now = datetime.now(timezone.utc).isoformat()
        return self._insert(
            "INSERT INTO portfolio_snapshots"
            " (snapshot_at, open_positions, total_value, deployed_pct,"
            "  cash_reserve, portfolio_beta, portfolio_vol, avg_correlation,"
            "  max_concentration, sector_json, strategy_json,"
            "  violations_today, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot_at, open_positions, total_value, deployed_pct,
                cash_reserve, portfolio_beta, portfolio_vol, avg_correlation,
                max_concentration, sector_json, strategy_json,
                violations_today, now,
            ),
        )

    def log_portfolio_violation(
        self,
        ticker: str,
        violation_type: str,
        reason: str,
        strategy: "str | None" = None,
        amount_usd: "float | None" = None,
    ) -> int:
        """Persist a blocked trade attempt and return its auto-generated ID."""
        now = datetime.now(timezone.utc).isoformat()
        return self._insert(
            "INSERT INTO portfolio_violations"
            " (ticker, strategy, amount_usd, violation_type, reason, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (ticker, strategy, amount_usd, violation_type, reason, now),
        )

    def get_recent_runs(self, limit: int = 10) -> list[dict]:
        """
        Return the most recent analysis runs, newest first.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of dicts, each representing one row from the runs table.
        """
        return self._select("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))

    def get_scores_for_run(self, run_id: int) -> list[dict]:
        """
        Return all headline scores associated with a given run.

        Args:
            run_id: Primary key of the run.

        Returns:
            List of dicts, each representing one headline_scores row.
        """
        return self._select("SELECT * FROM headline_scores WHERE run_id = ?", (run_id,))

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
        return self._insert(
            "INSERT INTO scheduler_logs"
            " (run_at, tickers, signals_generated, trades_executed,"
            "  portfolio_value, duration_seconds, errors, status,"
            "  summary, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def log_screener_results(self, run_at: str, results: list[dict]) -> int:
        """
        Persist a batch of screener candidates and return the row count inserted.

        Args:
            run_at:  ISO-8601 timestamp of the screener run.
            results: List of candidate dicts — each must contain at minimum
                     'ticker' and 'market'.  Optional keys: name, exchange,
                     country, hotness, price_change, volume_ratio, rsi,
                     market_cap, avg_volume.  The full dict is also stored
                     as a JSON blob in the ``metrics`` column.

        Returns:
            Number of rows inserted.
        """
        import json
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                run_at,
                r.get("ticker", ""),
                r.get("name"),
                r.get("market", ""),
                r.get("exchange"),
                r.get("country"),
                r.get("hotness"),
                r.get("price"),
                r.get("price_change"),
                r.get("volume_ratio"),
                r.get("volume"),
                r.get("rsi"),
                r.get("market_cap"),
                r.get("avg_volume"),
                json.dumps(r),
                now,
            )
            for r in results
        ]
        self._exec_many(
            "INSERT OR IGNORE INTO screener_results"
            " (run_at, ticker, name, market, exchange, country,"
            "  hotness, price, price_change, volume_ratio, volume,"
            "  rsi, market_cap, avg_volume, metrics, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def log_optimization_result(
        self,
        ticker: str,
        strategy: str,
        start_date: str,
        end_date: str,
        best_params: dict,
        default_params: dict,
        best_sharpe: float,
        default_sharpe: float,
        best_return: float,
        default_return: float,
        best_max_dd: float,
        default_max_dd: float,
        best_win_rate: float,
        default_win_rate: float,
        best_trade_count: float,
        stability_score: float,
        windows_tested: int,
        combos_tested: int,
        window_results_json: str,
    ) -> int:
        """
        Persist a parameter optimization result and return its auto-generated ID.

        Args:
            ticker:              Stock ticker symbol.
            strategy:            "momentum" | "mean_reversion" | "swing".
            start_date:          Optimization range start "YYYY-MM-DD".
            end_date:            Optimization range end   "YYYY-MM-DD".
            best_params:         Dict of optimized parameters.
            default_params:      Dict of default parameters for comparison.
            best_sharpe:         Avg test-window Sharpe with best params.
            default_sharpe:      Avg test-window Sharpe with default params.
            best_return:         Avg test-window return % with best params.
            default_return:      Avg test-window return % with default params.
            best_max_dd:         Avg test-window max drawdown % with best params.
            default_max_dd:      Avg test-window max drawdown % with default params.
            best_win_rate:       Avg win rate % with best params.
            default_win_rate:    Avg win rate % with default params.
            best_trade_count:    Avg trade count per test window (best params).
            stability_score:     Std-dev of Sharpe across test windows (lower = stable).
            windows_tested:      Number of walk-forward windows used.
            combos_tested:       Total parameter combinations evaluated.
            window_results_json: JSON list of per-window metrics for best params.

        Returns:
            The integer primary key of the newly inserted row.
        """
        import json as _json
        now = datetime.now(timezone.utc).isoformat()
        return self._insert(
            "INSERT INTO optimization_results"
            " (ticker, strategy, start_date, end_date,"
            "  best_params_json, default_params_json,"
            "  best_sharpe, default_sharpe,"
            "  best_return, default_return,"
            "  best_max_dd, default_max_dd,"
            "  best_win_rate, default_win_rate,"
            "  best_trade_count, stability_score,"
            "  windows_tested, combos_tested,"
            "  window_results_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker, strategy, start_date, end_date,
                _json.dumps(best_params), _json.dumps(default_params),
                best_sharpe, default_sharpe,
                best_return, default_return,
                best_max_dd, default_max_dd,
                best_win_rate, default_win_rate,
                best_trade_count, stability_score,
                windows_tested, combos_tested,
                window_results_json, now,
            ),
        )

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
        return self._insert(
            "INSERT INTO backtest_results"
            " (ticker, start_date, end_date, initial_balance, final_balance,"
            "  total_return_pct, buy_and_hold_return_pct, sharpe_ratio,"
            "  max_drawdown_pct, win_rate_pct, avg_win, avg_loss,"
            "  total_trades, sentiment_mode, trades_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker, start_date, end_date, initial_balance, final_balance,
                total_return_pct, buy_and_hold_return_pct, sharpe_ratio,
                max_drawdown_pct, win_rate_pct, avg_win, avg_loss,
                total_trades, sentiment_mode, trades_json, now,
            ),
        )

    def log_health_check(
        self,
        checked_at: str,
        database_ok: bool,
        api_keys_ok: bool,
        scheduler_ok: bool,
        disk_ok: bool,
        memory_ok: bool,
        overall_ok: bool,
        details_json: "str | None" = None,
    ) -> int:
        """Persist a system health-check result and return its auto-generated ID."""
        now = datetime.now(timezone.utc).isoformat()
        return self._insert(
            "INSERT INTO health_checks"
            " (checked_at, database_ok, api_keys_ok, scheduler_ok,"
            "  disk_ok, memory_ok, overall_ok, details_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                checked_at,
                int(database_ok), int(api_keys_ok), int(scheduler_ok),
                int(disk_ok), int(memory_ok), int(overall_ok),
                details_json, now,
            ),
        )

    def log_emergency_stop(
        self,
        action: str,
        reason: "str | None" = None,
        activated_by: "str | None" = None,
    ) -> int:
        """Persist a kill-switch activation event and return its auto-generated ID."""
        now = datetime.now(timezone.utc).isoformat()
        return self._insert(
            "INSERT INTO emergency_stops (action, reason, activated_by, created_at)"
            " VALUES (?, ?, ?, ?)",
            (action, reason, activated_by, now),
        )

    def log_recovery_event(
        self,
        service:         str,
        event_type:      str,
        ticker:          "str | None" = None,
        attempt:         "int | None" = None,
        error_msg:       "str | None" = None,
        recovery_action: "str | None" = None,
        duration_ms:     "int | None" = None,
        success:         bool         = True,
    ) -> int:
        """
        Persist a recovery event to recovery_log and return its auto-generated ID.

        Args:
            service:         Service name ("newsapi" | "anthropic" | "yfinance" |
                             "database" | "network" | "checkpoint").
            event_type:      Event category ("retry" | "circuit_open" |
                             "circuit_close" | "degraded_mode" | "cache_hit" |
                             "network_outage" | "network_restored" |
                             "checkpoint_save" | "checkpoint_resume").
            ticker:          Related ticker symbol if applicable.
            attempt:         Which retry attempt triggered this event.
            error_msg:       Error detail string.
            recovery_action: What recovery action was taken.
            duration_ms:     Wall-clock time in milliseconds.
            success:         Whether the recovery ultimately succeeded.
        """
        now = datetime.now(timezone.utc).isoformat()
        return self._insert(
            "INSERT INTO recovery_log"
            " (service, event_type, ticker, attempt, error_msg,"
            "  recovery_action, duration_ms, success, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                service, event_type, ticker, attempt, error_msg,
                recovery_action, duration_ms, int(success), now,
            ),
        )
