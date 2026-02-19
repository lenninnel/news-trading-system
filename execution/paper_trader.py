"""
PaperTrader — simulated trade execution with SQLite persistence.

Tables (added to the existing news_trading.db):

    portfolio
        ticker          TEXT    PRIMARY KEY
        shares          INTEGER Current holding (0 = flat)
        avg_price       REAL    Weighted average entry price
        current_value   REAL    shares × avg_price (entry-based)

    trade_history
        id              INTEGER PRIMARY KEY AUTOINCREMENT
        timestamp       TEXT    ISO-8601 UTC
        ticker          TEXT
        action          TEXT    BUY | SELL
        shares          INTEGER
        price           REAL    Price per share at execution
        stop_loss       REAL    Stop-loss price (NULL for SELL entries)
        take_profit     REAL    Take-profit price (NULL for SELL entries)
        pnl             REAL    Realised P&L (0 for opening trades)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config.settings import DB_PATH

# Kill-switch flag file location (same directory as project root)
_FLAG_FILE = Path(__file__).resolve().parent.parent / "emergency_stop.flag"


class PaperTrader:
    """Simulated trade execution backed by SQLite.

    Uses the same database file as the rest of the system so all data lives
    in one place.  Tables are created on first use.

    Args:
        db_path: Path to the SQLite file. Defaults to settings.DB_PATH.

    Examples::

        pt = PaperTrader()
        trade_id = pt.track_trade("AAPL", "BUY", 10, 182.50,
                                   stop_loss=178.85, take_profit=190.00)
        print(pt.get_portfolio())
        print(pt.get_trade_history())
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
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
                CREATE TABLE IF NOT EXISTS portfolio (
                    ticker          TEXT    PRIMARY KEY,
                    shares          INTEGER NOT NULL DEFAULT 0,
                    avg_price       REAL    NOT NULL DEFAULT 0,
                    current_value   REAL    NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS trade_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    ticker      TEXT    NOT NULL,
                    action      TEXT    NOT NULL,
                    shares      INTEGER NOT NULL,
                    price       REAL    NOT NULL,
                    stop_loss   REAL,
                    take_profit REAL,
                    pnl         REAL    NOT NULL DEFAULT 0
                );
                """
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_trade(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        stop_loss: "float | None" = None,
        take_profit: "float | None" = None,
    ) -> int:
        """
        Record a simulated trade and update the portfolio.

        For BUY trades the portfolio position is opened or averaged up.
        For SELL trades the realised P&L is computed against the average
        entry price and the position is reduced accordingly.

        Args:
            ticker:      Stock ticker symbol (e.g. "AAPL").
            action:      "BUY" or "SELL".
            shares:      Number of whole shares.
            price:       Price per share at simulated execution.
            stop_loss:   Stop-loss price (informational, stored for reference).
            take_profit: Take-profit price (informational, stored for reference).

        Returns:
            The integer primary key of the new trade_history row.
        """
        # ── Kill-switch check ────────────────────────────────────────────────
        if _FLAG_FILE.exists():
            import json as _json
            try:
                state = _json.loads(_FLAG_FILE.read_text(encoding="utf-8"))
                act = state.get("action", "unknown")
                at  = state.get("activated_at", "unknown")
            except Exception:
                act, at = "unknown", "unknown"
            raise RuntimeError(
                f"Kill switch active ({act} since {at}): trade blocked for {ticker}. "
                f"Run: python3 emergency_stop.py --resume"
            )
        # ── End kill-switch check ─────────────────────────────────────────────

        ticker = ticker.upper()
        action = action.upper()
        now = datetime.now(timezone.utc).isoformat()
        pnl = 0.0

        with self._connect() as conn:
            row = conn.execute(
                "SELECT shares, avg_price FROM portfolio WHERE ticker = ?",
                (ticker,),
            ).fetchone()

            if action == "BUY":
                if row:
                    old_shares = row["shares"]
                    old_avg = row["avg_price"]
                    new_shares = old_shares + shares
                    new_avg = (old_shares * old_avg + shares * price) / new_shares
                    conn.execute(
                        """UPDATE portfolio
                           SET shares = ?, avg_price = ?, current_value = ?
                           WHERE ticker = ?""",
                        (new_shares, new_avg, new_shares * new_avg, ticker),
                    )
                else:
                    conn.execute(
                        """INSERT INTO portfolio (ticker, shares, avg_price, current_value)
                           VALUES (?, ?, ?, ?)""",
                        (ticker, shares, price, shares * price),
                    )

            elif action == "SELL":
                if row and row["shares"] > 0:
                    pnl = (price - row["avg_price"]) * shares
                    new_shares = max(0, row["shares"] - shares)
                    conn.execute(
                        """UPDATE portfolio
                           SET shares = ?, current_value = ?
                           WHERE ticker = ?""",
                        (new_shares, new_shares * row["avg_price"], ticker),
                    )

            cur = conn.execute(
                """INSERT INTO trade_history
                       (timestamp, ticker, action, shares, price,
                        stop_loss, take_profit, pnl)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, ticker, action, shares, price, stop_loss, take_profit, pnl),
            )
            return cur.lastrowid

    def get_portfolio(self) -> list[dict]:
        """Return all current positions where shares > 0."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM portfolio WHERE shares > 0 ORDER BY ticker"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_trade_history(self) -> list[dict]:
        """Return all trades, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_history ORDER BY id DESC"
            ).fetchall()
            return [dict(row) for row in rows]
