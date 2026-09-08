"""
PortfolioManager — diversification and correlation controls for the paper portfolio.

Wraps PaperTrader to enforce position limits, sector concentration caps,
strategy-type caps, correlation-based blocks, and capital deployment limits.

Position limits (configurable via class constants)
---------------------------------------------------
  MAX_POSITIONS        8   total open positions
  MAX_PER_STRATEGY     2   positions per strategy type (momentum/mean-reversion/swing)
  MAX_PER_SECTOR       3   positions in the same broad sector
  MAX_DEPLOYED_PCT    60%  of account balance actually deployed
  MAX_POSITION_PCT    15%  a single position may represent of the portfolio
  MAX_SECTOR_PCT      40%  one sector may represent (blocks new entries above this)
  MAX_CORRELATION    0.80  pairwise 30-day price correlation that blocks a new entry
  WARN_CORRELATION   0.60  average portfolio correlation that triggers a warning

Sector groups
-------------
  yfinance "sector" strings are normalised to one of:
  Tech | Finance | Healthcare | Energy | Consumer | Industrial | Other

CLI
---
  python3 -m execution.portfolio_manager --balance 10000
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import (
    DB_PATH,
    DRAWDOWN_HALT_THRESHOLD,
    REENTRY_LOCK_INCLUDE_TRAILING,
    REENTRY_LOCK_SESSIONS,
)
from data.alpaca_data import AlpacaDataClient
from data.market_calendar import NY_TZ, next_us_trading_day, us_sessions_between
from execution.paper_trader import PaperTrader
from storage.database import Database

log = logging.getLogger(__name__)

# ── Sector normalisation ───────────────────────────────────────────────────────

_SECTOR_MAP: dict[str, str] = {
    "Technology":              "Tech",
    "Communication Services":  "Tech",
    "Financial Services":      "Finance",
    "Real Estate":             "Finance",
    "Healthcare":              "Healthcare",
    "Energy":                  "Energy",
    "Utilities":               "Energy",
    "Consumer Cyclical":       "Consumer",
    "Consumer Defensive":      "Consumer",
    "Industrials":             "Industrial",
    "Basic Materials":         "Industrial",
}

_SECTOR_CACHE: dict[str, str] = {}   # ticker → normalised sector (process-level cache)


# ── Re-entry lock after a stop-loss exit (Q-016 → 2026-09-08) ──────────────────
# Block a BUY on a ticker whose LAST closed round-trip ended in a stop-loss
# exit within REENTRY_LOCK_SESSIONS US trading sessions (blind time gate — no
# downtrend/price condition).  The stop is reconstructed from trade_history:
# a closing SELL is a stop-loss exit when its executed price (fallback: price)
# is at/below the opening BUY's stop_loss, allowing a small tolerance for
# gap-throughs.  On the audited book this rule and the PositionManager's own
# 'stop_loss_triggered' rows agree on 84/84 entry stops and flag 0/57
# trailing exits and 0/24 targets (scripts/reentry_after_stop_audit.py).
# Session counting is holiday-aware (data.market_calendar): the 1-session
# cool-down it replaces counted Fri 2026-07-03 (a holiday) as a session and
# let two re-entries through.  Trailing-stop exits (a PositionManager
# 'stop_loss_triggered' row at a level above the entry stop) arm the lock
# only when REENTRY_LOCK_INCLUDE_TRAILING is set.
_COOLDOWN_STOP_TOL = 0.003   # 0.3% above stop_loss still counts as a stop exit
_COOLDOWN_SESSIONS = REENTRY_LOCK_SESSIONS
# PositionManager logs its SELL decision to signal_events a few seconds
# around the trade_history row; this is the match window for that lookup.
_PM_EXIT_MATCH_BEFORE_S = 15 * 60
_PM_EXIT_MATCH_AFTER_S = 2 * 60


def _fetch_sector(ticker: str) -> str:
    """Fetch the normalised sector for *ticker* from yfinance.

    Alpaca does not provide sector data, so yfinance is retained for this
    metadata lookup.  Falls back to 'Other' on any failure.
    """
    if ticker in _SECTOR_CACHE:
        return _SECTOR_CACHE[ticker]
    try:
        import yfinance as yf
        info   = yf.Ticker(ticker).info
        raw    = info.get("sector", "") or ""
        sector = _SECTOR_MAP.get(raw, "Other") if raw else "Other"
    except Exception:
        sector = "Other"
    _SECTOR_CACHE[ticker] = sector
    return sector


# ── PortfolioManager ──────────────────────────────────────────────────────────

class PortfolioManager:
    """
    Diversification and correlation guard for the paper portfolio.

    Args:
        account_balance: Total account size in USD (used for deployment %).
        db_path:         Path to the shared SQLite file.
    """

    # ── Limits ────────────────────────────────────────────────────────────────
    MAX_POSITIONS     = 8
    # MAX_PER_STRATEGY tuned for the active 11-ticker universe: router
    # routes 3 tickers to Momentum and 8 to Pullback. A cap of 4 keeps
    # Pullback from filling every slot while never binding on Momentum
    # (which only has 3 candidates). Total is still bounded by MAX_POSITIONS.
    MAX_PER_STRATEGY  = 4
    # MAX_PER_SECTOR=4: 6 of 11 watchlist tickers are Tech (META, AAPL,
    # MSFT, AMZN, TSLA, VRT). A cap of 3 was too tight — would refuse the
    # 4th Tech name even if signals all fired. 4 leaves room for diversification
    # without forcing rejection on a Tech-heavy day.
    MAX_PER_SECTOR    = 4
    MAX_DEPLOYED_PCT  = 0.60
    MAX_POSITION_PCT  = 0.15
    MAX_SECTOR_PCT    = 0.40
    MAX_CORRELATION   = 0.80
    WARN_CORRELATION  = 0.60
    WARN_THRESHOLD    = 0.80   # 80 % of a hard limit → yellow warning

    # ── Correlation / beta window ─────────────────────────────────────────────
    LOOKBACK_DAYS = 45     # calendar days to download (≈ 30 trading days)

    def __init__(
        self,
        account_balance: float = 10_000.0,
        db_path: str = DB_PATH,
    ) -> None:
        self._balance      = account_balance
        self._db_path      = db_path
        # Latent bug: PaperTrader's first arg is `db: Database | None`,
        # not a path. Passing the path string left self._db as the string,
        # so `_paper_trader.get_portfolio() -> self._db.get_portfolio()`
        # raised "'str' object has no attribute 'get_portfolio'" the first
        # time PortfolioManager actually ran in production (US_PRE 2026-05-04
        # after wiring + EXECUTE_TRADES=true exposed the path). Pass a
        # proper Database instance bound to the same path.
        self._paper_trader = PaperTrader(Database(db_path))
        self._db           = Database(db_path)
        self._init_meta_schema()

    def set_account_balance(self, balance: float) -> None:
        """Update the balance used for `MAX_DEPLOYED_PCT` cap math.

        Coordinator instantiates the PortfolioManager once at startup
        with a placeholder balance; the scheduler refreshes this each
        session from live IBKR NetLiquidation so the deployment cap
        scales with the real account value (was hardcoded against an
        outdated $98k baseline).
        """
        if balance and balance > 0:
            self._balance = float(balance)

    # ------------------------------------------------------------------
    # Schema: position_metadata (strategy + sector per open position)
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_meta_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS position_metadata (
                    ticker      TEXT PRIMARY KEY,
                    strategy    TEXT NOT NULL DEFAULT 'unknown',
                    sector      TEXT NOT NULL DEFAULT 'Other',
                    entry_date  TEXT NOT NULL,
                    entry_price REAL NOT NULL DEFAULT 0
                );
                """
            )

    def register_position(
        self,
        ticker: str,
        strategy: str,
        entry_price: float = 0.0,
    ) -> None:
        """
        Record metadata for a newly opened position.

        Call this right after PaperTrader.track_trade("BUY", ...) to keep
        position_metadata in sync.

        Args:
            ticker:      Stock ticker symbol.
            strategy:    "momentum" | "mean-reversion" | "swing" | "all".
            entry_price: Price per share at entry.
        """
        sector     = _fetch_sector(ticker)
        entry_date = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO position_metadata
                    (ticker, strategy, sector, entry_date, entry_price)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ticker.upper(), strategy, sector, entry_date, entry_price),
            )

    def clear_position_meta(self, ticker: str) -> None:
        """Remove metadata when a position is fully closed."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM position_metadata WHERE ticker = ?",
                (ticker.upper(),),
            )

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    def can_add_position(
        self,
        ticker: str,
        strategy: str,
        amount_usd: float,
        session: str | None = None,
        price: float | None = None,
    ) -> tuple[bool, str]:
        """
        Check whether a new position in *ticker* is allowed under all limits.

        Args:
            ticker:     Stock ticker symbol.
            strategy:   Strategy requesting the trade.
            amount_usd: Dollar value of the proposed position.
            session:    Scheduler session name (US_OPEN, PEAD_OPEN, …); only
                        carried into the signal_events row a rejection writes.
            price:      Proposed entry price, same purpose.

        Returns:
            (allowed, reason)  — reason is empty string when allowed.
        """
        ticker   = ticker.upper()
        strategy = strategy.lower().replace("-", "_")

        # 0. Drawdown halt — BUY-only block. Designed 2026-05-01 / shipped
        #    2026-05-12. Manual unlock only (see execution.drawdown_halt
        #    CLI). PositionManager-driven SELLs / stops / TPs bypass this
        #    gate entirely because they never call can_add_position.
        halted, dd_reason = self._check_drawdown_halt()
        if halted:
            self._log_violation(ticker, strategy, amount_usd, "drawdown_halt", dd_reason)
            return False, dd_reason

        positions = self._open_positions_with_meta()
        tickers   = [p["ticker"] for p in positions]

        # 1. Duplicate ticker check
        if ticker in tickers:
            reason = f"Already holding {ticker}"
            self._log_violation(ticker, strategy, amount_usd, "duplicate", reason)
            return False, reason

        # 1b. Re-entry lock after a stop-loss exit (Q-016, widened 2026-09-08).
        #     A just-stopped ticker is no longer held, so the duplicate gate
        #     above cannot catch it; this is a per-ticker eligibility gate that
        #     blocks a re-entry within _COOLDOWN_SESSIONS trading sessions of
        #     the last stop-loss exit.  Every caller that can open a position
        #     (run_combined, run_combined_us_open, the cached US_OPEN executor,
        #     the PEAD path) comes through here, so this is the single place
        #     the lock lives.  A rejection is written to portfolio_violations
        #     AND to signal_events (reason + remaining sessions) so the effect
        #     can be measured later.  Fails OPEN — a guard bug must never block
        #     all trading.
        lock = self._reentry_lock_status(ticker)
        if lock is not None:
            cd_reason = lock["reason"]
            self._log_violation(ticker, strategy, amount_usd, "cooldown_stop", cd_reason)
            self._log_reentry_lock_event(ticker, strategy, session, price, lock)
            return False, cd_reason

        # 2. Total position cap
        if len(positions) >= self.MAX_POSITIONS:
            reason = f"Max {self.MAX_POSITIONS} open positions reached"
            self._log_violation(ticker, strategy, amount_usd, "max_positions", reason)
            return False, reason

        # 3. Per-strategy cap
        strat_count = sum(1 for p in positions if p["strategy"] == strategy)
        if strat_count >= self.MAX_PER_STRATEGY:
            reason = (
                f"Strategy '{strategy}' already has "
                f"{strat_count}/{self.MAX_PER_STRATEGY} positions"
            )
            self._log_violation(ticker, strategy, amount_usd, "max_per_strategy", reason)
            return False, reason

        # 4. Sector cap (fetch sector first)
        sector       = _fetch_sector(ticker)
        sector_count = sum(1 for p in positions if p["sector"] == sector)
        if sector_count >= self.MAX_PER_SECTOR:
            reason = (
                f"Sector '{sector}' already has "
                f"{sector_count}/{self.MAX_PER_SECTOR} positions"
            )
            self._log_violation(ticker, strategy, amount_usd, "max_per_sector", reason)
            return False, reason

        # 5. Capital deployment cap
        total_deployed = sum(p["current_value"] for p in positions)
        max_deploy     = self._balance * self.MAX_DEPLOYED_PCT
        if total_deployed + amount_usd > max_deploy:
            reason = (
                f"Deployment cap: ${total_deployed:,.0f} deployed + "
                f"${amount_usd:,.0f} new > ${max_deploy:,.0f} limit"
            )
            self._log_violation(ticker, strategy, amount_usd, "max_deployed", reason)
            return False, reason

        # 6. Correlation check (skip if fewer than 2 existing positions)
        if len(positions) >= 2:
            corr_blocked, corr_reason = self._check_correlation(ticker, tickers)
            if corr_blocked:
                self._log_violation(ticker, strategy, amount_usd, "correlation", corr_reason)
                return False, corr_reason

        return True, ""

    # ------------------------------------------------------------------
    # Diversification metrics
    # ------------------------------------------------------------------

    def get_diversification_metrics(self) -> dict:
        """
        Return a snapshot of portfolio diversification.

        Returns dict with keys:
            open_positions   (int)
            total_value      (float)
            deployed_pct     (float)  0–1
            cash_reserve     (float)
            by_sector        (dict str → int)   position counts
            by_strategy      (dict str → int)   position counts
            sector_pcts      (dict str → float) portfolio weight per sector
            strategy_pcts    (dict str → float) portfolio weight per strategy
        """
        positions = self._open_positions_with_meta()
        total     = sum(p["current_value"] for p in positions)

        by_sector   = {}
        by_strategy = {}
        sector_val  = {}
        strat_val   = {}

        for p in positions:
            s = p["sector"]
            t = p["strategy"]
            v = p["current_value"]
            by_sector[s]   = by_sector.get(s, 0) + 1
            by_strategy[t] = by_strategy.get(t, 0) + 1
            sector_val[s]  = sector_val.get(s, 0.0) + v
            strat_val[t]   = strat_val.get(t, 0.0) + v

        sector_pcts   = {k: v / total if total > 0 else 0.0 for k, v in sector_val.items()}
        strategy_pcts = {k: v / total if total > 0 else 0.0 for k, v in strat_val.items()}
        deployed_pct  = total / self._balance if self._balance > 0 else 0.0

        return {
            "open_positions": len(positions),
            "total_value":    total,
            "deployed_pct":   deployed_pct,
            "cash_reserve":   max(0.0, self._balance - total),
            "by_sector":      by_sector,
            "by_strategy":    by_strategy,
            "sector_pcts":    sector_pcts,
            "strategy_pcts":  strategy_pcts,
            "positions":      positions,
        }

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Return a pairwise 30-day price-return correlation matrix for open holdings.

        Returns an empty DataFrame when fewer than 2 positions are open.
        """
        positions = self._open_positions_with_meta()
        tickers   = [p["ticker"] for p in positions]
        if len(tickers) < 2:
            return pd.DataFrame()

        returns = self._download_returns(tickers)
        if returns.empty or returns.shape[1] < 2:
            return pd.DataFrame()
        return returns.corr()

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def check_risk_limits(self) -> dict:
        """
        Compute portfolio-level risk metrics and return warnings.

        Returns dict with keys:
            beta              (float | None)
            volatility        (float | None)   annualised, 0–1
            max_concentration (float | None)   0–1 weight of largest position
            avg_correlation   (float | None)
            cash_reserve_pct  (float)
            warnings          (list[str])      human-readable warnings
        """
        positions    = self._open_positions_with_meta()
        tickers      = [p["ticker"] for p in positions]
        total_value  = sum(p["current_value"] for p in positions)
        warnings: list[str] = []

        # Beta + volatility via yfinance returns
        beta     = None
        vol      = None
        avg_corr = None

        if tickers:
            returns = self._download_returns(tickers + ["SPY"])

            if not returns.empty:
                port_weights = {}
                for p in positions:
                    port_weights[p["ticker"]] = (
                        p["current_value"] / total_value if total_value > 0 else 0.0
                    )

                # Portfolio daily return = weighted sum of individual returns
                stock_cols = [c for c in returns.columns if c != "SPY"]
                if stock_cols:
                    port_ret = sum(
                        returns[t] * port_weights.get(t, 0.0)
                        for t in stock_cols
                        if t in returns.columns
                    )
                    spy_ret = returns.get("SPY", pd.Series(dtype=float))

                    # Beta
                    if not spy_ret.empty and spy_ret.std() > 0:
                        cov   = port_ret.cov(spy_ret)
                        var   = spy_ret.var()
                        beta  = round(cov / var, 2) if var > 0 else None

                    # Annualised volatility
                    if port_ret.std() > 0:
                        vol = round(port_ret.std() * (252 ** 0.5), 4)

                # Average pairwise correlation (stocks only)
                if len(stock_cols) >= 2:
                    corr_mat = returns[stock_cols].corr()
                    n        = len(stock_cols)
                    upper    = [
                        corr_mat.iloc[i, j]
                        for i in range(n)
                        for j in range(i + 1, n)
                    ]
                    if upper:
                        avg_corr = round(sum(upper) / len(upper), 3)

        # Max position concentration
        max_conc = None
        if total_value > 0 and positions:
            max_val  = max(p["current_value"] for p in positions)
            max_conc = round(max_val / total_value, 3)
            max_tick = next(
                p["ticker"] for p in positions
                if p["current_value"] == max_val
            )
            if max_conc > self.MAX_POSITION_PCT:
                pct = f"{max_conc:.0%}"
                warnings.append(
                    f"{max_tick} is {pct} of portfolio — consider partial close "
                    f"(limit: {self.MAX_POSITION_PCT:.0%})"
                )

        # Correlation warning
        if avg_corr is not None and avg_corr > self.WARN_CORRELATION:
            warnings.append(
                f"Average portfolio correlation {avg_corr:.2f} > "
                f"{self.WARN_CORRELATION:.2f} threshold — consider adding uncorrelated assets"
            )

        # Sector concentration warning
        div = self.get_diversification_metrics()
        for sector, pct in div["sector_pcts"].items():
            if pct > self.MAX_SECTOR_PCT:
                warnings.append(
                    f"Sector '{sector}' is {pct:.0%} of portfolio "
                    f"(limit: {self.MAX_SECTOR_PCT:.0%}) — new entries blocked"
                )

        cash_pct = div["cash_reserve"] / self._balance if self._balance > 0 else 1.0

        return {
            "beta":              beta,
            "volatility":        vol,
            "max_concentration": max_conc,
            "avg_correlation":   avg_corr,
            "cash_reserve_pct":  round(cash_pct, 3),
            "warnings":          warnings,
        }

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance_if_needed(self) -> list[dict]:
        """
        Identify positions / sectors that are out of balance.

        Returns a list of action dicts:
            {"action": "partial_close", "ticker": ..., "reason": ...}
            {"action": "block_sector", "sector": ..., "reason": ...}
            {"action": "hedge_warning", "reason": ...}
        """
        positions   = self._open_positions_with_meta()
        total_value = sum(p["current_value"] for p in positions)
        actions: list[dict] = []

        if total_value <= 0:
            return actions

        # Positions >15% of portfolio
        for p in positions:
            conc = p["current_value"] / total_value
            if conc > self.MAX_POSITION_PCT:
                actions.append({
                    "action": "partial_close",
                    "ticker": p["ticker"],
                    "reason": (
                        f"{p['ticker']} is {conc:.0%} of portfolio "
                        f"(limit: {self.MAX_POSITION_PCT:.0%}) — flag for partial close"
                    ),
                })

        # Sector concentration >40%
        div = self.get_diversification_metrics()
        for sector, pct in div["sector_pcts"].items():
            if pct > self.MAX_SECTOR_PCT:
                actions.append({
                    "action": "block_sector",
                    "sector": sector,
                    "reason": (
                        f"Sector '{sector}' is {pct:.0%} of portfolio "
                        f"— block new {sector} entries"
                    ),
                })

        # Correlation spike
        corr_df = self.get_correlation_matrix()
        if not corr_df.empty:
            n       = len(corr_df)
            upper   = [
                corr_df.iloc[i, j]
                for i in range(n)
                for j in range(i + 1, n)
            ]
            avg = sum(upper) / len(upper) if upper else 0.0
            if avg > self.WARN_CORRELATION:
                actions.append({
                    "action": "hedge_warning",
                    "reason": (
                        f"Average correlation {avg:.2f} — "
                        "consider adding a hedge or reducing correlated exposure"
                    ),
                })

        return actions

    # ------------------------------------------------------------------
    # Capacity summary (for CLI and display)
    # ------------------------------------------------------------------

    def capacity_summary(self) -> dict:
        """
        Return a dict summarising remaining capacity across all limit dimensions.

        Keys (per limit):
            positions_used / positions_max
            deployed_usd / deploy_max_usd
            deployed_pct
            by_strategy: {name: {used, max, remaining}}
            by_sector:   {name: {used, max, remaining, pct_of_portfolio}}
            warnings: list[str]    (80 % threshold)
        """
        positions   = self._open_positions_with_meta()
        total_value = sum(p["current_value"] for p in positions)
        deployed    = total_value
        deploy_max  = self._balance * self.MAX_DEPLOYED_PCT

        warnings: list[str] = []

        # Position count warning
        pos_pct = len(positions) / self.MAX_POSITIONS
        if pos_pct >= self.WARN_THRESHOLD:
            warnings.append(
                f"Position count at {len(positions)}/{self.MAX_POSITIONS} "
                f"({pos_pct:.0%}) — near limit"
            )

        # Deployment warning
        dep_pct = deployed / deploy_max if deploy_max > 0 else 0.0
        if dep_pct >= self.WARN_THRESHOLD:
            warnings.append(
                f"Capital deployed at {dep_pct:.0%} of {self.MAX_DEPLOYED_PCT:.0%} limit"
            )

        # Strategy capacity
        by_strategy = {}
        for strat in ("momentum", "mean_reversion", "swing"):
            used = sum(1 for p in positions if p["strategy"] == strat)
            remaining = self.MAX_PER_STRATEGY - used
            by_strategy[strat] = {
                "used": used, "max": self.MAX_PER_STRATEGY, "remaining": remaining,
            }
            if used >= self.MAX_PER_STRATEGY:
                warnings.append(f"Strategy '{strat}' is at max ({used}/{self.MAX_PER_STRATEGY})")
            elif used / self.MAX_PER_STRATEGY >= self.WARN_THRESHOLD:
                warnings.append(
                    f"Strategy '{strat}' near limit ({used}/{self.MAX_PER_STRATEGY})"
                )

        # Sector capacity
        all_sectors: set[str] = set(p["sector"] for p in positions) | set()
        by_sector: dict[str, dict] = {}
        for sec in all_sectors:
            used      = sum(1 for p in positions if p["sector"] == sec)
            sec_val   = sum(p["current_value"] for p in positions if p["sector"] == sec)
            sec_pct   = sec_val / total_value if total_value > 0 else 0.0
            remaining = self.MAX_PER_SECTOR - used
            by_sector[sec] = {
                "used": used, "max": self.MAX_PER_SECTOR, "remaining": remaining,
                "pct_of_portfolio": sec_pct,
            }
            if sec_pct > self.MAX_SECTOR_PCT:
                warnings.append(
                    f"Sector '{sec}' at {sec_pct:.0%} — new entries blocked "
                    f"(limit: {self.MAX_SECTOR_PCT:.0%})"
                )

        return {
            "positions_used":  len(positions),
            "positions_max":   self.MAX_POSITIONS,
            "deployed_usd":    deployed,
            "deploy_max_usd":  deploy_max,
            "deployed_pct":    deployed / self._balance if self._balance > 0 else 0.0,
            "cash_reserve":    max(0.0, self._balance - deployed),
            "by_strategy":     by_strategy,
            "by_sector":       by_sector,
            "warnings":        warnings,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_positions_with_meta(self) -> list[dict]:
        """Return open portfolio positions merged with position_metadata."""
        portfolio = self._paper_trader.get_portfolio()  # shares > 0
        if not portfolio:
            return []

        tickers = [p["ticker"] for p in portfolio]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT ticker, strategy, sector, entry_date, entry_price
                FROM position_metadata
                WHERE ticker IN ({','.join('?' * len(tickers))})
                """,
                tickers,
            ).fetchall()
        meta = {r["ticker"]: dict(r) for r in rows}

        merged = []
        for p in portfolio:
            t   = p["ticker"]
            m   = meta.get(t, {})
            merged.append({
                "ticker":        t,
                "shares":        p["shares"],
                "avg_price":     p["avg_price"],
                "current_value": p["current_value"],
                "strategy":      m.get("strategy", "unknown"),
                "sector":        m.get("sector", _fetch_sector(t)),
                "entry_date":    m.get("entry_date", ""),
                "entry_price":   m.get("entry_price", p["avg_price"]),
            })
        return merged

    def _check_correlation(
        self, ticker: str, existing_tickers: list[str]
    ) -> tuple[bool, str]:
        """
        Download 30-day returns for *ticker* and all existing holdings.

        Returns (blocked, reason).  blocked=True when any pairwise correlation
        with the new ticker exceeds MAX_CORRELATION.
        """
        try:
            all_tickers = existing_tickers + [ticker]
            returns     = self._download_returns(all_tickers)
            if returns.empty or ticker not in returns.columns:
                return False, ""

            new_ret = returns[ticker]
            for existing in existing_tickers:
                if existing not in returns.columns:
                    continue
                corr = new_ret.corr(returns[existing])
                if corr > self.MAX_CORRELATION:
                    return True, (
                        f"{ticker} has {corr:.2f} correlation with {existing} "
                        f"(limit: {self.MAX_CORRELATION:.2f})"
                    )
        except Exception as exc:
            log.warning("Correlation check failed for %s: %s", ticker, exc)
        return False, ""

    @staticmethod
    def _download_returns(tickers: list[str]) -> pd.DataFrame:
        """
        Download ~30 trading days of adjusted close for *tickers* via Alpaca
        and return daily percentage returns.  Empty DataFrame on failure.
        """
        try:
            alpaca = AlpacaDataClient()
            close_dict: dict[str, pd.Series] = {}
            for t in tickers:
                try:
                    bars = alpaca.get_bars(t, "1Day", limit=PortfolioManager.LOOKBACK_DAYS)
                    if not bars.empty and "Close" in bars.columns:
                        close_dict[t] = bars["Close"]
                except Exception as exc:
                    log.debug("Alpaca bars failed for %s in correlation check: %s", t, exc)

            if not close_dict:
                return pd.DataFrame()

            close = pd.DataFrame(close_dict)
            returns = close.pct_change().dropna()
            return returns
        except Exception as exc:
            log.warning("Price download failed: %s", exc)
            return pd.DataFrame()

    def _log_violation(
        self,
        ticker: str,
        strategy: str,
        amount_usd: float,
        violation_type: str,
        reason: str,
    ) -> None:
        try:
            self._db.log_portfolio_violation(
                ticker         = ticker,
                violation_type = violation_type,
                reason         = reason,
                strategy       = strategy,
                amount_usd     = amount_usd,
            )
        except Exception as exc:
            log.warning("Could not persist portfolio violation: %s", exc)

    def _recently_stopped(self, ticker: str) -> tuple[bool, "str | None"]:
        """(blocked, reason) view of :meth:`_reentry_lock_status`."""
        lock = self._reentry_lock_status(ticker)
        if lock is None:
            return False, None
        return True, lock["reason"]

    def _reentry_lock_status(self, ticker: str) -> "dict | None":
        """Return the active re-entry lock for *ticker*, or None.

        Locked when the ticker's LAST closed round-trip ended in a stop-loss
        exit (see the module constants for the classification) and fewer than
        ``_COOLDOWN_SESSIONS + 1`` US trading sessions have passed since that
        exit — same session = 0, next trading day = 1; weekends and full-day
        holidays are not sessions (``data.market_calendar``).

        The returned dict carries what the log row needs::

            exit_kind      "stop_loss" | "trailing_stop"
            stop_at        ISO timestamp of the closing SELL
            stop_date      New York date of that SELL
            gap_sessions   sessions elapsed since the stop (as of today)
            remaining      sessions still locked after today
            eligible_from  first New York date a BUY is allowed again
            reason         human-readable sentence for logs / violations

        Fails OPEN on any ambiguity (no closed round-trip, no opening
        stop_loss, unparseable prices/dates) or read error — a guard bug must
        never block all trading.
        """
        ticker = ticker.upper()
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT action, price, executed_price, stop_loss, created_at "
                    "FROM trade_history WHERE ticker = ? "
                    "ORDER BY created_at ASC, id ASC",
                    (ticker,),
                ).fetchall()
        except Exception as exc:
            log.warning("re-entry lock read failed for %s: %s", ticker, exc)
            return None

        # FIFO-pair BUYs with SELLs to isolate the LAST completed round-trip.
        open_buys: list = []
        last_rt: "tuple | None" = None  # (opening_buy_row, closing_sell_row)
        for r in rows:
            action = (r["action"] or "").upper()
            if action == "BUY":
                open_buys.append(r)
            elif action == "SELL":
                if open_buys:
                    last_rt = (open_buys.pop(0), r)  # FIFO
        if last_rt is None:
            return None  # never closed a round-trip → nothing to lock

        buy, sell = last_rt
        stop = buy["stop_loss"]
        if stop is None:
            return None  # cannot classify a stop without the opening stop

        close_px = sell["executed_price"]
        if close_px is None:
            close_px = sell["price"]
        if close_px is None:
            return None

        try:
            stop = float(stop)
            close_px = float(close_px)
            sell_dt = datetime.fromisoformat(sell["created_at"])
        except (TypeError, ValueError):
            return None
        if stop <= 0:
            return None
        if sell_dt.tzinfo is None:
            sell_dt = sell_dt.replace(tzinfo=timezone.utc)

        # Classification — price-vs-entry-stop first (blind, no PM row needed).
        if close_px <= stop * (1.0 + _COOLDOWN_STOP_TOL):
            exit_kind = "stop_loss"
        elif REENTRY_LOCK_INCLUDE_TRAILING and self._pm_logged_stop_exit(ticker, sell_dt):
            exit_kind = "trailing_stop"
        else:
            return None  # target / other exit → no lock

        # Trading-session gap between the stop and today, in New York dates.
        try:
            stop_date = sell_dt.astimezone(NY_TZ).date()
            today = datetime.now(NY_TZ).date()
            if stop_date > today:
                return None  # future-dated anomaly → fail open
            gap = us_sessions_between(stop_date, today)
            eligible_from = stop_date
            for _ in range(_COOLDOWN_SESSIONS + 1):
                eligible_from = next_us_trading_day(eligible_from)
        except Exception as exc:
            log.warning("re-entry lock gap calc failed for %s: %s", ticker, exc)
            return None

        if gap > _COOLDOWN_SESSIONS:
            return None

        remaining = _COOLDOWN_SESSIONS - gap
        label = "stop-loss" if exit_kind == "stop_loss" else "trailing-stop"
        reason = (
            f"Re-entry lock: {ticker} {label} exit on {stop_date.isoformat()} "
            f"({gap} of {_COOLDOWN_SESSIONS} trading session{'' if _COOLDOWN_SESSIONS == 1 else 's'} elapsed, "
            f"{remaining} remaining) — no BUY before {eligible_from.isoformat()}"
        )
        return {
            "exit_kind": exit_kind,
            "stop_at": sell_dt.isoformat(),
            "stop_date": stop_date.isoformat(),
            "gap_sessions": gap,
            "remaining": remaining,
            "eligible_from": eligible_from.isoformat(),
            "reason": reason,
        }

    def _pm_logged_stop_exit(self, ticker: str, sell_dt: datetime) -> bool:
        """True when PositionManager logged a 'stop_loss_triggered' SELL for
        *ticker* around *sell_dt* (its signal_events row lands within seconds
        of the trade_history row).  Used only for the trailing-stop option;
        any read problem counts as "no such row" (fail open)."""
        try:
            lo = (sell_dt - timedelta(seconds=_PM_EXIT_MATCH_BEFORE_S)).astimezone(timezone.utc)
            hi = (sell_dt + timedelta(seconds=_PM_EXIT_MATCH_AFTER_S)).astimezone(timezone.utc)
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM signal_events WHERE ticker = ? "
                    "AND strategy = 'PositionManager' AND signal = 'SELL' "
                    "AND bull_case LIKE 'stop_loss_triggered%' "
                    "AND timestamp BETWEEN ? AND ? LIMIT 1",
                    (ticker, lo.isoformat(), hi.isoformat()),
                ).fetchone()
            return row is not None
        except Exception as exc:
            log.warning("re-entry lock PM-row lookup failed for %s: %s", ticker, exc)
            return False

    def _log_reentry_lock_event(
        self,
        ticker: str,
        strategy: str,
        session: "str | None",
        price: "float | None",
        lock: dict,
    ) -> None:
        """Write the rejected BUY to signal_events (never raises).

        Row shape: strategy='PortfolioManager', signal='HOLD',
        signal_path='REENTRY_LOCK', trade_executed=0; bear_case carries the
        machine-readable detail (exit kind, stop timestamp, sessions elapsed /
        remaining, eligible date), bull_case the strategy that asked.
        """
        try:
            from analytics.signal_logger import SignalLogger
            SignalLogger(db=self._db).log({
                "session": session,
                "ticker": ticker,
                "strategy": "PortfolioManager",
                "signal": "HOLD",
                "signal_path": "REENTRY_LOCK",
                "trade_executed": 0,
                "price_at_signal": price,
                "bull_case": f"BUY requested by {strategy}",
                "bear_case": (
                    f"reentry_lock: {lock['exit_kind']} exit at {lock['stop_at']}; "
                    f"sessions_elapsed={lock['gap_sessions']}; "
                    f"sessions_remaining={lock['remaining']}; "
                    f"lock_sessions={_COOLDOWN_SESSIONS}; "
                    f"eligible_from={lock['eligible_from']}"
                ),
                "debate_outcome": "BUY_REJECTED",
            })
        except Exception as exc:
            log.warning("re-entry lock signal_events write failed for %s: %s", ticker, exc)

    def _check_drawdown_halt(self) -> tuple[bool, str]:
        """Return (halted, reason). Reason is empty when not halted.

        Reads from portfolio_peak — the peak is maintained by the daily
        scheduler (`_fetch_session_account_balance`) which calls
        Database.update_portfolio_peak each session. Any DB error here
        is non-fatal: better to allow a trade than to block on a transient
        read error (kill switch covers true emergencies).
        """
        try:
            state = self._db.get_drawdown_state()
        except Exception as exc:
            log.warning("Drawdown state read failed (allowing trade): %s", exc)
            return False, ""
        if not state.get("halted"):
            return False, ""
        reason = state.get("halt_reason") or "Drawdown halt active"
        return True, f"Drawdown halt: {reason} — manual unlock required"

    def _count_violations_today(self, date_str: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM portfolio_violations "
                "WHERE date(created_at) = ?",
                (date_str,),
            ).fetchone()
            return row["n"] if row else 0


# ══════════════════════════════════════════════════════════════════════════════
# CLI / __main__
# ══════════════════════════════════════════════════════════════════════════════

def _progress_bar(used: int, total: int, width: int = 20) -> str:
    filled = int(width * used / total) if total > 0 else 0
    bar    = "▓" * filled + "░" * (width - filled)
    return f"[{bar}]"


def _status_icon(used: float, limit: float) -> str:
    if limit <= 0:
        return ""
    ratio = used / limit
    if ratio >= 1.0:
        return "  🔴 FULL"
    if ratio >= 0.80:
        return "  ⚠️  WARN"
    return ""


def _print_portfolio_state(pm: PortfolioManager) -> None:
    cap   = pm.capacity_summary()
    div   = pm.get_diversification_metrics()
    risk  = pm.check_risk_limits()
    rebal = pm.rebalance_if_needed()

    W = 66
    print(f"\n{'═' * W}")
    print(f"  Portfolio Manager  (balance: ${pm._balance:,.2f})")
    print(f"{'═' * W}")

    # ── Capacity ───────────────────────────────────────────────────────────────
    pos_used = cap["positions_used"]
    pos_max  = cap["positions_max"]
    dep_pct  = cap["deployed_pct"]
    dep_usd  = cap["deployed_usd"]
    dep_max  = cap["deploy_max_usd"]

    print(f"\n  CAPACITY")
    print(f"  {'Open Positions':<22} {pos_used:>2}/{pos_max}  "
          f"{_progress_bar(pos_used, pos_max)}"
          f"{_status_icon(pos_used, pos_max)}")
    print(f"  {'Capital Deployed':<22} {dep_pct:.1%}   "
          f"{_progress_bar(int(dep_pct * 100), 100)}"
          f"  ${dep_usd:,.0f} / ${dep_max:,.0f}"
          f"{_status_icon(dep_usd, dep_max)}")
    print(f"  {'Cash Reserve':<22} ${cap['cash_reserve']:,.2f}")

    # ── Strategy breakdown ─────────────────────────────────────────────────────
    print(f"\n  STRATEGY CAPACITY")
    for strat, info in cap["by_strategy"].items():
        used = info["used"]
        mx   = info["max"]
        lbl  = strat.replace("_", " ").title()
        icon = _status_icon(used, mx)
        print(f"  {lbl:<22} {used}/{mx}  {_progress_bar(used, mx, 10)}{icon}")

    # ── Sector breakdown ───────────────────────────────────────────────────────
    if cap["by_sector"]:
        print(f"\n  SECTOR BREAKDOWN")
        for sec, info in sorted(cap["by_sector"].items()):
            used = info["used"]
            mx   = info["max"]
            pct  = info["pct_of_portfolio"]
            icon = _status_icon(pct, pm.MAX_SECTOR_PCT)
            print(f"  {sec:<22} {used} pos  {pct:.0%} of portfolio{icon}")
    else:
        print(f"\n  SECTOR BREAKDOWN  (no open positions)")

    # ── Risk metrics ───────────────────────────────────────────────────────────
    print(f"\n  RISK METRICS")
    beta = risk["beta"]
    vol  = risk["volatility"]
    corr = risk["avg_correlation"]
    conc = risk["max_concentration"]

    print(f"  {'Portfolio Beta':<22} {f'{beta:.2f}' if beta is not None else '—'}")
    print(f"  {'Portfolio Volatility':<22} {f'{vol:.1%}' if vol is not None else '—'}")
    print(f"  {'Avg Correlation':<22} "
          f"{f'{corr:.2f}' if corr is not None else '—'}"
          f"{'  ⚠️  WARN' if corr is not None and corr > pm.WARN_CORRELATION else ''}")
    print(f"  {'Max Concentration':<22} "
          f"{f'{conc:.0%}' if conc is not None else '—'}"
          f"{'  ⚠️  WARN' if conc is not None and conc > pm.MAX_POSITION_PCT else ''}")

    # ── Positions ──────────────────────────────────────────────────────────────
    positions = div["positions"]
    if positions:
        total_val = div["total_value"]
        print(f"\n  OPEN POSITIONS")
        hdr = f"  {'Ticker':<8} {'Shares':>6} {'AvgPx':>8} {'Value':>10} " \
              f"{'Weight':>7} {'Strategy':<16} {'Sector'}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for p in sorted(positions, key=lambda x: -x["current_value"]):
            weight = p["current_value"] / total_val if total_val > 0 else 0.0
            print(
                f"  {p['ticker']:<8} {p['shares']:>6} "
                f"${p['avg_price']:>7.2f} "
                f"${p['current_value']:>9.2f} "
                f"  {weight:>5.1%}  "
                f"{p['strategy']:<16} {p['sector']}"
            )
    else:
        print(f"\n  OPEN POSITIONS  (none)")

    # ── Rebalancing alerts ─────────────────────────────────────────────────────
    if rebal:
        print(f"\n  REBALANCING ALERTS")
        for act in rebal:
            print(f"  ⚠️  {act['reason']}")

    # ── Warnings ──────────────────────────────────────────────────────────────
    extra_warnings = [w for w in cap["warnings"] if w not in [a["reason"] for a in rebal]]
    if extra_warnings:
        print(f"\n  WARNINGS")
        for w in extra_warnings:
            print(f"  ⚠️  {w}")

    print(f"\n{'═' * W}\n")


def main() -> None:
    import argparse
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    parser = argparse.ArgumentParser(
        description="Portfolio Manager — show current state, risk metrics, and capacity",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10_000.0,
        metavar="USD",
        help="Account balance for deployment % calculations (default: 10000).",
    )
    args = parser.parse_args()

    pm = PortfolioManager(account_balance=args.balance)
    _print_portfolio_state(pm)


if __name__ == "__main__":
    main()
