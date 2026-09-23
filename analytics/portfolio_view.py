"""Read-only portfolio view — the one place API, MCP (and anything else
that shows "the portfolio") compute NAV, cash and daily P&L from.

Why this exists (2026-09-23)
----------------------------
``get_portfolio`` (MCP) and ``/api/portfolio`` showed identical numbers on
2026-09-19 and 2026-09-23: "Total value" was the sum of
``portfolio_positions.current_value`` (positions only, no cash), ``now``
equalled ``entry`` for days, "Daily P&L" was the realised P&L of today's
SELLs (0.00 on a day without an exit).  Root cause: every broker sync
(``IBKRTrader.get_portfolio`` — session start, every PositionManager
cycle) overwrote ``current_value`` with shares × avg_price because IBKR's
position list carries no market price, wiping the PositionManager's live
mark; outside RTH nothing re-marked, so the view read entry values.
"cash" was NetLiquidation (last ``risk_calculations`` row) minus cost
basis — not cash.

Definitions used here
---------------------
* positions_value = Σ shares × mark_price, where mark_price is the last
  mark written by the PositionManager (same yfinance 1-minute feed it
  trails stops on) or the last fill; falls back to avg_price for a row
  that was never marked (flagged in ``mark_source``).
* cash = TotalCashValue from the newest ``account_snapshots`` row (broker,
  written by the daemon / PositionManager).  Without any snapshot yet the
  view falls back to the old estimate (NetLiquidation − cost basis) and
  says so in ``cash_source``.
* value (NAV) = cash + positions_value.
* daily_pnl = NAV − previous close NAV.  Previous close = the broker's
  PreviousDayEquityWithLoanValue from the newest snapshot; if that tag is
  missing, the NetLiquidation of the last ``kind='eod'`` snapshot before
  today.  Neither available → ``daily_pnl`` is None and
  ``daily_pnl_basis`` says why (no invented zero).
* realized_today = Σ trade_history.pnl for today (kept, separately).
* broker_nav = NetLiquidation from the newest snapshot, with its
  timestamp, so the view can be reconciled against IBKR at a glance
  (``nav_minus_broker``).

Stdlib + sqlite3 only: ``api/main.py`` and ``mcp_server/nts_mcp.py`` must
not import the trading stack.  Tolerates DBs that predate the mark
columns / the snapshots table (older backups, test fixtures).
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from typing import Any


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    try:
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    except sqlite3.Error:
        return []


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _r(v: float | None, nd: int = 2) -> float | None:
    if v is None:
        return None
    out = round(v, nd)
    return 0.0 if out == 0 else out      # never "-0.00"


def build_portfolio_view(conn: sqlite3.Connection, *, now: datetime | None = None) -> dict:
    """Assemble the portfolio dict (see module docstring for definitions).

    `conn` may be any sqlite3 connection (read-only is fine); row_factory
    does not matter.
    """
    now = now or datetime.now(timezone.utc)
    today_iso = now.date().isoformat()
    notes: list[str] = []

    # ── positions ────────────────────────────────────────────────────────
    pcols = _columns(conn, "portfolio_positions")
    has_marks = {"mark_price", "mark_source", "marked_at"} <= pcols
    wanted = ["ticker", "shares", "avg_price", "current_value", "updated_at",
              "mark_price", "mark_source", "marked_at"]
    sel = ", ".join(c for c in wanted if c in pcols)   # tolerate minimal fixtures
    raw = _rows(conn, f"SELECT {sel} FROM portfolio_positions ORDER BY ticker") if sel else []

    positions: list[dict] = []
    positions_value = 0.0
    cost_basis = 0.0
    marks_as_of: str | None = None
    unmarked: list[str] = []
    for p in raw:
        shares = int(p.get("shares") or 0)
        if shares <= 0:
            continue
        avg = _f(p.get("avg_price")) or 0.0
        mark = _f(p.get("mark_price")) if has_marks else None
        source = p.get("mark_source") if has_marks else None
        marked_at = p.get("marked_at") if has_marks else None
        if not mark or mark <= 0:
            # Never marked (or pre-migration row): the only honest price we
            # have is the cost basis.  current_value may still carry a live
            # value written before the migration — prefer it when present.
            cv = _f(p.get("current_value")) or 0.0
            mark = (cv / shares) if cv > 0 else avg
            source = "unmarked:current_value" if cv > 0 else "unmarked:avg_price"
            marked_at = p.get("updated_at")
            unmarked.append(p["ticker"])
        mv = shares * mark
        cost = shares * avg
        positions_value += mv
        cost_basis += cost
        if marked_at and (marks_as_of is None or marked_at > marks_as_of):
            marks_as_of = marked_at
        positions.append({
            "ticker": p["ticker"],
            "shares": shares,
            "entry": round(avg, 2),
            "current": round(mark, 2),
            "market_value": round(mv, 2),
            "unrealized_pnl": round(mv - cost, 2),
            "pnl_pct": round(((mv - cost) / cost * 100) if cost else 0.0, 1),
            "mark_source": source,
            "marked_at": marked_at,
        })
    if unmarked:
        notes.append(f"no live mark yet for {', '.join(unmarked)} — shown at cost/last value")

    # ── broker account snapshot ──────────────────────────────────────────
    snap = None
    if "account_snapshots" in {
        r["name"] for r in _rows(conn, "SELECT name FROM sqlite_master WHERE type='table'")
    }:
        snaps = _rows(conn, "SELECT * FROM account_snapshots ORDER BY ts DESC, id DESC LIMIT 1")
        snap = snaps[0] if snaps else None

    cash: float | None = None
    cash_source = None
    cash_as_of = None
    broker_nav = None
    broker_nav_as_of = None
    prev_close_nav = None
    prev_close_source = None
    if snap:
        broker_nav = _f(snap.get("net_liquidation"))
        broker_nav_as_of = snap.get("ts")
        cash = _f(snap.get("total_cash"))
        cash_as_of = snap.get("ts")
        cash_source = f"{snap.get('source') or 'broker'}:TotalCashValue"
        pde = _f(snap.get("prev_day_equity"))
        if pde and pde > 0:
            prev_close_nav = pde
            prev_close_source = f"{snap.get('source') or 'broker'}:PreviousDayEquityWithLoanValue"
        if prev_close_nav is None:
            eod = _rows(
                conn,
                "SELECT net_liquidation, ts FROM account_snapshots "
                "WHERE kind = 'eod' AND substr(ts, 1, 10) < ? "
                "ORDER BY ts DESC, id DESC LIMIT 1",
                (today_iso,),
            )
            if eod and _f(eod[0].get("net_liquidation")):
                prev_close_nav = _f(eod[0]["net_liquidation"])
                prev_close_source = f"eod snapshot {str(eod[0]['ts'])[:16]}"
    if cash is None:
        # Pre-snapshot fallback: the old estimate, labelled as such.
        rcols = _columns(conn, "risk_calculations")
        acct = []
        if "account_balance" in rcols:
            sel = "account_balance" + (", created_at" if "created_at" in rcols else "")
            order = "id" if "id" in rcols else "rowid"
            acct = _rows(conn, f"SELECT {sel} FROM risk_calculations ORDER BY {order} DESC LIMIT 1")
        bal = _f(acct[0].get("account_balance")) if acct else None
        if bal is not None:
            cash = bal - cost_basis
            cash_source = "estimate:risk_calculations.account_balance − cost basis"
            cash_as_of = acct[0].get("created_at")
            notes.append("cash is an estimate (no broker account snapshot yet)")
        else:
            cash = 0.0
            cash_source = "unavailable"
            notes.append("cash unavailable (no account snapshot, no risk_calculations row)")

    nav = cash + positions_value

    # ── P&L ──────────────────────────────────────────────────────────────
    rt = _rows(conn, "SELECT COALESCE(SUM(pnl), 0) AS pnl FROM trade_history "
                     "WHERE substr(created_at, 1, 10) = ?", (today_iso,))
    realized_today = _f(rt[0].get("pnl")) if rt else 0.0
    realized_today = realized_today or 0.0

    if prev_close_nav is not None and prev_close_nav > 0:
        daily_pnl = nav - prev_close_nav
        daily_pnl_pct = daily_pnl / prev_close_nav * 100
        daily_pnl_basis = f"NAV − previous close ({prev_close_source})"
    else:
        daily_pnl = None
        daily_pnl_pct = None
        daily_pnl_basis = "unavailable: no previous-close equity (needs a broker account snapshot)"
        notes.append("daily P&L unavailable until the daemon has recorded an account snapshot")

    return {
        # Legacy keys (dashboard SummaryBar, MCP formatter) — `value` is
        # now NAV = cash + positions, not positions alone.
        "value": _r(nav),
        "cash": _r(cash),
        "daily_pnl": _r(daily_pnl),
        "daily_pnl_pct": _r(daily_pnl_pct),
        "positions": positions,
        # Explicit parts
        "nav": _r(nav),
        "positions_value": _r(positions_value),
        "cost_basis": _r(cost_basis),
        "unrealized_pnl": _r(positions_value - cost_basis),
        "realized_today": _r(realized_today),
        "prev_close_nav": _r(prev_close_nav),
        "daily_pnl_basis": daily_pnl_basis,
        "cash_source": cash_source,
        "cash_as_of": cash_as_of,
        "marks_as_of": marks_as_of,
        # Reconciliation against the broker
        "broker_nav": _r(broker_nav),
        "broker_nav_as_of": broker_nav_as_of,
        "nav_minus_broker": _r(nav - broker_nav) if broker_nav else None,
        "notes": notes,
        "as_of": now.isoformat(timespec="seconds"),
    }


def open_readonly(db_path: str) -> sqlite3.Connection:
    """Read-only sqlite3 connection (URI mode) with a short busy timeout."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)


__all__ = ["build_portfolio_view", "open_readonly"]
