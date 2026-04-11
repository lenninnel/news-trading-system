"""
PostSessionReviewer — post-session Haiku summariser for the trading system.

Runs after EOD (always) and after US_OPEN (only when at least one trade
was executed). Gathers the day's state — session signals, open
positions, today's executed trades, sector concentration, stale holds,
low-conviction trades — hands it to Claude Haiku, and returns a
200-word fixed-format review block ready for Telegram.

Purely a data-fetching + LLM call helper. It does NOT send the
Telegram message itself and does NOT write to signal_events; the
scheduler handles both of those so the agent stays pure and testable.
Failure mode is always the same: log warning, return "", move on.

Mirrors the shape of ``agents/macro_context_agent.py`` (flag gating,
Haiku model, SDK-level timeout, fire-and-forget) so the two can be
updated together.

Environment
-----------
    ENABLE_POST_SESSION_REVIEW   "true" to enable. Default off.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any

import anthropic

from config.settings import ANTHROPIC_API_KEY

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

# Same Haiku model as MacroContextAgent — cheap and fast.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# SDK-level HTTP timeout. Longer than macro context (10s) because the
# prompt is heavier — more signals, more positions to format.
TIMEOUT_SECONDS = 15.0

# Total schedule slots per weekday. Used in the "Sessions: X/Y" line.
TOTAL_SESSIONS = 7

MAX_TOKENS = 400

# Sessions that trigger a review. EOD is the primary case; US_OPEN is
# opt-in from the caller (only when a trade was actually executed).
REVIEWABLE_SESSIONS: frozenset[str] = frozenset({"EOD", "US_OPEN"})

# Flag thresholds — tuned once here so they're discoverable.
STALE_POSITION_DAYS = 15
LOW_CONVICTION_THRESHOLD = 0.30
CONCENTRATION_MIN_POSITIONS = 2


_SYSTEM_PROMPT = (
    "You are a trading system reviewer. You analyse one trading session's "
    "output and flag anything worth noting in a fixed format. Be factual and "
    "terse. No opinions, no disclaimers, no markdown other than the emojis "
    "already in the template. Maximum 200 words total.\n\n"
    "Return ONLY this structure, filling in the placeholders. If a flag has "
    "no matches, write 'None' on the FLAGS line for that category.\n\n"
    "📊 EOD REVIEW — {date}\n\n"
    "🔄 Sessions: {sessions_done}/{total_sessions} completed\n"
    "💼 Positions: {position_count} open | P&L: {daily_pnl_str}\n"
    "📈 Trades: {trade_count} executed today\n\n"
    "SIGNALS: <1 sentence summary of what fired, naming the top 1-2 tickers>\n\n"
    "⚠️ FLAGS:\n"
    "- Concentration: <list sector buckets with 2+ positions, or 'None'>\n"
    "- Stale position: <list tickers held >15 days, or 'None'>\n"
    "- Low conviction: <list trades executed with conf <0.30, or 'None'>\n\n"
    "TOMORROW: <1 sentence on what to watch — major catalysts, sector moves>"
)


def is_enabled() -> bool:
    """Read ``ENABLE_POST_SESSION_REVIEW`` at call time (not module load)."""
    return os.environ.get("ENABLE_POST_SESSION_REVIEW", "").strip().lower() in (
        "true", "1", "yes",
    )


# ── State gathering ──────────────────────────────────────────────────────


def _gather_state(
    session: str,
    tickers: list[str],
    signals: list[dict],
) -> dict:
    """Pull today's context out of the DB.

    Returns a dict with: positions, today_trades, concentrated, stale,
    low_conviction, sessions_done — everything the prompt builder needs.
    Never raises; empty lists / dicts on any DB error.
    """
    from storage.database import Database

    # Read DB_PATH at call time — storage.database._resolve_db_path binds its
    # default from config.settings.DB_PATH at import, so a monkeypatched env
    # var (in tests, or a late-set .env on Hetzner) would otherwise be
    # ignored. Passing the path explicitly to Database() bypasses the
    # import-time capture.
    db_path = os.environ.get("DB_PATH", "").strip() or None
    try:
        db = Database(db_path=db_path) if db_path else Database()
    except Exception as exc:
        log.warning("PostSessionReviewer: DB init failed: %s", exc)
        return _empty_state()

    # 1. Open positions with computed pnl_pct and days_held
    positions = _fetch_positions_with_age(db)

    # 2. Today's executed trades
    today_trades = _fetch_today_trades(db)

    # 3. Concentration buckets by sector
    concentrated = _compute_concentration(db, positions)

    # 4. Stale positions (held > threshold days)
    stale = [p for p in positions if (p.get("days_held") or 0) > STALE_POSITION_DAYS]

    # 5. Low-conviction executed trades from the signal list
    low_conviction = [
        s for s in signals
        if s.get("trade_executed") and (s.get("confidence") or 0) < LOW_CONVICTION_THRESHOLD
    ]

    # 6. Sessions completed today (distinct session names in signal_events)
    sessions_done = _sessions_completed_today(db)

    # 7. Daily PnL total from today's trades
    daily_pnl = sum(float(t.get("pnl") or 0) for t in today_trades)

    return {
        "positions": positions,
        "today_trades": today_trades,
        "concentrated": concentrated,
        "stale": stale,
        "low_conviction": low_conviction,
        "sessions_done": sessions_done,
        "daily_pnl": daily_pnl,
    }


def _empty_state() -> dict:
    return {
        "positions": [],
        "today_trades": [],
        "concentrated": {},
        "stale": [],
        "low_conviction": [],
        "sessions_done": 0,
        "daily_pnl": 0.0,
    }


def _fetch_positions_with_age(db) -> list[dict]:
    """Return open positions with pnl_pct and days_held computed.

    days_held is MIN(created_at) from trade_history BUY rows for each
    ticker — the first time we opened the position. One batched query
    rather than one per position.
    """
    try:
        with db._connect() as conn:
            pos_rows = conn.execute(
                "SELECT ticker, shares, avg_price, current_value "
                "FROM portfolio_positions WHERE shares > 0 ORDER BY ticker"
            ).fetchall()

            # Batch fetch the earliest BUY per held ticker
            age_rows = conn.execute(
                "SELECT ticker, MIN(created_at) AS first_buy "
                "FROM trade_history WHERE action = 'BUY' GROUP BY ticker"
            ).fetchall()
    except Exception as exc:
        log.warning("PostSessionReviewer: position query failed: %s", exc)
        return []

    first_buy: dict[str, str] = {
        r["ticker"]: r["first_buy"] for r in age_rows if r["first_buy"]
    }

    today = datetime.now(timezone.utc).date()
    out: list[dict] = []
    for r in pos_rows:
        shares = r["shares"] or 0
        avg = float(r["avg_price"] or 0)
        cur_val = float(r["current_value"] or 0)
        cost = avg * shares
        pnl_pct = ((cur_val - cost) / cost * 100) if cost else 0.0

        days_held: int | None = None
        buy_ts = first_buy.get(r["ticker"])
        if buy_ts:
            try:
                opened = datetime.fromisoformat(buy_ts.replace("Z", "+00:00")).date()
                days_held = (today - opened).days
            except ValueError:
                days_held = None

        out.append({
            "ticker": r["ticker"],
            "shares": shares,
            "entry": round(avg, 2),
            "current": round(cur_val / shares, 2) if shares else 0.0,
            "pnl_pct": round(pnl_pct, 1),
            "days_held": days_held,
        })
    return out


def _fetch_today_trades(db) -> list[dict]:
    """Return trade_history rows whose date part matches today's UTC date."""
    try:
        today_iso = date.today().isoformat()
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT ticker, action, shares, price, pnl, created_at "
                "FROM trade_history WHERE date(created_at) = date(?) "
                "ORDER BY created_at",
                (today_iso,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.warning("PostSessionReviewer: today's trades query failed: %s", exc)
        return []


def _compute_concentration(db, positions: list[dict]) -> dict[str, list[str]]:
    """Group current positions by sector; return buckets with 2+ tickers.

    Uses the module-level ``storage.database.get_sector`` which reads
    config/sector_map.json. Tickers with no sector entry are excluded —
    we only flag known-sector overlap.

    ``db`` is unused here but kept in the signature so tests can swap in
    a stub without refactoring the call site.
    """
    from storage.database import get_sector

    bucket: dict[str, list[str]] = defaultdict(list)
    for p in positions:
        try:
            sector = get_sector(p["ticker"]) or ""
        except Exception:
            sector = ""
        if sector:
            bucket[sector].append(p["ticker"])

    return {
        sector: tickers
        for sector, tickers in bucket.items()
        if len(tickers) >= CONCENTRATION_MIN_POSITIONS
    }


def _sessions_completed_today(db) -> int:
    """Count distinct session names in signal_events for today (UTC)."""
    try:
        today_iso = date.today().isoformat()
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT session FROM signal_events "
                "WHERE date(timestamp) = date(?) AND session IS NOT NULL",
                (today_iso,),
            ).fetchall()
        return len(rows)
    except Exception:
        return 0


# ── Prompt construction ──────────────────────────────────────────────────


def _build_user_prompt(session: str, tickers: list[str], signals: list[dict],
                       state: dict) -> str:
    """Pack the gathered state into a Claude user message.

    The system prompt already defines the output format — this just
    hands over the raw facts.
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Signals summary (top 8, non-HOLD preferred)
    ranked = sorted(
        signals,
        key=lambda s: (
            0 if (s.get("signal") or "").upper() not in ("HOLD", "CONFLICTING") else 1,
            -(s.get("confidence") or 0),
        ),
    )[:8]
    signal_lines = [
        f"  {s.get('ticker', '?'):<6} {s.get('signal', '?'):<12} "
        f"conf={s.get('confidence') or 0:.2f} "
        f"debate={s.get('debate_outcome') or '-'}"
        for s in ranked
    ] or ["  (no signals)"]

    # Positions block
    pos_lines = [
        f"  {p['ticker']:<6} entry=${p['entry']:.2f} "
        f"now=${p['current']:.2f} pnl={p['pnl_pct']:+.1f}% "
        f"days_held={p.get('days_held', '?')}"
        for p in state["positions"]
    ] or ["  (no open positions)"]

    # Trades block
    trade_lines = [
        f"  {t.get('ticker', '?'):<6} {t.get('action', '?'):<4} "
        f"{t.get('shares') or 0} @ ${float(t.get('price') or 0):.2f} "
        f"pnl=${float(t.get('pnl') or 0):+.2f}"
        for t in state["today_trades"]
    ] or ["  (none)"]

    # Concentration block
    if state["concentrated"]:
        conc_lines = [
            f"  {sector}: {', '.join(ticks)}"
            for sector, ticks in sorted(state["concentrated"].items())
        ]
    else:
        conc_lines = ["  (no sector with 2+ positions)"]

    # Stale / low-conviction blocks — pre-computed so the model can't
    # miss them.
    stale_str = (
        ", ".join(
            f"{p['ticker']} ({p['days_held']}d)" for p in state["stale"]
        ) or "None"
    )
    low_conv_str = (
        ", ".join(
            f"{s.get('ticker', '?')} ({(s.get('confidence') or 0):.2f})"
            for s in state["low_conviction"]
        ) or "None"
    )

    pnl = state["daily_pnl"]
    daily_pnl_str = f"${pnl:+,.2f}"
    position_count = len(state["positions"])
    trade_count = len(state["today_trades"])
    sessions_done = state["sessions_done"]

    # Top of prompt: fill the template vars for the system prompt's
    # "{date}" / "{sessions_done}" placeholders. Claude will then
    # replicate them literally into the output block.
    return (
        f"SESSION: {session} ({today_str})\n"
        f"\n"
        f"Fill these template vars into the header:\n"
        f"  date={today_str}\n"
        f"  sessions_done={sessions_done}\n"
        f"  total_sessions={TOTAL_SESSIONS}\n"
        f"  position_count={position_count}\n"
        f"  daily_pnl_str={daily_pnl_str}\n"
        f"  trade_count={trade_count}\n"
        f"\n"
        f"SIGNALS GENERATED ({len(signals)} total, showing top {len(ranked)}):\n"
        + "\n".join(signal_lines)
        + f"\n\nOPEN POSITIONS ({position_count}):\n"
        + "\n".join(pos_lines)
        + f"\n\nTRADES TODAY ({trade_count}):\n"
        + "\n".join(trade_lines)
        + f"\n\nCORRELATION CHECK:\n"
        + "\n".join(conc_lines)
        + f"\n\nSTALE POSITIONS (>{STALE_POSITION_DAYS}d held): {stale_str}\n"
        + f"LOW CONVICTION TRADES (conf <{LOW_CONVICTION_THRESHOLD:.2f}): {low_conv_str}\n"
    )


# ── Reviewer class ───────────────────────────────────────────────────────


class PostSessionReviewer:
    """End-of-session Claude Haiku summariser.

    Parameters
    ----------
    client:
        Optional pre-built ``anthropic.Anthropic`` instance. Tests pass a
        mock; production constructs one lazily.
    model:
        Override the Haiku model ID. Defaults to ``DEFAULT_MODEL``.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client
        self._model = model or DEFAULT_MODEL

    async def review(
        self,
        session: str,
        tickers: list[str],
        signals: list[dict],
    ) -> str:
        """Review session output and return the formatted Telegram message.

        Returns empty string when:
            * ENABLE_POST_SESSION_REVIEW is unset/false
            * session is not in REVIEWABLE_SESSIONS (only EOD / US_OPEN)
            * Claude call fails for any reason (timeout, API error, parse)

        The scheduler is responsible for actually *sending* the message
        and *logging* it to signal_events — this method is pure.
        """
        if not is_enabled():
            return ""
        if session not in REVIEWABLE_SESSIONS:
            log.debug("PostSessionReviewer: skipping non-reviewable session %s", session)
            return ""

        try:
            state = await asyncio.to_thread(
                _gather_state, session, tickers, signals,
            )
            return await asyncio.to_thread(self._call, session, tickers, signals, state)
        except Exception as exc:
            log.warning("PostSessionReviewer: failed (%s) — session=%s", exc, session)
            return ""

    # ── Private ──────────────────────────────────────────────────────────

    def _ensure_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._client

    def _call(self, session: str, tickers: list[str],
              signals: list[dict], state: dict) -> str:
        """Blocking Claude call. Invoked from ``asyncio.to_thread``."""
        client = self._ensure_client()
        user_msg = _build_user_prompt(session, tickers, signals, state)

        msg = client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            timeout=TIMEOUT_SECONDS,
        )
        return self._extract_text(msg)

    @staticmethod
    def _extract_text(msg: object) -> str:
        """Pull concatenated text blocks from a Claude response."""
        content = getattr(msg, "content", None) or []
        parts: list[str] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", "") or ""
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
