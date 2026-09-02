"""
Telegram notification helper for the News Trading System.

Uses the Telegram Bot HTTP API directly via ``requests`` (already a project
dependency), so no extra packages or asyncio are needed.

Formatting contract (single choke point: ``_send``)
---------------------------------------------------
Every message goes out with ``parse_mode="HTML"``. Callers may pass either

* **legacy Markdown-styled text** (the default): ``*bold*`` and
  ```code``` spans are converted to ``<b>``/``<code>``; everything else
  — including underscores, stray asterisks, unbalanced backticks and any
  ``<``, ``>``, ``&`` inside interpolated error strings — is HTML-escaped
  and delivered verbatim. The converter only ever emits balanced tags, so
  the message can never trip Telegram's entity parser (HTTP 400), no
  matter what a caller interpolates.
* **pre-built HTML** (``html=True``): used by the typed ``send_*`` methods,
  which escape their dynamic fields with :func:`escape`.

Messages longer than Telegram's 4096-character limit are sent as plain
text, truncated. A plain-text retry without ``parse_mode`` remains as a
safety net for anything unforeseen.

Quick setup
-----------
1. Create a bot with BotFather: https://t.me/BotFather  → get BOT_TOKEN
2. Send any message to your bot, then visit:
       https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
   and copy the ``chat.id`` value.
3. Export two environment variables (or set them in .env):
       export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
       export TELEGRAM_CHAT_ID="-1001234567890"
4. Optionally add a telegram section to config/watchlist.yaml (only used
   by :meth:`TelegramNotifier.from_config`; the daemon and the ops
   scripts use :meth:`TelegramNotifier.from_env`, which needs the two
   environment variables only).
"""

from __future__ import annotations

import html as _html
import json
import logging
import os
import re

import requests

log = logging.getLogger(__name__)

# Telegram Bot API base URL template
_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# Telegram's hard limit for a single text message.
MAX_MESSAGE_LEN = 4096

# Signal → emoji badge
_SIGNAL_EMOJI: dict[str, str] = {
    "STRONG BUY":  "🚀",
    "WEAK BUY":    "📈",
    "STRONG SELL": "🔻",
    "WEAK SELL":   "📉",
    "CONFLICTING": "⚠️",
    "HOLD":        "⏸",
}

# Status → emoji
_STATUS_EMOJI: dict[str, str] = {
    "success": "✅",
    "partial":  "⚠️",
    "failed":   "❌",
}

_TAG_RE = re.compile(r"</?(?:b|i|u|s|code|pre|a)(?:\s[^>]*)?>")


def _expand_env(value: str) -> str:
    """Replace ``${VAR}`` and ``$VAR`` placeholders with environment values."""
    return re.sub(
        r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
        lambda m: os.environ.get(m.group(1) or m.group(2), m.group(0)),
        str(value),
    )


def escape(text: object) -> str:
    """HTML-escape a dynamic value for inclusion in a Telegram HTML message."""
    return _html.escape(str(text), quote=False)


def legacy_markdown_to_html(text: str) -> str:
    """Convert legacy ``*bold*`` / ```code``` styling to Telegram HTML.

    Rules (deliberately narrow so data can never be mistaken for markup):

    * ``*...*`` → ``<b>...</b>`` and ```...``` → ``<code>...</code>``, but
      only when the closing marker sits on the same line and the span is
      non-empty. Anything else stays a literal character.
    * Inside a bold span backticks are literal; inside a code span
      asterisks are literal (Telegram forbids nesting code in bold).
    * Underscores are never interpreted — error strings such as
      ``daily_ohlc`` or ``news_feed.py`` pass through untouched.
    * ``<``, ``>`` and ``&`` are escaped everywhere.

    The output always has balanced tags, so Telegram's HTML parser
    accepts it for any input string.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ("*", "`"):
            close = text.find(ch, i + 1)
            newline = text.find("\n", i + 1)
            if close > i + 1 and (newline == -1 or close < newline):
                inner = text[i + 1:close]
                tag = "b" if ch == "*" else "code"
                out.append(f"<{tag}>{_html.escape(inner, quote=False)}</{tag}>")
                i = close + 1
                continue
        out.append(_html.escape(ch, quote=False))
        i += 1
    return "".join(out)


def strip_html(text: str) -> str:
    """Reduce a Telegram HTML message to plain text (tags removed, entities unescaped)."""
    return _html.unescape(_TAG_RE.sub("", text))


class TelegramNotifier:
    """
    Sends formatted Telegram messages for trading signals, trades, and summaries.

    All public ``send_*`` methods swallow exceptions so a Telegram outage can
    never interrupt the trading pipeline.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        dashboard_url: str = "",
    ) -> None:
        self._token        = bot_token
        self._chat_id      = chat_id
        self._dashboard_url = dashboard_url.rstrip("/")
        self._url          = _API_URL.format(token=bot_token)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, cfg: dict | None = None) -> "TelegramNotifier | None":
        """
        Build a notifier from ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``.

        This is the production path: the daemon, the kill switch, the
        drawdown CLI and the ops scripts all run with the two variables
        in their environment (``.env`` on the VPS). The YAML ``telegram``
        section — including its ``enabled`` flag — is consulted only as a
        fallback when the variables are absent and ``cfg`` is given.
        """
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if token and chat_id:
            dashboard_url = ""
            if cfg:
                dashboard_url = _expand_env((cfg.get("telegram") or {}).get("dashboard_url", ""))
            return cls(bot_token=token, chat_id=chat_id, dashboard_url=dashboard_url)
        if cfg:
            return cls.from_config(cfg)
        log.warning(
            "Telegram disabled — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set"
        )
        return None

    @classmethod
    def from_config(cls, cfg: dict) -> "TelegramNotifier | None":
        """
        Build a TelegramNotifier from the YAML config dict.

        Returns *None* if Telegram is disabled or credentials are missing.
        Environment variables (``TELEGRAM_BOT_TOKEN``, ``TELEGRAM_CHAT_ID``)
        always take precedence over YAML values.

        Args:
            cfg: Full config dict as returned by ``_load_config()``.

        Returns:
            TelegramNotifier instance, or None.
        """
        tg_cfg = cfg.get("telegram", {})
        if not tg_cfg.get("enabled", False):
            return None

        # Env vars take precedence over YAML literals / placeholders
        token = (
            os.environ.get("TELEGRAM_BOT_TOKEN")
            or _expand_env(tg_cfg.get("bot_token", ""))
        )
        chat_id = (
            os.environ.get("TELEGRAM_CHAT_ID")
            or _expand_env(str(tg_cfg.get("chat_id", "")))
        )
        dashboard_url = _expand_env(tg_cfg.get("dashboard_url", ""))

        if not token or not chat_id:
            log.warning(
                "Telegram enabled but bot_token/chat_id missing. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars."
            )
            return None

        return cls(bot_token=token, chat_id=chat_id, dashboard_url=dashboard_url)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _keyboard(self) -> dict | None:
        """Return an inline keyboard with a dashboard link, or None."""
        if not self._dashboard_url:
            return None
        return {
            "inline_keyboard": [[
                {"text": "📊 View Dashboard", "url": self._dashboard_url}
            ]]
        }

    def _send(
        self,
        text: str,
        *,
        reply_markup: dict | None = None,
        html: bool = False,
    ) -> bool:
        """
        POST a message to Telegram. Returns True on success, False on failure.
        Never raises.

        ``text`` is legacy Markdown-styled text unless ``html=True`` (see
        module docstring). Either way the payload that leaves this method
        is well-formed Telegram HTML, capped at :data:`MAX_MESSAGE_LEN`.
        """
        raw = text if isinstance(text, str) else str(text)
        body = raw if html else legacy_markdown_to_html(raw)

        payload: dict = {"chat_id": self._chat_id}
        if len(body) > MAX_MESSAGE_LEN:
            # Too long for one message: drop formatting rather than risk
            # cutting through a tag, and truncate to the hard limit.
            plain = strip_html(body)
            payload["text"] = plain[:MAX_MESSAGE_LEN - 1] + "…"
            log.warning(
                "Telegram message truncated from %d to %d chars",
                len(plain), MAX_MESSAGE_LEN,
            )
        else:
            payload["text"] = body
            payload["parse_mode"] = "HTML"

        if reply_markup is None:
            reply_markup = self._keyboard()
        if reply_markup:
            payload["reply_markup"] = json.dumps(reply_markup)

        try:
            resp = requests.post(self._url, json=payload, timeout=10)
            if resp.ok:
                return True
            log.warning(
                "Telegram API error %d: %s",
                resp.status_code,
                resp.text[:200],
            )
            # Safety net: retry once as plain text (no parse_mode). With
            # the HTML contract above this should never be needed for a
            # formatting reason — if it fires, the log line is the signal.
            if "parse_mode" in payload:
                retry_payload = {
                    k: v for k, v in payload.items() if k != "parse_mode"
                }
                retry_payload["text"] = strip_html(payload["text"])[:MAX_MESSAGE_LEN]
                retry = requests.post(self._url, json=retry_payload, timeout=10)
                if retry.ok:
                    return True
                log.warning(
                    "Telegram plain-text retry failed %d: %s",
                    retry.status_code,
                    retry.text[:200],
                )
            return False
        except Exception as exc:
            log.warning("Could not send Telegram message: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public send methods
    # ------------------------------------------------------------------

    def send_message(self, message: str) -> bool:
        """
        Send an arbitrary text message verbatim (no markup interpretation).

        Used by the kill switch, the drawdown CLI and the health monitor,
        whose messages are plain multi-line text.
        """
        return self._send(escape(message[:MAX_MESSAGE_LEN]), html=True)

    def send_signal(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        reasoning: str = "",
        debate_summary: str = "",
    ) -> None:
        """
        Send a trading signal alert.

        Args:
            ticker:         Stock symbol, e.g. "AAPL".
            signal:         Combined signal string, e.g. "STRONG BUY".
            confidence:     Confidence as a percentage (0–100).
            reasoning:      Optional short reasoning text.
            debate_summary: Optional bull/bear debate summary.
        """
        emoji = _SIGNAL_EMOJI.get(signal, "📌")
        lines = [
            f"{emoji} <b>{escape(ticker)}</b> — <code>{escape(signal)}</code>",
            f"Confidence: <b>{confidence:.0f}%</b>",
        ]
        if reasoning:
            # Truncate long reasoning to keep message tidy
            short = reasoning[:300] + ("…" if len(reasoning) > 300 else "")
            lines.append(f"<i>{escape(short)}</i>")
        if debate_summary:
            short = debate_summary[:300] + ("…" if len(debate_summary) > 300 else "")
            lines.append(f"🐂🐻 <i>{escape(short)}</i>")

        self._send("\n".join(lines), html=True)

    def send_trade_executed(
        self,
        ticker: str,
        action: str,
        shares: float,
        price: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        """
        Send a paper-trade execution notification.

        Args:
            ticker:      Stock symbol.
            action:      "BUY" or "SELL".
            shares:      Number of shares.
            price:       Execution price per share.
            stop_loss:   Stop-loss price.
            take_profit: Take-profit price.
        """
        direction_emoji = "🟢" if action == "BUY" else "🔴"
        position_value  = shares * price
        lines = [
            f"{direction_emoji} <b>Paper Trade Executed</b> — <code>{escape(ticker)}</code>",
            f"Action:      <b>{escape(action)}</b>",
            f"Shares:      {shares}",
            f"Price:       ${price:,.2f}",
            f"Value:       ${position_value:,.2f}",
            f"Stop-loss:   ${stop_loss:,.2f}",
            f"Take-profit: ${take_profit:,.2f}",
        ]
        self._send("\n".join(lines), html=True)

    def send_daily_summary(
        self,
        signals_count: int,
        trades_count: int,
        portfolio_value: float,
        results: list[dict],
        errors: list[str],
        status: str,
    ) -> None:
        """
        Send an end-of-day summary message.

        Args:
            signals_count:   Total signals generated.
            trades_count:    Total trades executed.
            portfolio_value: Current total portfolio value in USD.
            results:         List of per-ticker dicts with keys:
                             ticker, signal, conf, traded, trade_id.
            errors:          List of error strings.
            status:          "success" | "partial" | "failed".
        """
        status_emoji = _STATUS_EMOJI.get(status, "📋")
        lines = [
            f"{status_emoji} <b>Daily Trading Summary</b> — <code>{escape(status.upper())}</code>",
            "",
            f"Signals generated : {signals_count}",
            f"Trades executed   : {trades_count}",
            f"Portfolio value   : ${portfolio_value:,.2f}",
        ]

        if results:
            lines.append("")
            lines.append("<b>Signal breakdown:</b>")
            for r in results:
                sig_emoji = _SIGNAL_EMOJI.get(r.get("signal", ""), "•")
                conf_pct  = f"{r['conf']:.0%}"
                traded    = "→ trade" if r.get("traded") else ""
                ticker_col = escape(str(r["ticker"]).ljust(5))
                signal_col = escape(str(r["signal"]).ljust(14))
                lines.append(
                    f"  {sig_emoji} <code>{ticker_col}</code> "
                    f"{signal_col} {conf_pct} {traded}"
                )

        if errors:
            lines.append("")
            lines.append(f"<b>Errors ({len(errors)}):</b>")
            for e in errors[:5]:          # cap at 5 to avoid huge messages
                lines.append(f"  • {escape(e[:120])}")
            if len(errors) > 5:
                lines.append(f"  … and {len(errors) - 5} more")

        self._send("\n".join(lines), html=True)

    def send_price_alert(self, message: str) -> None:
        """
        Send a price / stop-loss / take-profit alert from PriceMonitor.

        Args:
            message: Pre-formatted alert string from PriceMonitor
                     (already contains emoji and ticker details).
        """
        self._send(escape(message[:500]), html=True)

    def send_error(self, message: str) -> None:
        """
        Send a plain error alert.

        Args:
            message: Error description.
        """
        self._send(
            f"❗ <b>Trading System Error</b>\n\n{escape(message[:500])}",
            html=True,
        )
