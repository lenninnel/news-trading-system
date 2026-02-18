"""
Telegram notification helper for the News Trading System.

Uses the Telegram Bot HTTP API directly via ``requests`` (already a project
dependency), so no extra packages or asyncio are needed.

Quick setup
-----------
1. Create a bot with BotFather: https://t.me/BotFather  → get BOT_TOKEN
2. Send any message to your bot, then visit:
       https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
   and copy the ``chat.id`` value.
3. Export two environment variables (or set them in .env):
       export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
       export TELEGRAM_CHAT_ID="-1001234567890"
4. Add a telegram section to config/watchlist.yaml:
       telegram:
         enabled: true
         bot_token: "${TELEGRAM_BOT_TOKEN}"
         chat_id: "${TELEGRAM_CHAT_ID}"
         dashboard_url: "http://localhost:8501"   # optional

Environment variables override any literal values in YAML.
"""

from __future__ import annotations

import json
import logging
import os
import re

import requests

log = logging.getLogger(__name__)

# Telegram Bot API base URL template
_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

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


def _expand_env(value: str) -> str:
    """Replace ``${VAR}`` and ``$VAR`` placeholders with environment values."""
    return re.sub(
        r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
        lambda m: os.environ.get(m.group(1) or m.group(2), m.group(0)),
        str(value),
    )


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
    # Factory
    # ------------------------------------------------------------------

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

    def _send(self, text: str, *, reply_markup: dict | None = None) -> bool:
        """
        POST a message to Telegram. Returns True on success, False on failure.
        Never raises.
        """
        payload: dict = {
            "chat_id":    self._chat_id,
            "text":       text,
            "parse_mode": "Markdown",
        }
        if reply_markup is None:
            reply_markup = self._keyboard()
        if reply_markup:
            payload["reply_markup"] = json.dumps(reply_markup)

        try:
            resp = requests.post(self._url, json=payload, timeout=10)
            if not resp.ok:
                log.warning(
                    "Telegram API error %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
            return True
        except Exception as exc:
            log.warning("Could not send Telegram message: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public send methods
    # ------------------------------------------------------------------

    def send_signal(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        reasoning: str = "",
    ) -> None:
        """
        Send a trading signal alert.

        Args:
            ticker:     Stock symbol, e.g. "AAPL".
            signal:     Combined signal string, e.g. "STRONG BUY".
            confidence: Confidence as a percentage (0–100).
            reasoning:  Optional short reasoning text.
        """
        emoji = _SIGNAL_EMOJI.get(signal, "📌")
        lines = [
            f"{emoji} *{ticker}* — `{signal}`",
            f"Confidence: *{confidence:.0f}%*",
        ]
        if reasoning:
            # Truncate long reasoning to keep message tidy
            short = reasoning[:300] + ("…" if len(reasoning) > 300 else "")
            lines.append(f"_{short}_")

        self._send("\n".join(lines))

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
            f"{direction_emoji} *Paper Trade Executed* — `{ticker}`",
            f"Action:      *{action}*",
            f"Shares:      {shares}",
            f"Price:       ${price:,.2f}",
            f"Value:       ${position_value:,.2f}",
            f"Stop-loss:   ${stop_loss:,.2f}",
            f"Take-profit: ${take_profit:,.2f}",
        ]
        self._send("\n".join(lines))

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
            f"{status_emoji} *Daily Trading Summary* — `{status.upper()}`",
            "",
            f"Signals generated : {signals_count}",
            f"Trades executed   : {trades_count}",
            f"Portfolio value   : ${portfolio_value:,.2f}",
        ]

        if results:
            lines.append("")
            lines.append("*Signal breakdown:*")
            for r in results:
                sig_emoji = _SIGNAL_EMOJI.get(r.get("signal", ""), "•")
                conf_pct  = f"{r['conf']:.0%}"
                traded    = "→ trade" if r.get("traded") else ""
                lines.append(
                    f"  {sig_emoji} `{r['ticker']:<5}` {r['signal']:<14} {conf_pct} {traded}"
                )

        if errors:
            lines.append("")
            lines.append(f"*Errors ({len(errors)}):*")
            for e in errors[:5]:          # cap at 5 to avoid huge messages
                lines.append(f"  • {e[:120]}")
            if len(errors) > 5:
                lines.append(f"  … and {len(errors) - 5} more")

        self._send("\n".join(lines))

    def send_error(self, message: str) -> None:
        """
        Send a plain error alert.

        Args:
            message: Error description.
        """
        self._send(f"❗ *Trading System Error*\n\n{message[:500]}")
