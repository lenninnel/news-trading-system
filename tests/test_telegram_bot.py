"""
Tests for notifications.telegram_bot.TelegramNotifier.

All HTTP calls are mocked — no real Telegram API requests are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from notifications.telegram_bot import TelegramNotifier


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def notifier():
    """A TelegramNotifier with dummy credentials."""
    return TelegramNotifier(
        bot_token="123456:ABC-DEF",
        chat_id="99999",
        dashboard_url="https://example.com/dashboard",
    )


@pytest.fixture
def mock_post():
    """Patch requests.post and return a successful Telegram API response."""
    with patch("notifications.telegram_bot.requests.post") as m:
        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        m.return_value = resp
        yield m


# ── from_config ──────────────────────────────────────────────────────────────

class TestFromConfig:
    """Factory method tests."""

    def test_returns_none_when_disabled(self):
        cfg = {"telegram": {"enabled": False, "bot_token": "t", "chat_id": "c"}}
        assert TelegramNotifier.from_config(cfg) is None

    def test_returns_none_when_section_missing(self):
        assert TelegramNotifier.from_config({}) is None

    def test_returns_none_when_credentials_missing(self):
        cfg = {"telegram": {"enabled": True}}
        with patch.dict("os.environ", {}, clear=True):
            result = TelegramNotifier.from_config(cfg)
        assert result is None

    def test_returns_notifier_with_env_vars(self):
        cfg = {"telegram": {"enabled": True, "bot_token": "", "chat_id": ""}}
        env = {"TELEGRAM_BOT_TOKEN": "tok123", "TELEGRAM_CHAT_ID": "chat456"}
        with patch.dict("os.environ", env, clear=False):
            result = TelegramNotifier.from_config(cfg)
        assert result is not None
        assert result._token == "tok123"
        assert result._chat_id == "chat456"

    def test_env_vars_override_yaml(self):
        cfg = {
            "telegram": {
                "enabled": True,
                "bot_token": "yaml-token",
                "chat_id": "yaml-chat",
            }
        }
        env = {"TELEGRAM_BOT_TOKEN": "env-token", "TELEGRAM_CHAT_ID": "env-chat"}
        with patch.dict("os.environ", env, clear=False):
            result = TelegramNotifier.from_config(cfg)
        assert result._token == "env-token"
        assert result._chat_id == "env-chat"


# ── send_signal ──────────────────────────────────────────────────────────────

class TestSendSignal:
    """Tests for send_signal()."""

    def test_sends_signal_message(self, notifier, mock_post):
        notifier.send_signal("AAPL", "STRONG BUY", 85.0, "Bullish sentiment")

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["chat_id"] == "99999"
        assert "AAPL" in payload["text"]
        assert "STRONG BUY" in payload["text"]
        assert "85%" in payload["text"]

    def test_includes_reasoning_when_provided(self, notifier, mock_post):
        notifier.send_signal("TSLA", "WEAK SELL", 40.0, "Bearish technicals")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert "Bearish technicals" in payload["text"]

    def test_works_without_reasoning(self, notifier, mock_post):
        notifier.send_signal("MSFT", "HOLD", 25.0)

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert "MSFT" in payload["text"]

    def test_does_not_raise_on_api_failure(self, notifier):
        with patch("notifications.telegram_bot.requests.post", side_effect=ConnectionError("down")):
            # Should not raise
            notifier.send_signal("AAPL", "STRONG BUY", 90.0)

    def test_does_not_raise_on_http_error(self, notifier):
        with patch("notifications.telegram_bot.requests.post") as m:
            resp = MagicMock()
            resp.ok = False
            resp.status_code = 403
            resp.text = "Forbidden"
            m.return_value = resp
            # Should not raise
            notifier.send_signal("AAPL", "STRONG BUY", 90.0)

    def test_includes_debate_summary_when_provided(self, notifier, mock_post):
        notifier.send_signal(
            "AAPL", "STRONG BUY", 85.0,
            reasoning="Bullish sentiment",
            debate_summary="Bull and bear broadly agree — confidence boosted.",
        )
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert "Bull and bear broadly agree" in payload["text"]

    def test_works_without_debate_summary(self, notifier, mock_post):
        notifier.send_signal("AAPL", "STRONG BUY", 85.0, "Bullish")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        # No debate line should appear
        assert "🐂🐻" not in payload["text"]


# ── send_trade_executed ──────────────────────────────────────────────────────

class TestSendTradeExecuted:
    """Tests for send_trade_executed()."""

    def test_sends_buy_trade(self, notifier, mock_post):
        notifier.send_trade_executed(
            ticker="AAPL", action="BUY", shares=10,
            price=195.42, stop_loss=191.42, take_profit=203.42,
        )

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        text = payload["text"]
        assert "AAPL" in text
        assert "BUY" in text
        assert "10" in text
        assert "195.42" in text

    def test_sends_sell_trade(self, notifier, mock_post):
        notifier.send_trade_executed(
            ticker="TSLA", action="SELL", shares=5,
            price=250.00, stop_loss=260.00, take_profit=230.00,
        )

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert "SELL" in payload["text"]
        assert "🔴" in payload["text"]


# ── send_daily_summary ───────────────────────────────────────────────────────

class TestSendDailySummary:
    """Tests for send_daily_summary()."""

    def test_sends_success_summary(self, notifier, mock_post):
        results = [
            {"ticker": "AAPL", "signal": "STRONG BUY", "conf": 0.85, "traded": True},
            {"ticker": "MSFT", "signal": "HOLD", "conf": 0.25, "traded": False},
        ]
        notifier.send_daily_summary(
            signals_count=2, trades_count=1, portfolio_value=10500.0,
            results=results, errors=[], status="success",
        )

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        text = payload["text"]
        assert "Daily Trading Summary" in text
        assert "SUCCESS" in text
        assert "AAPL" in text
        assert "$10,500.00" in text

    def test_sends_partial_summary_with_errors(self, notifier, mock_post):
        notifier.send_daily_summary(
            signals_count=3, trades_count=0, portfolio_value=10000.0,
            results=[], errors=["NVDA: API rate limit", "TSLA: timeout"],
            status="partial",
        )

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        text = payload["text"]
        assert "PARTIAL" in text
        assert "Errors (2)" in text

    def test_does_not_raise_on_failure(self, notifier):
        with patch("notifications.telegram_bot.requests.post", side_effect=Exception("boom")):
            notifier.send_daily_summary(
                signals_count=1, trades_count=0, portfolio_value=10000.0,
                results=[], errors=[], status="success",
            )


# ── send_error ───────────────────────────────────────────────────────────────

class TestSendError:
    """Tests for send_error()."""

    def test_sends_error_message(self, notifier, mock_post):
        notifier.send_error("Database connection lost")

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        text = payload["text"]
        assert "Error" in text
        assert "Database connection lost" in text

    def test_truncates_long_messages(self, notifier, mock_post):
        long_msg = "x" * 1000
        notifier.send_error(long_msg)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        # send_error truncates to 500 chars
        assert len(payload["text"]) < 600

    def test_does_not_raise_on_failure(self, notifier):
        with patch("notifications.telegram_bot.requests.post", side_effect=TimeoutError):
            notifier.send_error("test error")


# ── send_price_alert ─────────────────────────────────────────────────────────

class TestSendPriceAlert:
    """Tests for send_price_alert()."""

    def test_sends_alert(self, notifier, mock_post):
        notifier.send_price_alert("🔔 AAPL hit stop-loss at $190.00")

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert "AAPL" in payload["text"]
        assert "stop-loss" in payload["text"]


# ── _send internals ──────────────────────────────────────────────────────────

class TestSendInternal:
    """Tests for the low-level _send() method."""

    def test_returns_true_on_success(self, notifier, mock_post):
        assert notifier._send("test") is True

    def test_returns_false_on_http_error(self, notifier):
        with patch("notifications.telegram_bot.requests.post") as m:
            resp = MagicMock()
            resp.ok = False
            resp.status_code = 400
            resp.text = "Bad Request"
            m.return_value = resp
            assert notifier._send("test") is False

    def test_returns_false_on_exception(self, notifier):
        with patch("notifications.telegram_bot.requests.post", side_effect=ConnectionError):
            assert notifier._send("test") is False

    def test_includes_reply_markup_when_dashboard_url_set(self, notifier, mock_post):
        notifier._send("test")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert "reply_markup" in payload

    def test_no_reply_markup_without_dashboard_url(self, mock_post):
        notifier = TelegramNotifier(bot_token="tok", chat_id="cid")
        notifier._send("test")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert "reply_markup" not in payload

    def test_uses_markdown_parse_mode(self, notifier, mock_post):
        notifier._send("test")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["parse_mode"] == "Markdown"


# ── Misconfigured notifier ───────────────────────────────────────────────────

class TestMisconfigured:
    """Ensure methods return False / don't raise with bad credentials."""

    def test_send_returns_false_with_bad_token(self):
        notifier = TelegramNotifier(bot_token="invalid", chat_id="invalid")
        with patch("notifications.telegram_bot.requests.post") as m:
            resp = MagicMock()
            resp.ok = False
            resp.status_code = 401
            resp.text = "Unauthorized"
            m.return_value = resp
            assert notifier._send("hello") is False

    def test_send_signal_does_not_raise_with_bad_creds(self):
        notifier = TelegramNotifier(bot_token="bad", chat_id="bad")
        with patch("notifications.telegram_bot.requests.post", side_effect=Exception("auth fail")):
            # Should silently fail, not raise
            notifier.send_signal("AAPL", "STRONG BUY", 90.0)

    def test_send_daily_summary_does_not_raise_with_bad_creds(self):
        notifier = TelegramNotifier(bot_token="bad", chat_id="bad")
        with patch("notifications.telegram_bot.requests.post", side_effect=Exception("auth fail")):
            notifier.send_daily_summary(
                signals_count=0, trades_count=0, portfolio_value=0,
                results=[], errors=[], status="failed",
            )

    def test_send_error_does_not_raise_with_bad_creds(self):
        notifier = TelegramNotifier(bot_token="bad", chat_id="bad")
        with patch("notifications.telegram_bot.requests.post", side_effect=Exception("auth fail")):
            notifier.send_error("something broke")
