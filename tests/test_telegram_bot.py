"""
Tests for notifications.telegram_bot.TelegramNotifier.

All HTTP calls are mocked — no real Telegram API requests are made.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from notifications.telegram_bot import (
    MAX_MESSAGE_LEN,
    TelegramNotifier,
    escape,
    legacy_markdown_to_html,
    strip_html,
)


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

    def test_uses_html_parse_mode(self, notifier, mock_post):
        notifier._send("test")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["parse_mode"] == "HTML"


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


# ── Plain-text fallback retry ────────────────────────────────────────────────

class TestPlainTextFallback:
    """A Markdown 400 must trigger one retry without parse_mode."""

    def _resp(self, ok, status=200, text="ok"):
        resp = MagicMock()
        resp.ok = ok
        resp.status_code = status
        resp.text = text
        return resp

    def test_retries_without_parse_mode_and_returns_true(self, notifier, caplog):
        with patch("notifications.telegram_bot.requests.post") as m:
            m.side_effect = [
                self._resp(False, 400, "Bad Request: can't parse entities"),
                self._resp(True),
            ]
            with caplog.at_level("WARNING", logger="notifications.telegram_bot"):
                assert notifier._send("test _broken markdown") is True

        assert m.call_count == 2
        first_payload = m.call_args_list[0].kwargs["json"]
        retry_payload = m.call_args_list[1].kwargs["json"]
        assert first_payload["parse_mode"] == "HTML"
        assert "parse_mode" not in retry_payload
        # Retry carries the readable plain text, tags stripped
        assert retry_payload["text"] == "test _broken markdown"
        assert any("Telegram API error 400" in r.message for r in caplog.records)

    def test_logs_both_failures_when_retry_also_fails(self, notifier, caplog):
        with patch("notifications.telegram_bot.requests.post") as m:
            m.side_effect = [
                self._resp(False, 400, "Bad Request"),
                self._resp(False, 400, "still bad"),
            ]
            with caplog.at_level("WARNING", logger="notifications.telegram_bot"):
                assert notifier._send("test") is False

        assert m.call_count == 2
        warnings = [r.message for r in caplog.records]
        assert any("Telegram API error 400" in w for w in warnings)
        assert any("plain-text retry failed" in w for w in warnings)

    def test_no_retry_on_success(self, notifier, mock_post):
        assert notifier._send("test") is True
        assert mock_post.call_count == 1


# ── HTML contract: no formatting 400 for any input ──────────────────────────

_TAG = re.compile(r"</?([a-z]+)>")


def _assert_well_formed_telegram_html(body: str) -> None:
    """Tags balanced, never nested, and no raw < > & outside tags."""
    stack: list[str] = []
    pos = 0
    for m in _TAG.finditer(body):
        between = body[pos:m.start()]
        assert "<" not in between and ">" not in between, between
        # every & must be an entity
        assert re.search(r"&(?!(amp|lt|gt|quot|#\d+);)", between) is None, between
        tag = m.group(1)
        if m.group(0).startswith("</"):
            assert stack and stack[-1] == tag, body
            stack.pop()
        else:
            assert not stack, f"nested tag {tag} in {body!r}"
            stack.append(tag)
        pos = m.end()
    tail = body[pos:]
    assert "<" not in tail and ">" not in tail, tail
    assert not stack, body


class TestLegacyMarkdownToHtml:
    def test_bold_and_code_convert(self):
        assert legacy_markdown_to_html("*Daemon started*") == "<b>Daemon started</b>"
        assert legacy_markdown_to_html("Runner: `abc-1`") == "Runner: <code>abc-1</code>"

    def test_underscores_are_never_markup(self):
        text = "no such table: daily_ohlc in news_feed.py"
        assert legacy_markdown_to_html(text) == text

    def test_html_specials_escaped_everywhere(self):
        out = legacy_markdown_to_html("*a<b>* & `c>d`")
        assert out == "<b>a&lt;b&gt;</b> &amp; <code>c&gt;d</code>"

    def test_unbalanced_markers_stay_literal(self):
        assert legacy_markdown_to_html("lone * here") == "lone * here"
        assert legacy_markdown_to_html("lone ` here") == "lone ` here"
        assert legacy_markdown_to_html("empty ** pair") == "empty ** pair"

    def test_markers_do_not_pair_across_lines(self):
        out = legacy_markdown_to_html("*open\nnext *line*")
        assert out == "*open\nnext <b>line</b>"

    def test_no_nesting_inside_spans(self):
        out = legacy_markdown_to_html("*bold `not code`*")
        assert out == "<b>bold `not code`</b>"
        out = legacy_markdown_to_html("`code *not bold*`")
        assert out == "<code>code *not bold*</code>"

    @pytest.mark.parametrize("text", [
        "🚨 *Scheduler error in EOD:*\nsqlite3.OperationalError: no such table: daily_ohlc",
        "PositionManager failed to start\n`Error 502 <html><body>Bad Gateway</body></html>`",
        "got an unexpected keyword argument '**kwargs' and *args",
        "``` triple ``` and _it_ and *b* and & < >",
        "*" * 7 + "`" * 5,
        "",
        "*",
        "\n*\n`\n",
    ])
    def test_output_is_always_well_formed(self, text):
        _assert_well_formed_telegram_html(legacy_markdown_to_html(text))

    def test_round_trip_through_strip(self):
        raw = "🚨 *Scheduler error in US_OPEN:*\nTimeout <60s> & retry"
        assert strip_html(legacy_markdown_to_html(raw)) == raw.replace("*", "")

    def test_escape_helper(self):
        assert escape("<a & b>") == "&lt;a &amp; b&gt;"


class TestSendHtmlContract:
    def test_legacy_text_is_converted_and_sent_as_html(self, notifier, mock_post):
        notifier._send("🚨 *Scheduler error in EOD:*\nno such table: daily_ohlc")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["parse_mode"] == "HTML"
        assert payload["text"] == (
            "🚨 <b>Scheduler error in EOD:</b>\nno such table: daily_ohlc"
        )

    def test_html_flag_sends_text_verbatim(self, notifier, mock_post):
        notifier._send("<b>x</b> &amp; y", html=True)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["text"] == "<b>x</b> &amp; y"
        assert payload["parse_mode"] == "HTML"

    def test_overlong_message_sent_plain_and_truncated(self, notifier, mock_post, caplog):
        text = "*head*\n" + "x" * (MAX_MESSAGE_LEN + 50)
        with caplog.at_level("WARNING", logger="notifications.telegram_bot"):
            assert notifier._send(text) is True
        payload = mock_post.call_args.kwargs["json"]
        assert "parse_mode" not in payload
        assert len(payload["text"]) == MAX_MESSAGE_LEN
        assert payload["text"].startswith("head\n")
        assert payload["text"].endswith("…")
        assert any("truncated" in r.message for r in caplog.records)

    def test_non_string_input_does_not_raise(self, notifier, mock_post):
        assert notifier._send(RuntimeError("boom <x>")) is True
        assert mock_post.call_args.kwargs["json"]["text"] == "boom &lt;x&gt;"


class TestTypedMethodsEscape:
    """Every typed method must HTML-escape its dynamic fields."""

    def test_send_message_is_verbatim_plain_text(self, notifier, mock_post):
        assert notifier.send_message("🛑 KILL SWITCH: STOP_TRADING\nReason: <manual> & *test*") is True
        payload = mock_post.call_args.kwargs["json"]
        assert payload["parse_mode"] == "HTML"
        assert payload["text"] == (
            "🛑 KILL SWITCH: STOP_TRADING\nReason: &lt;manual&gt; &amp; *test*"
        )
        _assert_well_formed_telegram_html(payload["text"])

    def test_send_error_escapes_message(self, notifier, mock_post):
        notifier.send_error("nts-ohlc-ingest FAILED: freshness gate: <20> tickers & daily_ohlc `x`")
        text = mock_post.call_args.kwargs["json"]["text"]
        assert text.startswith("❗ <b>Trading System Error</b>\n\n")
        assert "&lt;20&gt; tickers &amp; daily_ohlc `x`" in text
        _assert_well_formed_telegram_html(text)

    def test_send_price_alert_escapes(self, notifier, mock_post):
        notifier.send_price_alert("⚠️ STALE FEED: AAPL <last bar> 7.5min & counting")
        text = mock_post.call_args.kwargs["json"]["text"]
        assert text == "⚠️ STALE FEED: AAPL &lt;last bar&gt; 7.5min &amp; counting"

    def test_send_signal_escapes_reasoning(self, notifier, mock_post):
        notifier.send_signal("AAPL", "STRONG BUY", 85.0,
                             reasoning="a <b> & c_d *e*", debate_summary="x > y")
        text = mock_post.call_args.kwargs["json"]["text"]
        assert "<i>a &lt;b&gt; &amp; c_d *e*</i>" in text
        assert "🐂🐻 <i>x &gt; y</i>" in text
        _assert_well_formed_telegram_html(text)

    def test_send_daily_summary_escapes_errors(self, notifier, mock_post):
        notifier.send_daily_summary(
            signals_count=1, trades_count=0, portfolio_value=1.0,
            results=[{"ticker": "A<B", "signal": "HOLD", "conf": 0.1}],
            errors=["NVDA: <timeout> & daily_ohlc"], status="partial",
        )
        text = mock_post.call_args.kwargs["json"]["text"]
        assert "&lt;timeout&gt; &amp; daily_ohlc" in text
        assert "A&lt;B" in text
        _assert_well_formed_telegram_html(text)

    def test_send_trade_executed_escapes_ticker(self, notifier, mock_post):
        notifier.send_trade_executed("A&B", "BUY", 1, 1.0, 0.9, 1.1)
        text = mock_post.call_args.kwargs["json"]["text"]
        assert "<code>A&amp;B</code>" in text
        _assert_well_formed_telegram_html(text)


class TestFromEnv:
    def test_env_credentials_win_even_when_yaml_disabled(self):
        cfg = {"telegram": {"enabled": False, "dashboard_url": "http://d"}}
        env = {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "cid"}
        with patch.dict("os.environ", env, clear=False):
            n = TelegramNotifier.from_env(cfg)
        assert n is not None
        assert n._token == "tok" and n._chat_id == "cid"
        assert n._dashboard_url == "http://d"

    def test_falls_back_to_config_without_env(self):
        cfg = {"telegram": {"enabled": True, "bot_token": "yt", "chat_id": "yc"}}
        with patch.dict("os.environ", {}, clear=True):
            n = TelegramNotifier.from_env(cfg)
        assert n is not None and n._token == "yt"

    def test_none_when_nothing_configured(self):
        with patch.dict("os.environ", {}, clear=True):
            assert TelegramNotifier.from_env() is None
            assert TelegramNotifier.from_env({"telegram": {"enabled": False}}) is None
