"""Tests for notifications/telegram_plain.py and scripts/nts_alert.py."""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.error
from datetime import datetime, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from notifications import telegram_plain  # noqa: E402
from scripts import nts_alert  # noqa: E402


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestPlainSender:
    def test_posts_plain_text_without_parse_mode(self):
        with patch("notifications.telegram_plain.urllib.request.urlopen",
                   return_value=_Resp(b"{}")) as m:
            ok, detail = telegram_plain.send_plain_text("tok", "cid", "hello *_<>& world")
        assert ok and detail == "HTTP 200"
        req = m.call_args[0][0]
        assert req.full_url == "https://api.telegram.org/bottok/sendMessage"
        body = json.loads(req.data.decode())
        assert body == {"chat_id": "cid", "text": "hello *_<>& world"}

    def test_truncates_to_limit(self):
        with patch("notifications.telegram_plain.urllib.request.urlopen",
                   return_value=_Resp(b"{}")) as m:
            telegram_plain.send_plain_text("t", "c", "x" * 5000)
        body = json.loads(m.call_args[0][0].data.decode())
        assert len(body["text"]) == telegram_plain.MAX_MESSAGE_LEN
        assert body["text"].endswith("…")

    def test_http_error_reported_not_raised(self):
        err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b'{"ok":false}'))
        with patch("notifications.telegram_plain.urllib.request.urlopen", side_effect=err):
            ok, detail = telegram_plain.send_plain_text("t", "c", "x")
        assert ok is False
        assert detail.startswith("HTTP 401")

    def test_network_error_reported_not_raised(self):
        with patch("notifications.telegram_plain.urllib.request.urlopen",
                   side_effect=OSError("network down")):
            ok, detail = telegram_plain.send_plain_text("t", "c", "x")
        assert ok is False and "network down" in detail

    def test_credentials_from_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", " t ")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        assert telegram_plain.credentials_from_env() == ("t", "c")
        monkeypatch.delenv("TELEGRAM_CHAT_ID")
        assert telegram_plain.credentials_from_env() is None


class TestNtsAlert:
    SHOW = "Result=exit-code\nExecMainStatus=1\nActiveState=failed\nNRestarts=0\n"

    def test_message_shape(self):
        msg = nts_alert.build_message(
            "nts-ohlc-ingest.service", show_output=self.SHOW,
            journal_tail="2026-09-01T22:31:00+0000 claw python3[1]: FRESHNESS GATE FAILED",
            host="claw", now=datetime(2026, 9, 1, 22, 31, tzinfo=timezone.utc),
        )
        assert msg.splitlines()[0] == "❌ nts-ohlc-ingest.service FAILED on claw — 2026-09-01 22:31 UTC"
        assert "Result=exit-code exit=1 state=failed NRestarts=0" in msg
        assert msg.endswith("FRESHNESS GATE FAILED")

    def test_main_sends_and_always_exits_zero(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        sent = []
        with patch("scripts.nts_alert._run", side_effect=[self.SHOW, "journal line"]), \
             patch("scripts.nts_alert.send_plain_text",
                   side_effect=lambda t, c, m: sent.append(m) or (True, "HTTP 200")):
            assert nts_alert.main(["nts-backup.service"]) == 0
        assert sent and sent[0].startswith("❌ nts-backup.service FAILED on ")
        assert "journal line" in sent[0]

    def test_delivery_failure_still_exits_zero(self, monkeypatch, capsys):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        with patch("scripts.nts_alert._run", return_value=""), \
             patch("scripts.nts_alert.send_plain_text", return_value=(False, "HTTP 502")):
            assert nts_alert.main(["x.service"]) == 0
        assert "delivery failed (HTTP 502)" in capsys.readouterr().err

    def test_missing_credentials_logs_and_exits_zero(self, monkeypatch, capsys):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        with patch("scripts.nts_alert._run", return_value=""):
            assert nts_alert.main(["x.service"]) == 0
        assert "not set" in capsys.readouterr().err
