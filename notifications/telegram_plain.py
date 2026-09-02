"""
Dependency-free Telegram sender for the ops layer (watchdog, OnFailure handler).

Deliberately uses only the standard library and sends **plain text**
(no ``parse_mode``), so it keeps working when the application venv or
the repo import graph is broken — which is exactly when an ops alert is
most needed. Nothing here can produce a Telegram formatting 400.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

MAX_MESSAGE_LEN = 4096
_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def credentials_from_env() -> tuple[str, str] | None:
    """Return ``(token, chat_id)`` from the environment, or None if unset."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        return token, chat_id
    return None


def send_plain_text(
    token: str,
    chat_id: str,
    text: str,
    *,
    timeout: float = 15.0,
) -> tuple[bool, str]:
    """
    POST ``text`` to ``chat_id`` as a plain-text message.

    Returns ``(ok, detail)``. Never raises; ``detail`` carries the HTTP
    status / error text so callers can log why a send failed.
    """
    if len(text) > MAX_MESSAGE_LEN:
        text = text[:MAX_MESSAGE_LEN - 1] + "…"
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        _API_URL.format(token=token),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
            status = getattr(resp, "status", 200)
            if 200 <= status < 300:
                return True, f"HTTP {status}"
            return False, f"HTTP {status}"
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        return False, f"HTTP {exc.code}: {body}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
