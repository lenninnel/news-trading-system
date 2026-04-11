"""
MacroContextAgent — once-per-session macro context for the bull/bear debate.

Fetches a short, structured summary of today's macro environment (VIX,
key events, sector moves, market tone, risk flags) via a single Claude
Haiku call with the server-side ``web_search`` tool. The result is
prepended to every bull and bear prompt for the session, so each ticker's
debate has the same shared context.

Design notes
------------
- ONE Claude call per session (not per ticker). Haiku + max_tokens=300 is
  cheap; the web_search tool cost is bounded by ``max_uses``.
- HARD 10-second wall clock; on any timeout / failure / disabled flag the
  method returns ``""`` and the caller proceeds with the legacy per-ticker
  debate prompt. The feature can never block a session.
- Only runs for US sessions (US_PRE / US_OPEN / PEAD_OPEN / PREMARKET_SCAN).
  XETRA sessions get an empty string.
- Disabled by default via ``ENABLE_MACRO_CONTEXT``; a test / prod roll-out
  flips the flag to ``true``.

Public API
----------
    agent = MacroContextAgent()
    ctx = await agent.get_context("US_PRE")  # str, possibly empty
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

import anthropic

from config.settings import ANTHROPIC_API_KEY

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

# Task spec says Haiku (cheapest). Matches the user memory entry for the
# latest Haiku model ID.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# US session names that should get a macro block. Other session names
# (XETRA_PRE / XETRA_OPEN / MIDDAY / EOD / PREMARKET_SCAN) return "".
#
# MIDDAY and EOD are excluded deliberately: MIDDAY is a monitor-only
# session (no debate), and EOD triggers outside US trading hours when the
# "today's macro" framing no longer applies.
US_SESSIONS: frozenset[str] = frozenset({
    "US_PRE",
    "US_OPEN",
    "PEAD_OPEN",
})

# Hard timeout on the entire get_context call. If Claude / web_search is
# slow, we prefer to drop the context and run the debate without it.
TIMEOUT_SECONDS = 10.0

MAX_TOKENS = 300

# Bounded spend: web_search costs per query, so cap at 5 searches per call.
WEB_SEARCH_MAX_USES = 5


# System prompt — tells Haiku exactly what output shape we want. Short
# and rigid so the agent doesn't add commentary or hedge. The {date} /
# {session} placeholders are filled at runtime.
_SYSTEM_PROMPT = (
    "You are a macro context summarizer for a trading system. "
    "Fetch current market data using web search and return ONLY a "
    "structured context block. Maximum 150 words. Be factual and "
    "concise. No opinions, no disclaimers, no markdown. "
    "Output exactly this format:\n\n"
    "MACRO CONTEXT [{date} {session}]:\n"
    "- VIX: <level> (<rising|falling|flat> vs yesterday)\n"
    "- Key events today: <comma-separated list of Fed speakers, "
    "economic releases, major earnings; or \"none\" if quiet>\n"
    "- Sector moves: Tech <pct>%, Energy <pct>%, Financials <pct>%, "
    "Healthcare <pct>%\n"
    "- Market tone: <one sentence>\n"
    "- Risk flags: <any major risks in last 4h, or \"none\">\n"
)


def is_enabled() -> bool:
    """Read ``ENABLE_MACRO_CONTEXT`` at call time (not module load)."""
    return os.environ.get("ENABLE_MACRO_CONTEXT", "").strip().lower() in (
        "true", "1", "yes",
    )


class MacroContextAgent:
    """Once-per-session Claude Haiku + web_search macro summarizer.

    Parameters
    ----------
    client:
        Optional pre-built ``anthropic.Anthropic`` instance. Tests pass a
        mock; production constructs one lazily so the agent is cheap to
        instantiate even when the feature is disabled.
    model:
        Override the Haiku model ID. Defaults to ``DEFAULT_MODEL``.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client  # created lazily in _call
        self._model = model or DEFAULT_MODEL

    async def get_context(self, session: str) -> str:
        """Fetch macro context for this session.

        Returns
        -------
        str
            A 150-word-max structured context block, or ``""`` when the
            feature is disabled, the session is non-US, Claude fails, or
            the call exceeds ``TIMEOUT_SECONDS``.

        Notes
        -----
        The timeout is enforced at the HTTP layer via the Anthropic SDK's
        ``timeout=`` kwarg on ``messages.create(...)``. We deliberately do
        NOT use ``asyncio.wait_for`` around ``asyncio.to_thread`` because
        under Python 3.14 + ``nest_asyncio`` (loaded by
        ``execution/ibkr_trader.py``) the combination raises
        "Timeout should be used inside a task" intermittently. The SDK
        timeout gives us the same guarantee with zero asyncio interaction
        and surfaces as a plain exception we catch below.
        """
        if not is_enabled():
            return ""
        if session not in US_SESSIONS:
            log.debug("MacroContext: skipping non-US session %s", session)
            return ""

        try:
            return await asyncio.to_thread(self._call, session)
        except Exception as exc:
            log.warning("MacroContext: call failed (%s) — session=%s",
                        exc, session)
            return ""

    # ── Private ──────────────────────────────────────────────────────────

    def _ensure_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._client

    def _call(self, session: str) -> str:
        """Blocking Claude call. Called from ``asyncio.to_thread``."""
        client = self._ensure_client()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        system = _SYSTEM_PROMPT.format(date=today, session=session)
        user_msg = (
            f"Generate today's macro context for a {session} trading "
            f"session on {today}. Use web search to get current VIX level, "
            f"Fed calendar, major US economic data releases, sector ETF "
            f"performance, and any market-moving headlines in the last "
            f"4 hours. Return ONLY the structured block described in the "
            f"system prompt."
        )

        msg = client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": WEB_SEARCH_MAX_USES,
            }],
            # HTTP-level timeout enforced by the Anthropic SDK — keeps the
            # async wrapper simple and avoids Python 3.14 asyncio quirks.
            timeout=TIMEOUT_SECONDS,
        )

        return self._extract_text(msg)

    @staticmethod
    def _extract_text(msg: object) -> str:
        """Pull the final text block out of a Claude response.

        A response that used tools contains a mix of ``text``, ``tool_use``,
        and server-side ``web_search_tool_result`` blocks. We only want the
        assistant's final narrative text — concatenate all text blocks in
        order and strip whitespace.
        """
        content = getattr(msg, "content", None) or []
        parts: list[str] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", "") or ""
                if text:
                    parts.append(text)
        joined = "\n".join(parts).strip()
        return joined
