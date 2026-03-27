"""
Bull / Bear Debate agent — adversarial reasoning before trade decisions.

Inspired by TradingAgents (github.com/TauricResearch/TradingAgents).

Two virtual analysts argue opposing sides of every non-HOLD signal:

    BullResearcher — argues FOR the trade, finds supporting evidence
    BearResearcher — argues AGAINST the trade, surfaces risks

A synthesis step merges both perspectives into a final verdict with an
adjusted confidence score.

When both researchers *agree* on direction, confidence gets a boost.
When they *disagree*, confidence is penalised and the system becomes
more cautious.

Recovery behaviour
------------------
If the debate fails for any reason (API error, timeout, malformed
response), the original signal and confidence pass through unchanged
so the trading pipeline is never blocked.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import re

import anthropic

from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL
from utils.api_recovery import APIRecovery

log = logging.getLogger(__name__)

# ── Result container ─────────────────────────────────────────────────────────


@dataclass
class DebateResult:
    """Immutable container for the outcome of a bull/bear debate."""

    ticker: str
    original_signal: str
    original_confidence: float
    bull_case: str = ""
    bear_case: str = ""
    final_signal: str = ""
    adjusted_confidence: float = 0.0
    debate_summary: str = ""
    degraded: bool = False


# ── Prompt templates ─────────────────────────────────────────────────────────

_BULL_PROMPT = (
    "You are a bullish equity analyst making the strongest possible case "
    "FOR taking a {signal} position in {ticker}.\n\n"
    "Current data:\n"
    "- Combined signal: {signal}\n"
    "- Confidence: {confidence:.0%}\n"
    "- Technical indicators: {technical}\n"
    "- Sentiment summary: {sentiment}\n\n"
    "Argue convincingly why this trade SHOULD be taken. Cite the data above.\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"bull_case": "<2-3 sentences>", "confidence_boost": <float -0.2 to 0.2>}}'
)

_BEAR_PROMPT = (
    "You are a skeptical risk manager making the strongest possible case "
    "AGAINST taking a {signal} position in {ticker}.\n\n"
    "Current data:\n"
    "- Combined signal: {signal}\n"
    "- Confidence: {confidence:.0%}\n"
    "- Technical indicators: {technical}\n"
    "- Sentiment summary: {sentiment}\n\n"
    "Argue convincingly why this trade is RISKY and should NOT be taken. "
    "Identify weaknesses, contrary indicators, and potential traps.\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"bear_case": "<2-3 sentences>", "confidence_penalty": <float -0.2 to 0.0>}}'
)


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON output from Claude."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


# ── Individual researchers ───────────────────────────────────────────────────


class BullResearcher:
    """Argues FOR the trade. Finds supporting evidence."""

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def analyze(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        technical_data: dict,
        sentiment_data: dict,
    ) -> dict:
        prompt = _BULL_PROMPT.format(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            technical=json.dumps(technical_data, default=str),
            sentiment=json.dumps(sentiment_data, default=str),
        )

        def _call():
            msg = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(_strip_fences(msg.content[0].text))

        try:
            result = APIRecovery.call("anthropic", _call)
            return {
                "bull_case": str(result.get("bull_case", "")),
                "confidence_boost": max(-0.2, min(0.2, float(result.get("confidence_boost", 0.0)))),
            }
        except Exception as exc:
            log.warning("BullResearcher failed: %s", exc)
            return {"bull_case": "Analysis unavailable", "confidence_boost": 0.0}


class BearResearcher:
    """Argues AGAINST the trade. Finds risks."""

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def analyze(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        technical_data: dict,
        sentiment_data: dict,
    ) -> dict:
        prompt = _BEAR_PROMPT.format(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            technical=json.dumps(technical_data, default=str),
            sentiment=json.dumps(sentiment_data, default=str),
        )

        def _call():
            msg = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(_strip_fences(msg.content[0].text))

        try:
            result = APIRecovery.call("anthropic", _call)
            return {
                "bear_case": str(result.get("bear_case", "")),
                "confidence_penalty": max(-0.2, min(0.0, float(result.get("confidence_penalty", 0.0)))),
            }
        except Exception as exc:
            log.warning("BearResearcher failed: %s", exc)
            return {"bear_case": "Analysis unavailable", "confidence_penalty": 0.0}


# ── Debate orchestrator ──────────────────────────────────────────────────────

# Signals that skip debate entirely (no point debating a HOLD)
_SKIP_SIGNALS = {"HOLD", "CONFLICTING"}


class BullBearDebate:
    """Runs both researchers and synthesises a final verdict."""

    def __init__(
        self,
        bull: BullResearcher | None = None,
        bear: BearResearcher | None = None,
    ) -> None:
        self._bull = bull or BullResearcher()
        self._bear = bear or BearResearcher()

    @staticmethod
    def is_enabled() -> bool:
        return os.environ.get("ENABLE_BULL_BEAR_DEBATE", "").lower() in ("true", "1", "yes")

    def run(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        technical_data: dict,
        sentiment_data: dict,
    ) -> DebateResult:
        """Run bull/bear debate and return adjusted signal + confidence."""

        # Skip debate for HOLD / CONFLICTING
        if signal in _SKIP_SIGNALS:
            return DebateResult(
                ticker=ticker,
                original_signal=signal,
                original_confidence=confidence,
                final_signal=signal,
                adjusted_confidence=confidence,
                debate_summary="Debate skipped — signal is {}.".format(signal),
            )

        kwargs = dict(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            technical_data=technical_data,
            sentiment_data=sentiment_data,
        )

        try:
            # Run both researchers (sync — each is a single short API call)
            bull_result = self._bull.analyze(**kwargs)
            bear_result = self._bear.analyze(**kwargs)
        except Exception as exc:
            log.error("BullBearDebate failed: %s — using original signal", exc)
            return DebateResult(
                ticker=ticker,
                original_signal=signal,
                original_confidence=confidence,
                final_signal=signal,
                adjusted_confidence=confidence,
                debate_summary="Debate failed — using original signal.",
                degraded=True,
            )

        return self._synthesise(
            ticker, signal, confidence, bull_result, bear_result,
        )

    async def run_async(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        technical_data: dict,
        sentiment_data: dict,
    ) -> DebateResult:
        """Async version — runs bull and bear in parallel via asyncio."""

        if signal in _SKIP_SIGNALS:
            return DebateResult(
                ticker=ticker,
                original_signal=signal,
                original_confidence=confidence,
                final_signal=signal,
                adjusted_confidence=confidence,
                debate_summary="Debate skipped — signal is {}.".format(signal),
            )

        kwargs = dict(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            technical_data=technical_data,
            sentiment_data=sentiment_data,
        )

        try:
            bull_result, bear_result = await asyncio.gather(
                asyncio.to_thread(self._bull.analyze, **kwargs),
                asyncio.to_thread(self._bear.analyze, **kwargs),
            )
        except Exception as exc:
            log.error("BullBearDebate async failed: %s — using original signal", exc)
            return DebateResult(
                ticker=ticker,
                original_signal=signal,
                original_confidence=confidence,
                final_signal=signal,
                adjusted_confidence=confidence,
                debate_summary="Debate failed — using original signal.",
                degraded=True,
            )

        return self._synthesise(
            ticker, signal, confidence, bull_result, bear_result,
        )

    # ── Private synthesis ────────────────────────────────────────────────

    @staticmethod
    def _synthesise(
        ticker: str,
        signal: str,
        confidence: float,
        bull: dict,
        bear: dict,
    ) -> DebateResult:
        """Merge bull and bear perspectives into a final verdict.

        Outcome classification based on bear pushback strength:
            agree    (penalty > -0.05) → 0 adjustment
            cautious (penalty -0.05 to -0.12) → -0.05 to -0.15
            disagree (penalty <= -0.12) → -0.15 to -0.30
        """

        penalty = bear.get("confidence_penalty", 0.0)

        # Classify outcome by bear pushback strength
        if penalty > -0.05:
            # Agree: bear sees little risk → no penalty
            adjustment = 0.0
            summary = "Bull and bear broadly agree — confidence unchanged."
        elif penalty > -0.12:
            # Cautious: moderate bear pushback → -0.05 to -0.15
            bear_strength = abs(penalty)
            t = (bear_strength - 0.05) / 0.07  # 0.0–1.0
            adjustment = -(0.05 + t * 0.10)
            summary = "Bear raises concerns — cautious penalty applied."
        else:
            # Disagree: strong bear pushback → -0.15 to -0.30
            bear_strength = min(abs(penalty), 0.20)
            t = (bear_strength - 0.12) / 0.08  # 0.0–1.0
            adjustment = -(0.15 + t * 0.15)
            summary = "Bull and bear disagree — significant penalty applied."

        adjusted = round(max(0.0, min(1.0, confidence + adjustment)), 2)

        # If confidence drops below 0.15, downgrade signal to HOLD
        final_signal = signal
        if adjusted < 0.15:
            final_signal = "HOLD"
            summary += " Confidence too low — downgraded to HOLD."

        bull_case = bull.get("bull_case", "")
        bear_case = bear.get("bear_case", "")

        log.debug(
            "Debate %s: penalty=%.3f adjustment=%.3f conf=%.2f→%.2f signal=%s→%s",
            ticker, penalty, adjustment, confidence, adjusted, signal, final_signal,
        )

        return DebateResult(
            ticker=ticker,
            original_signal=signal,
            original_confidence=confidence,
            bull_case=bull_case,
            bear_case=bear_case,
            final_signal=final_signal,
            adjusted_confidence=adjusted,
            debate_summary=summary,
        )
