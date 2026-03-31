"""Tests for the bull/bear debate agent."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bull_bear_debate import (
    BearResearcher,
    BullBearDebate,
    BullResearcher,
    DebateResult,
    _BULL_SYSTEM,
    _BEAR_SYSTEM,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_claude_response(payload: dict, *, fenced: bool = False) -> MagicMock:
    """Build a mock Anthropic message with the given JSON payload."""
    text = json.dumps(payload)
    if fenced:
        text = f"```json\n{text}\n```"
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _mock_client(payload: dict, *, fenced: bool = False) -> MagicMock:
    """Build a mock anthropic.Anthropic client returning *payload*."""
    client = MagicMock()
    client.messages.create.return_value = _mock_claude_response(payload, fenced=fenced)
    return client


_TECH_DATA = {"rsi": 55.0, "price": 150.0, "macd_histogram": 0.5}
_SENT_DATA = {"signal": "BUY", "avg_score": 0.3}


# ── BullResearcher ───────────────────────────────────────────────────────────

class TestBullResearcher:

    def test_returns_bull_case_and_boost(self):
        client = _mock_client({"bull_case": "Strong momentum.", "confidence_boost": 0.1})
        bull = BullResearcher(client=client)
        result = bull.analyze("AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA)
        assert result["bull_case"] == "Strong momentum."
        assert result["confidence_boost"] == 0.1

    def test_clamps_boost_to_max(self):
        client = _mock_client({"bull_case": "x", "confidence_boost": 0.99})
        result = BullResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["confidence_boost"] == 0.2

    def test_clamps_boost_to_min(self):
        client = _mock_client({"bull_case": "x", "confidence_boost": -0.99})
        result = BullResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["confidence_boost"] == -0.2

    def test_handles_fenced_json_response(self):
        client = _mock_client({"bull_case": "Fenced OK.", "confidence_boost": 0.08}, fenced=True)
        result = BullResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["bull_case"] == "Fenced OK."
        assert result["confidence_boost"] == 0.08

    def test_fallback_on_api_error(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        result = BullResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["bull_case"] == "Analysis unavailable"
        assert result["confidence_boost"] == 0.0


# ── BearResearcher ───────────────────────────────────────────────────────────

class TestBearResearcher:

    def test_returns_bear_case_and_penalty(self):
        client = _mock_client({"bear_case": "RSI divergence.", "confidence_penalty": -0.1})
        bear = BearResearcher(client=client)
        result = bear.analyze("AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA)
        assert result["bear_case"] == "RSI divergence."
        assert result["confidence_penalty"] == -0.1

    def test_clamps_penalty_to_zero(self):
        client = _mock_client({"bear_case": "x", "confidence_penalty": 0.5})
        result = BearResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["confidence_penalty"] == 0.0

    def test_clamps_penalty_to_min(self):
        client = _mock_client({"bear_case": "x", "confidence_penalty": -0.99})
        result = BearResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["confidence_penalty"] == -0.2

    def test_handles_fenced_json_response(self):
        client = _mock_client({"bear_case": "Fenced risk.", "confidence_penalty": -0.12}, fenced=True)
        result = BearResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["bear_case"] == "Fenced risk."
        assert result["confidence_penalty"] == -0.12

    def test_fallback_on_api_error(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        result = BearResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["bear_case"] == "Analysis unavailable"
        assert result["confidence_penalty"] == 0.0


# ── BullBearDebate ───────────────────────────────────────────────────────────

class TestBullBearDebate:

    def _make_debate(self, bull_payload: dict, bear_payload: dict) -> BullBearDebate:
        bull = BullResearcher(client=_mock_client(bull_payload))
        bear = BearResearcher(client=_mock_client(bear_payload))
        return BullBearDebate(bull=bull, bear=bear)

    # -- HOLD signals skip debate --

    def test_hold_skips_debate(self):
        debate = self._make_debate({}, {})
        result = debate.run("AAPL", "HOLD", 0.25, {}, {})
        assert result.final_signal == "HOLD"
        assert result.adjusted_confidence == 0.25
        assert "skipped" in result.debate_summary.lower()

    def test_conflicting_skips_debate(self):
        debate = self._make_debate({}, {})
        result = debate.run("AAPL", "CONFLICTING", 0.10, {}, {})
        assert result.final_signal == "CONFLICTING"
        assert result.adjusted_confidence == 0.10

    # -- Strong bull case: boost --

    def test_strong_bull_boosts_confidence(self):
        """Strong bull case with high boost and weak bear → confidence increases."""
        debate = self._make_debate(
            {"bull_case": "Strong trend.", "confidence_boost": 0.15},
            {"bear_case": "Minor worry.", "confidence_penalty": -0.02},
        )
        result = debate.run("AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA)
        assert result.adjusted_confidence > result.original_confidence
        assert result.debate_adjustment > 0
        assert result.final_signal == "STRONG BUY"

    # -- Cautious: moderate penalty --

    def test_cautious_reduces_confidence(self):
        """Bear risk words with modest penalty → adjusted_confidence < original."""
        debate = self._make_debate(
            {"bull_case": "Upside potential.", "confidence_boost": 0.1},
            {"bear_case": "Earnings risk.", "confidence_penalty": -0.08},
        )
        result = debate.run("AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA)
        assert result.adjusted_confidence < result.original_confidence
        assert result.debate_adjustment < 0

    def test_cautious_penalty_varies_with_strength(self):
        """Two cautious cases with different bear strength → different deltas."""
        debate_mild = self._make_debate(
            {"bull_case": "OK.", "confidence_boost": 0.05},
            {"bear_case": "Some risk.", "confidence_penalty": -0.06},
        )
        debate_strong = self._make_debate(
            {"bull_case": "OK.", "confidence_boost": 0.05},
            {"bear_case": "More risk.", "confidence_penalty": -0.10},
        )
        r_mild = debate_mild.run("AAPL", "BUY", 0.65, _TECH_DATA, _SENT_DATA)
        r_strong = debate_strong.run("AAPL", "BUY", 0.65, _TECH_DATA, _SENT_DATA)
        # Both cautious, but different penalties
        assert r_mild.adjusted_confidence > r_strong.adjusted_confidence

    # -- Disagreement: larger penalty --

    def test_disagree_reduces_more_than_cautious(self):
        """disagree outcome → adjusted_confidence < cautious penalty."""
        debate_cautious = self._make_debate(
            {"bull_case": "x.", "confidence_boost": 0.1},
            {"bear_case": "Moderate concern.", "confidence_penalty": -0.08},
        )
        debate_disagree = self._make_debate(
            {"bull_case": "x.", "confidence_boost": 0.1},
            {"bear_case": "Major risk.", "confidence_penalty": -0.15},
        )
        r_cautious = debate_cautious.run("AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA)
        r_disagree = debate_disagree.run("AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA)
        assert r_disagree.adjusted_confidence < r_cautious.adjusted_confidence
        assert "disagree" in r_disagree.debate_summary.lower()

    def test_disagreement_reduces_confidence(self):
        debate = self._make_debate(
            {"bull_case": "Uncertain.", "confidence_boost": -0.05},
            {"bear_case": "Major risk.", "confidence_penalty": -0.15},
        )
        result = debate.run("AAPL", "WEAK BUY", 0.45, _TECH_DATA, _SENT_DATA)
        assert result.adjusted_confidence < result.original_confidence
        assert "disagree" in result.debate_summary.lower()

    # -- Very low confidence downgrades to HOLD --

    def test_low_confidence_downgrades_to_hold(self):
        debate = self._make_debate(
            {"bull_case": "Weak.", "confidence_boost": -0.1},
            {"bear_case": "Very risky.", "confidence_penalty": -0.2},
        )
        result = debate.run("AAPL", "WEAK BUY", 0.20, _TECH_DATA, _SENT_DATA)
        assert result.final_signal == "HOLD"
        assert "downgraded" in result.debate_summary.lower()

    # -- Graceful fallback on debate failure --

    def test_fallback_on_debate_failure(self):
        bull = MagicMock()
        bull.analyze.side_effect = RuntimeError("catastrophic")
        bear = BearResearcher(client=_mock_client({"bear_case": "x", "confidence_penalty": -0.1}))
        debate = BullBearDebate(bull=bull, bear=bear)
        result = debate.run("AAPL", "STRONG BUY", 0.70, {}, {})
        assert result.final_signal == "STRONG BUY"
        assert result.adjusted_confidence == 0.70
        assert result.degraded is True

    # -- DebateResult dataclass --

    def test_debate_result_fields(self):
        r = DebateResult(
            ticker="AAPL",
            original_signal="STRONG BUY",
            original_confidence=0.65,
            bull_case="Good momentum",
            bear_case="RSI overbought",
            final_signal="STRONG BUY",
            adjusted_confidence=0.72,
            debate_summary="Bull and bear broadly agree.",
        )
        assert r.ticker == "AAPL"
        assert r.degraded is False

    # -- Confidence clamped to [0, 1] --

    def test_confidence_clamped_high(self):
        debate = self._make_debate(
            {"bull_case": "x", "confidence_boost": 0.2},
            {"bear_case": "x", "confidence_penalty": 0.0},
        )
        result = debate.run("AAPL", "STRONG BUY", 0.95, {}, {})
        assert result.adjusted_confidence <= 1.0

    def test_confidence_clamped_low(self):
        debate = self._make_debate(
            {"bull_case": "x", "confidence_boost": -0.2},
            {"bear_case": "x", "confidence_penalty": -0.2},
        )
        result = debate.run("AAPL", "WEAK BUY", 0.05, {}, {})
        assert result.adjusted_confidence >= 0.0

    # -- Debate can produce sub-floor values (coordinator enforces floors) --

    def test_harsh_penalty_can_drop_weak_buy_below_floor(self):
        """The debate _synthesise step itself does NOT enforce signal-type
        floors — that is the coordinator's job.  This test documents that
        the raw debate output CAN drop below the WEAK BUY floor of 0.35."""
        debate = self._make_debate(
            {"bull_case": "Uncertain.", "confidence_boost": -0.1},
            {"bear_case": "Major risk.", "confidence_penalty": -0.2},
        )
        result = debate.run("AAPL", "WEAK BUY", 0.45, _TECH_DATA, _SENT_DATA)
        # Max penalty is -0.15 → adjusted = 0.45 - 0.15 = 0.30
        assert result.adjusted_confidence < 0.35
        assert result.debate_adjustment == -0.15
        # The coordinator must clamp this back to 0.35

    def test_harsh_penalty_can_drop_strong_buy_below_floor(self):
        """Same documentation for STRONG BUY: debate can drop below 0.60."""
        debate = self._make_debate(
            {"bull_case": "Uncertain.", "confidence_boost": -0.15},
            {"bear_case": "Very risky.", "confidence_penalty": -0.2},
        )
        result = debate.run("AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA)
        assert result.adjusted_confidence < 0.60
        assert result.debate_adjustment == -0.15


# ── Async path ───────────────────────────────────────────────────────────────

class TestBullBearDebateAsync:

    def _make_debate(self, bull_payload: dict, bear_payload: dict) -> BullBearDebate:
        bull = BullResearcher(client=_mock_client(bull_payload))
        bear = BearResearcher(client=_mock_client(bear_payload))
        return BullBearDebate(bull=bull, bear=bear)

    def test_async_strong_bull_boosts(self):
        debate = self._make_debate(
            {"bull_case": "Strong.", "confidence_boost": 0.1},
            {"bear_case": "Minor.", "confidence_penalty": -0.02},
        )
        result = asyncio.run(debate.run_async(
            "AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA,
        ))
        assert result.adjusted_confidence > result.original_confidence
        assert result.debate_adjustment > 0

    def test_async_hold_skips(self):
        debate = self._make_debate({}, {})
        result = asyncio.run(debate.run_async("AAPL", "HOLD", 0.25, {}, {}))
        assert result.final_signal == "HOLD"

    def test_async_fallback_on_failure(self):
        bull = MagicMock()
        bull.analyze.side_effect = RuntimeError("fail")
        bear = MagicMock()
        bear.analyze.side_effect = RuntimeError("fail")
        debate = BullBearDebate(bull=bull, bear=bear)
        result = asyncio.run(debate.run_async("AAPL", "STRONG BUY", 0.7, {}, {}))
        assert result.final_signal == "STRONG BUY"
        assert result.degraded is True


# ── is_enabled flag ──────────────────────────────────────────────────────────

class TestIsEnabled:

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENABLE_BULL_BEAR_DEBATE", None)
            assert BullBearDebate.is_enabled() is False

    def test_enabled_with_true(self):
        with patch.dict(os.environ, {"ENABLE_BULL_BEAR_DEBATE": "true"}):
            assert BullBearDebate.is_enabled() is True

    def test_enabled_with_one(self):
        with patch.dict(os.environ, {"ENABLE_BULL_BEAR_DEBATE": "1"}):
            assert BullBearDebate.is_enabled() is True

    def test_disabled_with_false(self):
        with patch.dict(os.environ, {"ENABLE_BULL_BEAR_DEBATE": "false"}):
            assert BullBearDebate.is_enabled() is False


# ── Dynamic adjustment tests ────────────────────────────────────────────────


class TestDynamicDebateAdjustment:
    """Tests for the new dynamic debate adjustment (replaces hardcoded -6%)."""

    def _make_debate(self, bull_payload: dict, bear_payload: dict) -> BullBearDebate:
        bull = BullResearcher(client=_mock_client(bull_payload))
        bear = BearResearcher(client=_mock_client(bear_payload))
        return BullBearDebate(bull=bull, bear=bear)

    def test_debate_returns_dynamic_adjustment(self):
        """DebateResult includes a numeric debate_adjustment field."""
        debate = self._make_debate(
            {"bull_case": "Solid trend.", "confidence_boost": 0.05},
            {"bear_case": "Some concern.", "confidence_penalty": -0.06},
        )
        result = debate.run("AAPL", "BUY", 0.60, _TECH_DATA, _SENT_DATA)
        assert hasattr(result, "debate_adjustment")
        assert isinstance(result.debate_adjustment, float)
        assert result.debate_adjustment != 0  # not the old fixed value
        assert result.adjusted_confidence == round(
            result.original_confidence + result.debate_adjustment, 2,
        )

    def test_strong_bull_case_gives_positive_adjustment(self):
        """Clear bull win: strong conviction + high boost → positive adjustment."""
        debate = self._make_debate(
            {"bull_case": "Strong compelling trend with clear decisive momentum.",
             "confidence_boost": 0.18},
            {"bear_case": "Minor worry.", "confidence_penalty": -0.01},
        )
        result = debate.run("AAPL", "STRONG BUY", 0.60, _TECH_DATA, _SENT_DATA)
        assert result.debate_adjustment > 0
        assert result.adjusted_confidence > result.original_confidence

    def test_contested_debate_gives_negative_adjustment(self):
        """Contested / slight bear win: weak bull + risk words → negative."""
        debate = self._make_debate(
            {"bull_case": "Maybe upside, although uncertain.",
             "confidence_boost": -0.02},
            {"bear_case": "Significant risk of declining value, concern about weakness.",
             "confidence_penalty": -0.12},
        )
        result = debate.run("AAPL", "BUY", 0.60, _TECH_DATA, _SENT_DATA)
        assert result.debate_adjustment < 0
        assert result.adjusted_confidence < result.original_confidence

    def test_adjustment_bounded_between_minus15_and_plus10(self):
        """Adjustment is always clamped to [-0.15, +0.10] regardless of inputs."""
        # Extreme bull case
        debate_bull = self._make_debate(
            {"bull_case": "Strong compelling robust significant clear solid decisive.",
             "confidence_boost": 0.20},
            {"bear_case": "None.", "confidence_penalty": 0.0},
        )
        r_bull = debate_bull.run("AAPL", "STRONG BUY", 0.50, {}, {})
        assert r_bull.debate_adjustment <= 0.10

        # Extreme bear case
        debate_bear = self._make_debate(
            {"bull_case": "Weak uncertain unclear.",
             "confidence_boost": -0.20},
            {"bear_case": "Significant risk, vulnerable, declining, dangerous concern.",
             "confidence_penalty": -0.20},
        )
        r_bear = debate_bear.run("AAPL", "WEAK BUY", 0.50, {}, {})
        assert r_bear.debate_adjustment >= -0.15


# ── Prompt caching tests ────────────────────────────────────────────────────


class TestPromptCaching:
    """Verify cache_control is added to system prompts when enabled."""

    def test_debate_includes_cache_control_when_enabled(self):
        """BullResearcher and BearResearcher pass cache_control in system param."""
        client = _mock_client({"bull_case": "OK.", "confidence_boost": 0.0})
        bull = BullResearcher(client=client)

        with patch("agents.bull_bear_debate.ENABLE_PROMPT_CACHING", True):
            bull.analyze("AAPL", "BUY", 0.5, {}, {})

        call_kwargs = client.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["type"] == "text"
        assert system[0]["text"] == _BULL_SYSTEM
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_disabled_when_flag_false(self):
        """When ENABLE_PROMPT_CACHING=False, system is a plain string."""
        client = _mock_client({"bear_case": "OK.", "confidence_penalty": 0.0})
        bear = BearResearcher(client=client)

        with patch("agents.bull_bear_debate.ENABLE_PROMPT_CACHING", False):
            bear.analyze("AAPL", "BUY", 0.5, {}, {})

        call_kwargs = client.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert isinstance(system, str)
        assert system == _BEAR_SYSTEM
