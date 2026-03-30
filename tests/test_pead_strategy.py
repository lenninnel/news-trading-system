"""
Tests for PEADStrategy — Post-Earnings Announcement Drift.

Covers:
  - Signal generation on recent beat
  - No signal on miss
  - No signal outside hold window
  - Confidence scaling by surprise magnitude
  - Logging to signal_events
"""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from strategies.pead_strategy import PEADStrategy


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def cache_file(tmp_path):
    """Create a temp earnings cache with known data."""
    data = {
        "CASY": [
            {
                "period_end": "2024-03-31",
                "announce_date": "2024-04-22",
                "actual_eps": 4.85,
                "estimate_eps": 3.91,
                "surprise_pct": 24.04,
            },
            {
                "period_end": "2024-06-30",
                "announce_date": "2024-07-25",
                "actual_eps": 1.50,
                "estimate_eps": 1.40,
                "surprise_pct": 7.14,
            },
        ],
        "TXRH": [
            {
                "period_end": "2024-03-31",
                "announce_date": "2024-04-18",
                "actual_eps": 2.10,
                "estimate_eps": 1.75,
                "surprise_pct": 20.0,
            },
        ],
        "DECK": [
            {
                "period_end": "2024-03-31",
                "announce_date": "2024-05-01",
                "actual_eps": 3.00,
                "estimate_eps": 2.80,
                "surprise_pct": 12.5,
            },
        ],
        "PBR": [
            {
                "period_end": "2024-03-31",
                "announce_date": "2024-04-20",
                "actual_eps": 1.00,
                "estimate_eps": 1.10,
                "surprise_pct": -9.1,
            },
        ],
        "MEDP": [
            {
                "period_end": "2024-03-31",
                "announce_date": "2024-04-22",
                "actual_eps": 2.50,
                "estimate_eps": 2.40,
                "surprise_pct": 4.17,
            },
        ],
    }
    path = tmp_path / "test_earnings.json"
    path.write_text(json.dumps(data, indent=2))
    return path


@pytest.fixture
def strategy(cache_file):
    return PEADStrategy(cache_path=cache_file)


# ── Tests ────────────────────────────────────────────────────────────


class TestPEADGeneratesSignalOnRecentBeat:
    def test_signal_on_beat_within_2_days(self, strategy):
        """Earnings beat >5% within 2 days generates a BUY signal."""
        # CASY announced on 2024-04-22 with 24.04% surprise
        result = strategy.generate_signal("CASY", date(2024, 4, 23))  # 1 day after

        assert result is not None
        assert result.signal == "BUY"
        assert result.strategy_name == "PEAD"
        assert "24.0%" in result.reasoning[0]

    def test_signal_on_exact_announce_date(self, strategy):
        """Signal generated on the announcement date itself."""
        result = strategy.generate_signal("CASY", date(2024, 4, 22))

        assert result is not None
        assert result.signal == "BUY"

    def test_signal_2_days_after(self, strategy):
        """Signal still active 2 days after announcement."""
        result = strategy.generate_signal("CASY", date(2024, 4, 24))  # 2 days after

        assert result is not None
        assert result.signal == "BUY"


class TestPEADNoSignalOnMiss:
    def test_no_signal_on_earnings_miss(self, strategy):
        """Earnings miss (negative surprise) generates no signal."""
        # PBR had -9.1% surprise on 2024-04-20
        result = strategy.generate_signal("PBR", date(2024, 4, 21))

        assert result is None

    def test_no_signal_on_small_beat(self, strategy):
        """Beat below 5% threshold generates no signal."""
        # MEDP had 4.17% surprise — below threshold
        result = strategy.generate_signal("MEDP", date(2024, 4, 23))

        assert result is None

    def test_no_signal_for_unknown_ticker(self, strategy):
        """Ticker not in cache returns None."""
        result = strategy.generate_signal("ZZZZZ", date(2024, 4, 22))

        assert result is None


class TestPEADNoSignalOutsideHoldWindow:
    def test_no_signal_3_days_after(self, strategy):
        """No signal 3 days after announcement (outside 2-day lookback)."""
        # CASY announced 2024-04-22; 3 days after = 2024-04-25
        result = strategy.generate_signal("CASY", date(2024, 4, 25))

        assert result is None

    def test_no_signal_before_announcement(self, strategy):
        """No signal before the announcement date."""
        result = strategy.generate_signal("CASY", date(2024, 4, 21))

        assert result is None

    def test_no_signal_weeks_after(self, strategy):
        """No signal weeks after announcement."""
        result = strategy.generate_signal("CASY", date(2024, 5, 15))

        assert result is None


class TestPEADConfidenceScalesWithSurprise:
    def test_high_surprise_high_confidence(self, strategy):
        """Surprise >20% → confidence 75."""
        # CASY: 24.04% surprise
        result = strategy.generate_signal("CASY", date(2024, 4, 23))

        assert result is not None
        assert result.confidence == 75.0

    def test_medium_surprise_medium_confidence(self, strategy):
        """Surprise 10-20% → confidence 65."""
        # DECK: 12.5% surprise
        result = strategy.generate_signal("DECK", date(2024, 5, 2))

        assert result is not None
        assert result.confidence == 65.0

    def test_low_surprise_low_confidence(self, strategy):
        """Surprise 5-10% → confidence 55."""
        # CASY July: 7.14% surprise
        result = strategy.generate_signal("CASY", date(2024, 7, 26))

        assert result is not None
        assert result.confidence == 55.0

    def test_20_pct_surprise_is_high_bucket(self, strategy):
        """Exactly 20% surprise → confidence 65 (10-20 range)."""
        # TXRH: exactly 20.0% surprise
        result = strategy.generate_signal("TXRH", date(2024, 4, 19))

        assert result is not None
        assert result.confidence == 65.0


class TestPEADLogsToSignalEvents:
    def test_run_pead_logs_strategy_result(self, cache_file, tmp_path):
        """_run_pead logs the PEAD result to signal_events."""
        from orchestrator.coordinator import Coordinator
        from storage.database import Database

        db_path = str(tmp_path / "test_pead.db")
        db = Database(db_path)
        coord = Coordinator(db=db)
        coord._pead_enabled = True
        coord._pead_tickers = {"CASY"}
        coord._pead_strategy = PEADStrategy(cache_path=cache_file)

        from strategies.base import StrategyResult
        fake_result = StrategyResult(
            signal="BUY", confidence=75.0,
            strategy_name="PEAD",
            indicators={"surprise_pct": 24.04, "announce_date": "2024-04-22"},
            reasoning=["Earnings beat: +24.0% surprise on 2024-04-22"],
        )
        coord._pead_strategy.generate_signal = MagicMock(return_value=fake_result)

        result = coord._run_pead("CASY", session="PEAD_OPEN")

        assert result is not None
        assert result["signal"] == "BUY"
        assert result["strategy_name"] == "PEAD"

        # Verify it was logged
        events = coord.signal_logger.get_signals("CASY")
        pead_events = [e for e in events if e.get("strategy") == "PEAD"]
        assert len(pead_events) >= 1
        assert pead_events[0]["signal"] == "BUY"

    def test_run_pead_returns_none_when_disabled(self, cache_file):
        """_run_pead returns None when PEAD_ENABLED=False."""
        from orchestrator.coordinator import Coordinator
        from storage.database import Database

        db = Database(":memory:")
        coord = Coordinator(db=db)
        coord._pead_enabled = False

        result = coord._run_pead("CASY")
        assert result is None

    def test_run_pead_returns_none_for_non_pead_ticker(self, cache_file):
        """_run_pead returns None for tickers not in PEAD_TICKERS."""
        from orchestrator.coordinator import Coordinator
        from storage.database import Database

        db = Database(":memory:")
        coord = Coordinator(db=db)
        coord._pead_enabled = True
        coord._pead_tickers = {"CASY"}

        result = coord._run_pead("AAPL")
        assert result is None


class TestPEADBaseStrategyInterface:
    def test_analyze_returns_hold_without_beat(self, strategy):
        """analyze() returns HOLD when no recent beat."""
        import pandas as pd

        bars = pd.DataFrame({
            "Open": [100], "High": [101], "Low": [99],
            "Close": [100], "Volume": [1000],
        })
        result = strategy.analyze("ZZZZZ", bars)

        assert result.signal == "HOLD"
        assert result.strategy_name == "PEAD"
        assert result.confidence == 0.0

    def test_name_property(self, strategy):
        assert strategy.name == "PEAD"
