"""Unit tests for the persisted circuit-breaker state (src.trading.risk.state)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.risk.rules import RiskManager
from src.trading.risk.state import (
    evaluate_circuit_breaker,
    load_breaker_state,
    reset_breaker_state,
    save_breaker_state,
)


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "circuit_breaker.json"


@pytest.fixture
def equity_log_path(tmp_path):
    return tmp_path / "equity_log.csv"


def _write_equity_log(path, equities):
    pd.DataFrame({"equity": equities}).to_csv(path, index=False)


class TestBreakerStateRoundTrip:
    def test_default_state_is_not_triggered_when_file_missing(self, state_path):
        state = load_breaker_state(state_path)
        assert state["triggered"] is False

    def test_save_then_load_round_trips(self, state_path):
        save_breaker_state(state_path, triggered=True, reason="too much loss", triggered_at="2026-01-01T00:00:00")

        state = load_breaker_state(state_path)
        assert state["triggered"] is True
        assert state["reason"] == "too much loss"
        assert state["triggered_at"] == "2026-01-01T00:00:00"

    def test_reset_clears_a_tripped_state(self, state_path):
        save_breaker_state(state_path, triggered=True, reason="too much loss")
        reset_breaker_state(state_path)

        state = load_breaker_state(state_path)
        assert state["triggered"] is False
        assert state["reason"] is None


class TestEvaluateCircuitBreaker:
    def test_no_equity_log_never_trips(self, equity_log_path):
        risk_manager = RiskManager(max_daily_loss=0.02, max_drawdown=0.10)
        tripped, reason = evaluate_circuit_breaker(equity_log_path, 100_000.0, risk_manager)
        assert tripped is False
        assert reason is None

    def test_trips_on_daily_loss_beyond_threshold(self, equity_log_path):
        _write_equity_log(equity_log_path, [100_000.0])
        risk_manager = RiskManager(max_daily_loss=0.02, max_drawdown=0.10)

        # 3% drop in one day, beyond the 2% daily-loss limit.
        tripped, reason = evaluate_circuit_breaker(equity_log_path, 97_000.0, risk_manager)

        assert tripped is True
        assert "Daily loss" in reason

    def test_trips_on_drawdown_beyond_threshold(self, equity_log_path):
        _write_equity_log(equity_log_path, [100_000.0, 105_000.0, 104_000.0])
        risk_manager = RiskManager(max_daily_loss=0.5, max_drawdown=0.10)

        # Peak is 105_000; 93_000 is a ~11.4% drawdown from peak, beyond the 10% limit,
        # even though the day-over-day drop from the prior row (104_000) isn't itself
        # a daily-loss breach at this threshold.
        tripped, reason = evaluate_circuit_breaker(equity_log_path, 93_000.0, risk_manager)

        assert tripped is True
        assert "Drawdown" in reason

    def test_normal_day_does_not_trip(self, equity_log_path):
        _write_equity_log(equity_log_path, [100_000.0, 100_500.0])
        risk_manager = RiskManager(max_daily_loss=0.02, max_drawdown=0.10)

        tripped, reason = evaluate_circuit_breaker(equity_log_path, 100_400.0, risk_manager)

        assert tripped is False
        assert reason is None

    def test_tripped_state_persists_across_a_fresh_risk_manager_until_reset(self, state_path):
        save_breaker_state(state_path, triggered=True, reason="daily loss breach")

        # Simulate a brand-new process/invocation: fresh RiskManager, state reloaded from disk.
        RiskManager(max_daily_loss=0.02, max_drawdown=0.10)
        state = load_breaker_state(state_path)
        assert state["triggered"] is True

        reset_breaker_state(state_path)
        state = load_breaker_state(state_path)
        assert state["triggered"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
