"""Unit tests for vectorized backtest engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.backtest import VectorizedBacktest
from src.trading.strategy.examples import MovingAverageCrossover


def generate_sample_data(n_days=252, start_price=100.0):
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

    # Random walk
    returns = np.random.randn(n_days) * 0.02
    close = start_price * (1 + returns).cumprod()

    data = pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n_days) * 0.005),
            "high": close * (1 + abs(np.random.randn(n_days)) * 0.01),
            "low": close * (1 - abs(np.random.randn(n_days)) * 0.01),
            "close": close,
            "volume": np.random.randint(1000000, 10000000, n_days),
        },
        index=dates,
    )

    return data


class TestVectorizedBacktest:
    """Test VectorizedBacktest class."""

    def test_initialization(self):
        """Test backtest initialization."""
        data = generate_sample_data()
        strategy = MovingAverageCrossover(fast_window=10, slow_window=30)

        backtest = VectorizedBacktest(
            strategy=strategy, data=data, initial_capital=100000.0, commission=0.001
        )

        assert backtest.initial_capital == 100000.0
        assert backtest.commission == 0.001
        assert backtest.results is None

    def test_run_backtest(self):
        """Test running backtest."""
        data = generate_sample_data()
        strategy = MovingAverageCrossover(fast_window=10, slow_window=30)

        backtest = VectorizedBacktest(
            strategy=strategy, data=data, initial_capital=100000.0
        )

        results = backtest.run()

        # Check results structure
        assert results is not None
        assert "position" in results.columns
        assert "equity" in results.columns
        assert "drawdown" in results.columns
        assert len(results) == len(data)

    def test_summary_metrics(self):
        """Test summary statistics."""
        data = generate_sample_data()
        strategy = MovingAverageCrossover(fast_window=10, slow_window=30)

        backtest = VectorizedBacktest(strategy=strategy, data=data)
        backtest.run()

        summary = backtest.get_summary()

        # Check required metrics exist
        assert "total_return" in summary
        assert "sharpe_ratio" in summary
        assert "max_drawdown" in summary
        assert "win_rate" in summary
        assert "n_trades" in summary

        # Check value ranges
        assert -1 <= summary["total_return"] <= 10  # Reasonable bounds
        assert -5 <= summary["sharpe_ratio"] <= 5
        assert -1 <= summary["max_drawdown"] <= 0
        assert 0 <= summary["win_rate"] <= 1

    def test_commission_impact(self):
        """Test that commission reduces returns."""
        data = generate_sample_data()
        strategy = MovingAverageCrossover(fast_window=10, slow_window=30)

        # Run with no commission
        bt_no_comm = VectorizedBacktest(strategy=strategy, data=data, commission=0.0)
        bt_no_comm.run()
        summary_no_comm = bt_no_comm.get_summary()

        # Run with commission
        bt_with_comm = VectorizedBacktest(strategy=strategy, data=data, commission=0.01)
        bt_with_comm.run()
        summary_with_comm = bt_with_comm.get_summary()

        # Commission should reduce returns
        assert summary_with_comm["total_return"] <= summary_no_comm["total_return"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
