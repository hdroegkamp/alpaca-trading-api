"""Unit tests for strategy base classes."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.strategy.base import Strategy, Signal
from src.trading.strategy.examples import MovingAverageCrossover, MeanReversion


def generate_sample_data(n_days=100, start_price=100.0):
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Random walk for price
    returns = np.random.randn(n_days) * 0.02
    close = start_price * (1 + returns).cumprod()
    
    # Simple OHLC from close
    data = pd.DataFrame({
        'open': close * (1 + np.random.randn(n_days) * 0.005),
        'high': close * (1 + abs(np.random.randn(n_days)) * 0.01),
        'low': close * (1 - abs(np.random.randn(n_days)) * 0.01),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)
    
    return data


class TestSignal:
    """Test Signal dataclass."""
    
    def test_valid_signal(self):
        """Test creating valid signal."""
        signal = Signal(symbol='AAPL', position=0.5, confidence=0.8)
        assert signal.symbol == 'AAPL'
        assert signal.position == 0.5
        assert signal.confidence == 0.8
    
    def test_invalid_position(self):
        """Test that invalid position raises error."""
        with pytest.raises(ValueError):
            Signal(symbol='AAPL', position=1.5)
    
    def test_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError):
            Signal(symbol='AAPL', position=0.5, confidence=1.5)


class TestMovingAverageCrossover:
    """Test MovingAverageCrossover strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MovingAverageCrossover(fast_window=10, slow_window=30)
        assert strategy.fast_window == 10
        assert strategy.slow_window == 30
    
    def test_generate_signals(self):
        """Test signal generation."""
        data = generate_sample_data(n_days=100)
        strategy = MovingAverageCrossover(fast_window=10, slow_window=30)
        
        signals = strategy.generate_signals(data)
        
        # Check output structure
        assert 'position' in signals.columns
        assert len(signals) == len(data)
        
        # Check position values are valid
        assert signals['position'].isin([-1.0, 0.0, 1.0]).all()
    
    def test_signals_with_trend(self):
        """Test signals on trending data."""
        # Create uptrending data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        close = 100 + np.arange(100) * 0.5  # Linear uptrend
        
        data = pd.DataFrame({
            'open': close,
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
            'volume': 1000000
        }, index=dates)
        
        strategy = MovingAverageCrossover(fast_window=5, slow_window=20)
        signals = strategy.generate_signals(data)
        
        # In uptrend, fast MA should be above slow MA -> long positions
        # Check last signal after MAs have stabilized
        assert signals['position'].iloc[-1] == 1.0


class TestMeanReversion:
    """Test MeanReversion strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MeanReversion(window=20, num_std=2.0)
        assert strategy.window == 20
        assert strategy.num_std == 2.0
    
    def test_generate_signals(self):
        """Test signal generation."""
        data = generate_sample_data(n_days=100)
        strategy = MeanReversion(window=20, num_std=2.0)
        
        signals = strategy.generate_signals(data)
        
        # Check output structure
        assert 'position' in signals.columns
        assert len(signals) == len(data)
        
        # Check position values are valid
        assert signals['position'].isin([-1.0, 0.0, 1.0]).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
