"""Moving average based strategies."""

import pandas as pd
import numpy as np
from ..base import Strategy


class MovingAverageCrossover(Strategy):
    """Simple moving average crossover strategy.
    
    Goes long when fast MA crosses above slow MA, short when it crosses below.
    
    Parameters:
        fast_window: Fast moving average window (default: 20)
        slow_window: Slow moving average window (default: 50)
    """
    
    def __init__(self, fast_window: int = 20, slow_window: int = 50):
        super().__init__(fast_window=fast_window, slow_window=slow_window)
        self.fast_window = fast_window
        self.slow_window = slow_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on MA crossover.
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            DataFrame with 'position' column
        """
        signals = pd.DataFrame(index=data.index)
        
        # Calculate moving averages on close price
        signals['fast_ma'] = data['close'].rolling(window=self.fast_window).mean()
        signals['slow_ma'] = data['close'].rolling(window=self.slow_window).mean()
        
        # Generate position signals
        signals['position'] = 0.0
        signals.loc[signals['fast_ma'] > signals['slow_ma'], 'position'] = 1.0
        signals.loc[signals['fast_ma'] < signals['slow_ma'], 'position'] = -1.0
        
        return signals[['position']]


class MeanReversion(Strategy):
    """Mean reversion strategy using Bollinger Bands.
    
    Goes long when price crosses below lower band, short when above upper band.
    Exits when price returns to middle band.
    
    Parameters:
        window: Rolling window for mean and std (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(window=window, num_std=num_std)
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Bollinger Bands.
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            DataFrame with 'position' column
        """
        signals = pd.DataFrame(index=data.index)
        
        # Calculate Bollinger Bands
        rolling_mean = data['close'].rolling(window=self.window).mean()
        rolling_std = data['close'].rolling(window=self.window).std()
        
        signals['upper_band'] = rolling_mean + (rolling_std * self.num_std)
        signals['lower_band'] = rolling_mean - (rolling_std * self.num_std)
        signals['middle_band'] = rolling_mean
        
        # Generate position signals
        signals['position'] = 0.0
        
        # Long when price below lower band
        signals.loc[data['close'] < signals['lower_band'], 'position'] = 1.0
        
        # Short when price above upper band
        signals.loc[data['close'] > signals['upper_band'], 'position'] = -1.0
        
        # Forward fill to maintain positions until reversal
        signals['position'] = signals['position'].replace(0, np.nan).ffill().fillna(0)
        
        return signals[['position']]
