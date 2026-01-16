"""Vectorized backtesting engine for fast strategy evaluation."""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from ..strategy.base import Strategy


class VectorizedBacktest:
    """Fast vectorized backtesting engine.
    
    Executes strategy on historical data using vectorized operations for speed.
    Assumes end-of-bar execution and simple fill model.
    
    Attributes:
        strategy: Strategy instance to backtest
        data: Historical OHLCV data
        initial_capital: Starting capital (default: 100000)
        commission: Commission per trade as fraction (default: 0.001 = 0.1%)
    """
    
    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        commission: float = 0.001
    ):
        """Initialize backtest.
        
        Args:
            strategy: Strategy to test
            data: OHLCV DataFrame with DatetimeIndex
            initial_capital: Starting capital
            commission: Commission rate per trade
        """
        self.strategy = strategy
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.results: Optional[pd.DataFrame] = None
    
    def run(self) -> pd.DataFrame:
        """Run the backtest.
        
        Returns:
            DataFrame with positions, returns, equity curve, and drawdown
        """
        # Generate signals from strategy
        signals = self.strategy.generate_signals(self.data)
        
        # Merge signals with price data
        df = self.data.join(signals, how='left')
        df['position'] = df['position'].fillna(0)
        
        # Calculate position changes (trades)
        df['position_change'] = df['position'].diff()
        
        # Calculate returns (assume we trade at close of each bar)
        # Use next bar's return as our realized return
        df['market_return'] = df['close'].pct_change()
        df['strategy_return'] = df['position'].shift(1) * df['market_return']
        
        # Apply commission on position changes
        df['commission_paid'] = df['position_change'].abs() * self.commission
        df['strategy_return'] = df['strategy_return'] - df['commission_paid']
        
        # Calculate cumulative returns and equity
        df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
        df['equity'] = self.initial_capital * df['cumulative_return']
        
        # Calculate drawdown
        df['cumulative_max'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cumulative_max']) / df['cumulative_max']
        
        self.results = df
        return df
    
    def get_summary(self) -> Dict:
        """Get summary statistics of backtest.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.results is None:
            raise ValueError("Run backtest first by calling run()")
        
        df = self.results
        
        # Filter out NaN returns
        returns = df['strategy_return'].dropna()
        
        # Calculate metrics
        total_return = (df['equity'].iloc[-1] / self.initial_capital) - 1
        
        # Annualize assuming 252 trading days
        n_days = len(df)
        n_years = n_days / 252.0
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        max_drawdown = df['drawdown'].min()
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_days = len(returns[returns != 0])
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # Number of trades (position changes)
        n_trades = (df['position_change'] != 0).sum()
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': df['equity'].iloc[-1],
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'n_days': n_days,
            'strategy': str(self.strategy)
        }
    
    def plot(self, figsize=(12, 8)):
        """Plot equity curve and drawdown.
        
        Args:
            figsize: Figure size tuple
        """
        if self.results is None:
            raise ValueError("Run backtest first by calling run()")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return
        
        df = self.results
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Equity curve
        ax1.plot(df.index, df['equity'], label='Strategy Equity', linewidth=1.5)
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_ylabel('Equity ($)')
        ax1.set_title(f'Backtest Results: {self.strategy}')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Drawdown
        ax2.fill_between(df.index, df['drawdown'] * 100, 0, alpha=0.3, color='red')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
