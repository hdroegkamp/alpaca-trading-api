"""Performance metrics and analysis tools."""

import pandas as pd
import numpy as np
from typing import Dict


class PerformanceMetrics:
    """Calculate various performance metrics for trading strategies."""

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252
    ) -> float:
        """Calculate annualized Sharpe ratio.

        Args:
            returns: Series of period returns
            risk_free_rate: Annual risk-free rate
            periods: Number of periods per year (252 for daily)

        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / periods
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(periods) * excess_returns.mean() / excess_returns.std()

    @staticmethod
    def sortino_ratio(
        returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252
    ) -> float:
        """Calculate Sortino ratio (Sharpe using only downside volatility).

        Args:
            returns: Series of period returns
            risk_free_rate: Annual risk-free rate
            periods: Number of periods per year

        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate / periods
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        downside_std = downside_returns.std()
        return np.sqrt(periods) * excess_returns.mean() / downside_std

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        """Calculate maximum drawdown.

        Args:
            equity: Series of equity values

        Returns:
            Maximum drawdown as negative fraction
        """
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        return drawdown.min()

    @staticmethod
    def calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
        """Calculate Calmar ratio (CAGR / max drawdown).

        Args:
            returns: Series of period returns
            periods: Number of periods per year

        Returns:
            Calmar ratio
        """
        # Calculate CAGR
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / periods
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Calculate max drawdown from cumulative returns
        cum_returns = (1 + returns).cumprod()
        max_dd = PerformanceMetrics.max_drawdown(cum_returns)

        if max_dd == 0:
            return 0.0
        return cagr / abs(max_dd)

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """Calculate win rate (fraction of positive returns).

        Args:
            returns: Series of period returns

        Returns:
            Win rate in [0, 1]
        """
        non_zero = returns[returns != 0]
        if len(non_zero) == 0:
            return 0.0
        return (non_zero > 0).sum() / len(non_zero)

    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """Calculate profit factor (sum wins / sum losses).

        Args:
            returns: Series of period returns

        Returns:
            Profit factor
        """
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return np.inf if wins > 0 else 0.0
        return wins / losses

    @staticmethod
    def calculate_all(
        returns: pd.Series, equity: pd.Series, periods: int = 252
    ) -> Dict:
        """Calculate comprehensive set of metrics.

        Args:
            returns: Series of period returns
            equity: Series of equity values
            periods: Number of periods per year

        Returns:
            Dictionary of all metrics
        """
        return {
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(returns, periods=periods),
            "sortino_ratio": PerformanceMetrics.sortino_ratio(returns, periods=periods),
            "calmar_ratio": PerformanceMetrics.calmar_ratio(returns, periods=periods),
            "max_drawdown": PerformanceMetrics.max_drawdown(equity),
            "win_rate": PerformanceMetrics.win_rate(returns),
            "profit_factor": PerformanceMetrics.profit_factor(returns),
            "total_return": (equity.iloc[-1] / equity.iloc[0]) - 1,
            "volatility": returns.std() * np.sqrt(periods),
        }
