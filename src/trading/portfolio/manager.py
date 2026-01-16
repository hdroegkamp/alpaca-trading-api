"""Portfolio and position management."""

from typing import Dict, Optional


class PortfolioManager:
    """Manage portfolio positions and sizing.
    
    For vectorized backtesting, this is a simple utility class.
    For live trading, this would track actual positions and handle sizing.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """Initialize portfolio manager.
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, float] = {}
    
    def calculate_position_size(
        self,
        symbol: str,
        signal: float,
        price: float,
        max_position_pct: float = 0.1
    ) -> int:
        """Calculate position size in shares.
        
        Args:
            symbol: Trading symbol
            signal: Signal strength [-1, 1]
            price: Current price
            max_position_pct: Max % of capital per position
        
        Returns:
            Number of shares to trade (signed)
        """
        if signal == 0 or price <= 0:
            return 0
        
        # Calculate max position value
        max_position_value = self.capital * max_position_pct
        
        # Calculate target shares
        target_shares = int(abs(signal) * max_position_value / price)
        
        # Apply sign
        return target_shares if signal > 0 else -target_shares
    
    def update_capital(self, new_capital: float) -> None:
        """Update current capital.
        
        Args:
            new_capital: New capital amount
        """
        self.capital = new_capital
    
    def get_position(self, symbol: str) -> float:
        """Get current position for symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Number of shares held
        """
        return self.positions.get(symbol, 0)
    
    def set_position(self, symbol: str, shares: float) -> None:
        """Set position for symbol.
        
        Args:
            symbol: Trading symbol
            shares: Number of shares
        """
        if shares == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = shares
