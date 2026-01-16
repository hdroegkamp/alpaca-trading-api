"""Risk management rules and checks."""

from typing import Optional, Tuple


class RiskManager:
    """Enforce risk limits and trading rules.

    Attributes:
        max_position_size: Maximum position size as fraction of capital
        max_daily_loss: Maximum daily loss as fraction
        max_drawdown: Maximum allowed drawdown
    """

    def __init__(
        self,
        max_position_size: float = 0.2,
        max_daily_loss: float = 0.02,
        max_drawdown: float = 0.1,
    ):
        """Initialize risk manager.

        Args:
            max_position_size: Max position as fraction of capital
            max_daily_loss: Max daily loss as fraction
            max_drawdown: Max drawdown as fraction
        """
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown

        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.circuit_breaker_triggered = False

    def check_position_size(
        self, position_value: float, portfolio_value: float
    ) -> Tuple[bool, Optional[str]]:
        """Check if position size is within limits.

        Args:
            position_value: Value of proposed position
            portfolio_value: Total portfolio value

        Returns:
            (allowed, reason) tuple
        """
        if portfolio_value <= 0:
            return False, "Portfolio value is zero or negative"

        position_pct = abs(position_value) / portfolio_value

        if position_pct > self.max_position_size:
            return (
                False,
                f"Position size {position_pct:.1%} exceeds limit {self.max_position_size:.1%}",
            )

        return True, None

    def check_daily_loss(
        self, daily_pnl: float, portfolio_value: float
    ) -> Tuple[bool, Optional[str]]:
        """Check if daily loss limit exceeded.

        Args:
            daily_pnl: Today's PnL
            portfolio_value: Current portfolio value

        Returns:
            (allowed, reason) tuple
        """
        if portfolio_value <= 0:
            return False, "Portfolio value is zero or negative"

        loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0

        if loss_pct > self.max_daily_loss:
            self.circuit_breaker_triggered = True
            return (
                False,
                f"Daily loss {loss_pct:.1%} exceeds limit {self.max_daily_loss:.1%}",
            )

        return True, None

    def check_drawdown(
        self, current_equity: float, peak_equity: float
    ) -> Tuple[bool, Optional[str]]:
        """Check if drawdown limit exceeded.

        Args:
            current_equity: Current equity
            peak_equity: Peak equity

        Returns:
            (allowed, reason) tuple
        """
        if peak_equity <= 0:
            return False, "Peak equity is zero or negative"

        drawdown = (peak_equity - current_equity) / peak_equity
        self.current_drawdown = drawdown

        if drawdown > self.max_drawdown:
            self.circuit_breaker_triggered = True
            return (
                False,
                f"Drawdown {drawdown:.1%} exceeds limit {self.max_drawdown:.1%}",
            )

        return True, None

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (use carefully)."""
        self.circuit_breaker_triggered = False
        self.daily_pnl = 0.0

    def is_trading_allowed(self) -> Tuple[bool, Optional[str]]:
        """Check if trading is currently allowed.

        Returns:
            (allowed, reason) tuple
        """
        if self.circuit_breaker_triggered:
            return False, "Circuit breaker triggered - trading halted"
        return True, None
