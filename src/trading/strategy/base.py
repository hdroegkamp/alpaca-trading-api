"""Base strategy interface and signal definitions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd


@dataclass
class Signal:
    """Trading signal with position and metadata.

    Attributes:
        symbol: Trading symbol
        position: Target position (-1 for short, 0 for flat, 1 for long)
        confidence: Optional confidence score [0, 1]
        metadata: Optional metadata dict for analysis
    """

    symbol: str
    position: float  # -1 (short), 0 (flat), 1 (long) or fractional
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Validate signal values."""
        if not -1 <= self.position <= 1:
            raise ValueError(f"Position must be in [-1, 1], got {self.position}")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


class Strategy(ABC):
    """Abstract base class for trading strategies.

    Subclasses must implement generate_signals() to produce trading signals
    based on historical price data.
    """

    def __init__(self, **params):
        """Initialize strategy with parameters.

        Args:
            **params: Strategy-specific parameters
        """
        self.params = params
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from price data.

        Args:
            data: DataFrame with OHLCV data, indexed by timestamp.
                  Columns: open, high, low, close, volume

        Returns:
            DataFrame with same index as input, containing 'position' column
            with values in [-1, 1] indicating target position.
        """
        pass

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"
