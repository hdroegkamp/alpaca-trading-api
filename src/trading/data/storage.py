"""Data storage utilities for historical market data."""

import pandas as pd
from pathlib import Path
from typing import Optional
import os


class DataStore:
    """Manage storage and retrieval of historical market data.
    
    Stores data in parquet format for efficient I/O.
    
    Attributes:
        data_dir: Directory for storing data files
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data store.
        
        Args:
            data_dir: Directory path for data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
    
    def save(self, symbol: str, data: pd.DataFrame, timeframe: str = "1D") -> None:
        """Save historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: OHLCV DataFrame
            timeframe: Data timeframe (e.g., '1D', '1H')
        """
        filename = f"{symbol}_{timeframe}.parquet"
        filepath = self.data_dir / filename
        data.to_parquet(filepath, compression='gzip')
        print(f"Saved {len(data)} rows to {filepath}")
    
    def load(self, symbol: str, timeframe: str = "1D") -> Optional[pd.DataFrame]:
        """Load historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
        
        Returns:
            OHLCV DataFrame or None if not found
        """
        filename = f"{symbol}_{timeframe}.parquet"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"Data not found: {filepath}")
            return None
        
        data = pd.read_parquet(filepath)
        print(f"Loaded {len(data)} rows from {filepath}")
        return data
    
    def list_symbols(self) -> list[str]:
        """List all symbols with stored data.
        
        Returns:
            List of symbol strings
        """
        symbols = set()
        for file in self.data_dir.glob("*.parquet"):
            symbol = file.stem.split('_')[0]
            symbols.add(symbol)
        return sorted(list(symbols))
    
    def exists(self, symbol: str, timeframe: str = "1D") -> bool:
        """Check if data exists for symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
        
        Returns:
            True if data file exists
        """
        filename = f"{symbol}_{timeframe}.parquet"
        return (self.data_dir / filename).exists()
