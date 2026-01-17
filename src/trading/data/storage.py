"""Data storage utilities for historical market data."""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import os
import json
from datetime import datetime


class DataStore:
    """Manage storage and retrieval of historical market data.

    Stores data in parquet format for efficient I/O with organized directory structure.

    Attributes:
        data_dir: Root directory for storing data files
        organize_by_timeframe: Whether to organize files by timeframe subdirectories
    """

    def __init__(self, data_dir: str = "data", organize_by_timeframe: bool = True):
        """Initialize data store.

        Args:
            data_dir: Root directory path for data storage
            organize_by_timeframe: If True, organize data in subdirectories by timeframe
        """
        self.data_dir = Path(data_dir)
        self.organize_by_timeframe = organize_by_timeframe
        self.data_dir.mkdir(exist_ok=True, parents=True)

        # Create metadata directory
        self.metadata_dir = self.data_dir / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)

    def save(self, symbol: str, data: pd.DataFrame, timeframe: str = "1D") -> None:
        """Save historical data for a symbol.

        Args:
            symbol: Trading symbol
            data: OHLCV DataFrame
            timeframe: Data timeframe (e.g., '1D', '1H')
        """
        # Determine directory structure
        if self.organize_by_timeframe:
            target_dir = self.data_dir / timeframe
            target_dir.mkdir(exist_ok=True, parents=True)
            filename = f"{symbol}.parquet"
        else:
            target_dir = self.data_dir
            filename = f"{symbol}_{timeframe}.parquet"

        filepath = target_dir / filename
        data.to_parquet(filepath, compression="gzip")

        # Save metadata
        self._save_metadata(symbol, data, timeframe, filepath)

        print(f"Saved {len(data)} rows to {filepath}")

    def load(self, symbol: str, timeframe: str = "1D") -> Optional[pd.DataFrame]:
        """Load historical data for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            OHLCV DataFrame or None if not found
        """
        # Determine file path based on organization
        if self.organize_by_timeframe:
            filepath = self.data_dir / timeframe / f"{symbol}.parquet"
        else:
            filepath = self.data_dir / f"{symbol}_{timeframe}.parquet"

        if not filepath.exists():
            print(f"Data not found: {filepath}")
            return None

        data = pd.read_parquet(filepath)
        print(f"Loaded {len(data)} rows from {filepath}")
        return data

    def list_symbols(self, timeframe: Optional[str] = None) -> List[str]:
        """List all symbols with stored data.

        Args:
            timeframe: Optional timeframe to filter by

        Returns:
            List of symbol strings
        """
        symbols = set()

        if self.organize_by_timeframe:
            # Search in timeframe subdirectories
            if timeframe:
                search_dirs = [self.data_dir / timeframe]
            else:
                search_dirs = [
                    d
                    for d in self.data_dir.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]

            for dir_path in search_dirs:
                if dir_path.exists():
                    for file in dir_path.glob("*.parquet"):
                        symbols.add(file.stem)
        else:
            # Original flat structure
            for file in self.data_dir.glob("*.parquet"):
                symbol = file.stem.split("_")[0]
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
        if self.organize_by_timeframe:
            filepath = self.data_dir / timeframe / f"{symbol}.parquet"
        else:
            filepath = self.data_dir / f"{symbol}_{timeframe}.parquet"
        return filepath.exists()

    def _save_metadata(
        self, symbol: str, data: pd.DataFrame, timeframe: str, filepath: Path
    ) -> None:
        """Save metadata about the dataset.

        Args:
            symbol: Trading symbol
            data: OHLCV DataFrame
            timeframe: Data timeframe
            filepath: Path where data was saved
        """
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": len(data),
            "start_date": str(data.index.min()),
            "end_date": str(data.index.max()),
            "filepath": str(filepath),
            "file_size_mb": float(filepath.stat().st_size / (1024 * 1024)),
            "last_updated": datetime.now().isoformat(),
            "columns": list(data.columns),
        }

        metadata_file = self.metadata_dir / f"{symbol}_{timeframe}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_metadata(self, symbol: str, timeframe: str = "1D") -> Optional[Dict]:
        """Get metadata for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_file = self.metadata_dir / f"{symbol}_{timeframe}.json"
        if not metadata_file.exists():
            return None

        with open(metadata_file, "r") as f:
            return json.load(f)

    def get_inventory(self) -> pd.DataFrame:
        """Get inventory of all stored data.

        Returns:
            DataFrame with metadata for all stored symbols
        """
        inventory = []
        for metadata_file in self.metadata_dir.glob("*.json"):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                inventory.append(metadata)

        if not inventory:
            return pd.DataFrame()

        df = pd.DataFrame(inventory)
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        df["last_updated"] = pd.to_datetime(df["last_updated"])
        return df.sort_values(["timeframe", "symbol"])
