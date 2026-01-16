"""Data ingestion from Alpaca and other sources."""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import os


class AlpacaDataFetcher:
    """Fetch historical data from Alpaca API.

    Attributes:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        base_url: API base URL (paper or live)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize Alpaca data fetcher.

        Args:
            api_key: API key (reads from env if not provided)
            api_secret: API secret (reads from env if not provided)
            base_url: Base URL (reads from env if not provided)
        """
        self.api_key = api_key or os.getenv("APCA_API_KEY_ID")
        self.api_secret = api_secret or os.getenv("APCA_API_SECRET_KEY")
        self.base_url = base_url or os.getenv(
            "APCA_API_BASE_URL", "https://paper-api.alpaca.markets"
        )

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "API credentials not found. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY"
            )

    def fetch_bars(
        self,
        symbol: str,
        start: str,
        end: Optional[str] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Fetch historical bar data.

        Args:
            symbol: Trading symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD), defaults to today
            timeframe: Bar timeframe (1Day, 1Hour, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")

        # Create client
        client = StockHistoricalDataClient(self.api_key, self.api_secret)

        # Parse timeframe
        timeframe_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "Minute"),
            "15Min": TimeFrame(15, "Minute"),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        tf = timeframe_map.get(timeframe, TimeFrame.Day)

        # Build request
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=datetime.strptime(start, "%Y-%m-%d"),
            end=datetime.strptime(end, "%Y-%m-%d"),
        )

        # Fetch data
        bars = client.get_stock_bars(request)

        # Convert to DataFrame
        df = bars.df

        # If multi-index (multiple symbols), extract single symbol
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Ensure we have the expected columns
        expected_cols = ["open", "high", "low", "close", "volume"]
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        return df[expected_cols]

    def fetch_multiple(
        self,
        symbols: list[str],
        start: str,
        end: Optional[str] = None,
        timeframe: str = "1Day",
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.

        Args:
            symbols: List of trading symbols
            start: Start date
            end: End date
            timeframe: Bar timeframe

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            try:
                df = self.fetch_bars(symbol, start, end, timeframe)
                data[symbol] = df
                print(f"Fetched {len(df)} bars for {symbol}")
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        return data
