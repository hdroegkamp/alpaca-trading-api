"""Data ingestion from Alpaca and other sources.

Bars are fetched split- and dividend-adjusted by default (``adjustment="all"``).
Raw bars turn every split into a fake one-day crash (NVDA's 10:1 split reads
as -90%), which poisons returns, features and labels alike. The catch: an
adjusted history is rewritten whenever a new split or dividend occurs, so
appending new bars to a stored adjusted file is only safe if the overlap
still matches — see :func:`src.trading.data.update.merge_new_bars`.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, cast
import os

#: Adjustment applied when none is specified. Recorded in each stored file's
#: metadata by DataStore.save so mixed raw/adjusted archives can be detected.
DEFAULT_ADJUSTMENT = "all"

_EXPECTED_COLS = ["open", "high", "low", "close", "volume"]


def _alpaca_timeframe(timeframe: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    timeframe_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, cast(TimeFrameUnit, TimeFrameUnit.Minute)),
        "15Min": TimeFrame(15, cast(TimeFrameUnit, TimeFrameUnit.Minute)),
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }
    return timeframe_map.get(timeframe, TimeFrame.Day)


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
        adjustment: str = DEFAULT_ADJUSTMENT,
    ) -> pd.DataFrame:
        """Fetch historical bar data.

        Args:
            symbol: Trading symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD), defaults to today
            timeframe: Bar timeframe (1Day, 1Hour, etc.)
            adjustment: Alpaca price adjustment — "all" (splits + dividends,
                default), "split", "dividend" or "raw"

        Returns:
            DataFrame with OHLCV data
        """
        try:
            from alpaca.data.enums import Adjustment
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
        except ImportError:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")

        # Create client
        client = StockHistoricalDataClient(self.api_key, self.api_secret)

        # Build request
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_alpaca_timeframe(timeframe),
            start=datetime.strptime(start, "%Y-%m-%d"),
            end=datetime.strptime(end, "%Y-%m-%d"),
            adjustment=Adjustment(adjustment),
        )

        # Fetch data
        bars = client.get_stock_bars(request)

        # Convert to DataFrame (alpaca client may return different objects depending on version)
        df = getattr(bars, "df", None)
        if df is None:
            # Try common fallbacks: callable conversions (to_dataframe/to_frame/to_df), iterable of bar objects/dicts, or raw attributes
            to_df_callable = None
            for name in ("to_dataframe", "to_frame", "to_df"):
                attr = getattr(bars, name, None)
                if callable(attr):
                    to_df_callable = attr
                    break

            if to_df_callable is not None:
                try:
                    df = to_df_callable()
                except Exception:
                    df = None

            if df is None:
                try:
                    df = pd.DataFrame(
                        [
                            getattr(b, "_raw", None) or getattr(b, "raw", None) or b
                            for b in bars
                        ]
                    )
                except Exception:
                    raise ValueError(
                        "Could not convert alpaca bars to DataFrame; inspect the 'bars' object returned by the client"
                    )
        # Ensure we have a DataFrame
        if not isinstance(df, pd.DataFrame):
            # cast to Any to satisfy pandas' constructor typing when df may be an arbitrary object/iterable
            df = pd.DataFrame(cast(Any, df))

        # If multi-index (multiple symbols), extract single symbol
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Ensure we have the expected columns
        for col in _EXPECTED_COLS:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Explicitly cast to DataFrame to satisfy static type checkers
        return cast(pd.DataFrame, df[_EXPECTED_COLS])

    def fetch_bars_batch(
        self,
        symbols: List[str],
        start: str,
        end: Optional[str] = None,
        timeframe: str = "1Day",
        adjustment: str = DEFAULT_ADJUSTMENT,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch bars for many symbols in one (paginated) request.

        Far fewer API calls than :meth:`fetch_multiple` for archive-sized
        symbol lists. Unlike ``fetch_multiple``, an API error (e.g. one
        invalid symbol) raises for the whole batch — callers decide whether
        to retry symbols individually.

        Args:
            symbols: Trading symbols
            start: Start date (YYYY-MM-DD)
            end: Exclusive end date (YYYY-MM-DD), defaults to today
            timeframe: Bar timeframe
            adjustment: See :meth:`fetch_bars`

        Returns:
            Dict mapping symbol -> OHLCV DataFrame; symbols with no bars in
            the range are absent.
        """
        from alpaca.data.enums import Adjustment
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest

        client = StockHistoricalDataClient(self.api_key, self.api_secret)
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        bars = client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=_alpaca_timeframe(timeframe),
                start=datetime.strptime(start, "%Y-%m-%d"),
                end=datetime.strptime(end, "%Y-%m-%d"),
                adjustment=Adjustment(adjustment),
            )
        )
        df = bars.df
        if df is None or df.empty:
            return {}
        df.columns = [c.lower() for c in df.columns]
        return {
            sym: cast(pd.DataFrame, g.droplevel("symbol")[_EXPECTED_COLS])
            for sym, g in df.groupby(level="symbol")
        }

    def fetch_multiple(
        self,
        symbols: List[str],
        start: str,
        end: Optional[str] = None,
        timeframe: str = "1Day",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.

        Args:
            symbols: List of trading symbols
            start: Start date
            end: End date
            timeframe: Bar timeframe

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = self.fetch_bars(symbol, start, end, timeframe)
                data[symbol] = df
                print(f"Fetched {len(df)} bars for {symbol}")
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        return data
