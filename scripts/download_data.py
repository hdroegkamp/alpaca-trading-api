#!/usr/bin/env python3
"""Download historical data from Alpaca and save locally.

Usage:
    python scripts/download_data.py AAPL MSFT GOOGL --start 2020-01-01
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

from src.trading.data import AlpacaDataFetcher, DataStore
from src.trading.utils.logging import setup_logger

logger = setup_logger("download_data")


def main():
    parser = argparse.ArgumentParser(
        description="Download historical market data from Alpaca"
    )
    parser.add_argument("symbols", nargs="+", help="Trading symbols to download")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end", default=None, help="End date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--timeframe", default="1Day", help="Bar timeframe (1Day, 1Hour, etc.)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data storage directory",
    )
    parser.add_argument(
        "--organize",
        action="store_true",
        default=True,
        help="Organize data by timeframe subdirectories (default: True)",
    )
    parser.add_argument(
        "--show-inventory",
        action="store_true",
        help="Show inventory of downloaded data after completion",
    )

    args = parser.parse_args()

    logger.info(f"Downloading data for {len(args.symbols)} symbols")
    logger.info(f"Period: {args.start} to {args.end or 'today'}")
    logger.info(f"Timeframe: {args.timeframe}")

    try:
        # Initialize fetcher and storage
        fetcher = AlpacaDataFetcher()
        store = DataStore(data_dir=args.data_dir, organize_by_timeframe=args.organize)

        # Download data for each symbol
        for symbol in args.symbols:
            try:
                logger.info(f"Fetching {symbol}...")
                data = fetcher.fetch_bars(
                    symbol=symbol,
                    start=args.start,
                    end=args.end,
                    timeframe=args.timeframe,
                )

                # Save to storage
                store.save(symbol, data, timeframe=args.timeframe)
                logger.info(f"✓ {symbol}: {len(data)} bars saved")

            except Exception as e:
                logger.error(f"✗ {symbol}: {e}")
                continue

        logger.info("Download complete!")
        logger.info(f"Available symbols: {store.list_symbols()}")

        # Show inventory if requested
        if args.show_inventory:
            print("\n" + "=" * 80)
            print("DATA INVENTORY")
            print("=" * 80)
            inventory = store.get_inventory()
            if not inventory.empty:
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", None)
                print(inventory.to_string(index=False))
                print(f"\nTotal storage: {inventory['file_size_mb'].sum():.2f} MB")
            else:
                print("No data stored yet.")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

