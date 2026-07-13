#!/usr/bin/env python3
"""Batch download historical data from a symbol list.

Usage:
    python scripts/batch_download.py --universe universes/sp500.txt
    python scripts/batch_download.py --data-dir data
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

from src.trading.data import AlpacaDataFetcher, DataStore
from src.trading.utils.logging import setup_logger

logger = setup_logger("batch_download")


# Predefined symbol universes
UNIVERSES = {
    "indices": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO"],
    "sectors": [
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLI",
        "XLP",
        "XLY",
        "XLU",
        "XLB",
        "XLRE",
        "XLC",
    ],
    "mega_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B"],
    "tech": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "NVDA",
        "AMD",
        "INTC",
        "CRM",
        "ORCL",
        "CSCO",
    ],
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "HES", "OXY"],
    "healthcare": [
        "UNH",
        "JNJ",
        "LLY",
        "PFE",
        "ABBV",
        "TMO",
        "MRK",
        "DHR",
        "ABT",
        "BMY",
    ],
    "starter": [
        "SPY",
        "QQQ",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "JPM",
    ],
}


def load_universe_file(filepath: str) -> List[str]:
    """Load symbols from a text file (one per line).

    Args:
        filepath: Path to universe file

    Returns:
        List of symbols
    """
    symbols = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                symbols.append(line.upper())
    return symbols


def main():
    parser = argparse.ArgumentParser(
        description="Batch download historical market data from Alpaca"
    )

    # Universe selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--universe", choices=list(UNIVERSES.keys()), help="Predefined symbol universe"
    )
    group.add_argument(
        "--universe-file", help="Path to file with symbol list (one per line)"
    )
    group.add_argument("--symbols", nargs="+", help="Space-separated list of symbols")

    # Data parameters
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end", default=None, help="End date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--timeframe", default="1Day", help="Bar timeframe (1Day, 1Hour, 15Min, etc.)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data storage directory",
    )

    # Control parameters
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (avoid rate limits)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip symbols that already have data",
    )
    parser.add_argument(
        "--show-inventory", action="store_true", help="Show inventory after completion"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )

    args = parser.parse_args()

    # Determine symbol list
    if args.universe:
        symbols = UNIVERSES[args.universe]
        logger.info(f"Using predefined universe: {args.universe}")
    elif args.universe_file:
        symbols = load_universe_file(args.universe_file)
        logger.info(f"Loaded {len(symbols)} symbols from {args.universe_file}")
    elif args.symbols:
        symbols = args.symbols
    else:
        # Default to starter universe
        symbols = UNIVERSES["starter"]
        logger.info("No universe specified, using 'starter' universe")

    logger.info(f"Symbols to download: {symbols}")
    logger.info(f"Period: {args.start} to {args.end or 'today'}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Storage: {args.data_dir}")

    if args.dry_run:
        logger.info("[DRY RUN] No data will be downloaded")
        return

    try:
        # Initialize fetcher and storage
        fetcher = AlpacaDataFetcher()
        store = DataStore(data_dir=args.data_dir, organize_by_timeframe=True)

        # Track statistics
        success_count = 0
        skip_count = 0
        error_count = 0

        # Download data for each symbol
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")

                # Check if already exists
                if args.skip_existing and store.exists(symbol, args.timeframe):
                    logger.info(f"  ↷ Skipping {symbol} (already exists)")
                    skip_count += 1
                    continue

                # Fetch data
                data = fetcher.fetch_bars(
                    symbol=symbol,
                    start=args.start,
                    end=args.end,
                    timeframe=args.timeframe,
                )

                # Save to storage
                store.save(symbol, data, timeframe=args.timeframe)
                logger.info(f"  ✓ {symbol}: {len(data)} bars saved")
                success_count += 1

                # Rate limiting delay
                if i < len(symbols):  # Don't delay after last symbol
                    time.sleep(args.delay)

            except Exception as e:
                logger.error(f"  ✗ {symbol}: {e}")
                error_count += 1
                continue

        # Summary
        logger.info("=" * 80)
        logger.info("BATCH DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ Success: {success_count}")
        logger.info(f"↷ Skipped: {skip_count}")
        logger.info(f"✗ Errors:  {error_count}")
        logger.info(f"Total: {len(symbols)}")

        # Show inventory if requested
        if args.show_inventory:
            print("\n" + "=" * 80)
            print("DATA INVENTORY")
            print("=" * 80)
            inventory = store.get_inventory()
            if not inventory.empty:
                # Show summary by timeframe
                summary = (
                    inventory.groupby("timeframe")
                    .agg(
                        {
                            "symbol": "count",
                            "rows": "sum",
                            "file_size_mb": "sum",
                            "start_date": "min",
                            "end_date": "max",
                        }
                    )
                    .round(2)
                )
                print("\nSummary by Timeframe:")
                print(summary)
                print(f"\nTotal symbols: {len(inventory)}")
                print(f"Total storage: {inventory['file_size_mb'].sum():.2f} MB")
            else:
                print("No data stored yet.")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
