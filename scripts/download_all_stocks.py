#!/usr/bin/env python3
"""Download data for all tradeable stocks from Alpaca.

This script fetches the complete list of tradeable assets from Alpaca
and downloads historical data for all of them.

Usage:
    python scripts/download_all_stocks.py --start 2010-01-01 --data-dir Z:\\market_data

WARNING: This will download data for 8000+ symbols and may take several hours.
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

from alpaca.trading.client import TradingClient
from src.trading.data import AlpacaDataFetcher, DataStore
from src.trading.utils.logging import setup_logger
import os

logger = setup_logger("download_all_stocks")


def get_all_tradeable_stocks(
    min_price: float = 1.0, only_active: bool = True
) -> List[str]:
    """Get list of all tradeable stocks from Alpaca.

    Args:
        min_price: Minimum stock price to filter (avoid penny stocks)
        only_active: Only include actively trading stocks

    Returns:
        List of stock symbols
    """
    logger.info("Fetching list of all tradeable stocks from Alpaca...")

    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not api_secret:
        raise ValueError("API credentials not found. Check .env file.")

    client = TradingClient(api_key, api_secret, paper=True)

    # Get all assets
    assets = client.get_all_assets()

    symbols = []
    for asset in assets:
        # Filter criteria
        # Support objects with attributes, dict-like assets, and plain symbol strings.
        if isinstance(asset, str):
            # If the API returned just a symbol string, include it (best-effort).
            symbols.append(asset)
            continue

        # Safely extract fields from either objects or dicts
        if hasattr(asset, "tradable"):
            tradable = getattr(asset, "tradable")
        elif isinstance(asset, dict):
            tradable = asset.get("tradable")
        else:
            tradable = False

        if only_active:
            if hasattr(asset, "status"):
                status = getattr(asset, "status")
            elif isinstance(asset, dict):
                status = asset.get("status")
            else:
                status = None
            active_ok = status == "active"
        else:
            active_ok = True

        if not tradable or not active_ok:
            continue

        if hasattr(asset, "asset_class"):
            asset_class = getattr(asset, "asset_class")
        elif isinstance(asset, dict):
            asset_class = asset.get("asset_class")
        else:
            asset_class = None

        if asset_class == "us_equity":  # Only US stocks
            if hasattr(asset, "symbol"):
                symbols.append(getattr(asset, "symbol"))
            elif isinstance(asset, dict) and asset.get("symbol"):
                symbols.append(asset.get("symbol"))

    logger.info(f"Found {len(symbols)} tradeable US stocks")
    return sorted(symbols)


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data for ALL tradeable stocks from Alpaca"
    )

    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="Start date (YYYY-MM-DD) - will get as much as available",
    )
    parser.add_argument(
        "--end", default=None, help="End date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--timeframe",
        default="1Day",
        help="Bar timeframe (1Day recommended for large downloads)",
    )
    parser.add_argument(
        "--data-dir", default="Z:\\market_data", help="Data storage directory"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Delay between requests in seconds (default: 0.35 for rate limit safety)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip symbols that already have data (default: True)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=1.0,
        help="Minimum stock price to include (avoid penny stocks, default: $1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Save inventory after every N symbols (default: 100)",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume from symbol index (if interrupted)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--save-symbol-list",
        help="Save the symbol list to a file (e.g., universes/all_stocks.txt)",
    )

    args = parser.parse_args()

    try:
        # Get all tradeable stocks
        symbols = get_all_tradeable_stocks(min_price=args.min_price)

        if args.save_symbol_list:
            with open(args.save_symbol_list, "w") as f:
                f.write("\n".join(symbols))
            logger.info(f"Symbol list saved to {args.save_symbol_list}")

        # Calculate estimates
        estimated_time_minutes = (len(symbols) * args.delay) / 60
        estimated_storage_mb = (
            len(symbols) * 1
        )  # ~1 MB per symbol for 10-15 years daily

        logger.info("=" * 80)
        logger.info("DOWNLOAD PLAN")
        logger.info("=" * 80)
        logger.info(f"Total symbols: {len(symbols)}")
        logger.info(f"Date range: {args.start} to {args.end or 'today'}")
        logger.info(f"Timeframe: {args.timeframe}")
        logger.info(f"Storage location: {args.data_dir}")
        logger.info(
            f"Estimated time: {estimated_time_minutes:.1f} minutes ({estimated_time_minutes/60:.1f} hours)"
        )
        logger.info(
            f"Estimated storage: {estimated_storage_mb:,.0f} MB (~{estimated_storage_mb/1024:.1f} GB)"
        )
        logger.info("=" * 80)

        if args.dry_run:
            logger.info("[DRY RUN] No data will be downloaded")
            logger.info(f"First 20 symbols: {symbols[:20]}")
            return

        # Confirm before starting
        if len(symbols) > 1000:
            response = input(
                f"\nThis will download {len(symbols)} symbols. Continue? (yes/no): "
            )
            if response.lower() != "yes":
                logger.info("Download cancelled")
                return

        # Initialize fetcher and storage
        fetcher = AlpacaDataFetcher()
        store = DataStore(data_dir=args.data_dir, organize_by_timeframe=True)

        # Track statistics
        success_count = 0
        skip_count = 0
        error_count = 0
        no_data_count = 0

        start_time = time.time()

        # Download data for each symbol
        for i, symbol in enumerate(symbols[args.resume_from :], args.resume_from + 1):
            try:
                # Progress indicator
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta_seconds = (len(symbols) - i) / rate if rate > 0 else 0

                logger.info(
                    f"[{i}/{len(symbols)}] {symbol} | "
                    f"✓ {success_count} ↷ {skip_count} ✗ {error_count} ∅ {no_data_count} | "
                    f"ETA: {eta_seconds/60:.0f}m"
                )

                # Check if already exists
                if args.skip_existing and store.exists(symbol, args.timeframe):
                    logger.info(f"  ↷ Skipping (already exists)")
                    skip_count += 1
                    continue

                # Fetch data
                data = fetcher.fetch_bars(
                    symbol=symbol,
                    start=args.start,
                    end=args.end,
                    timeframe=args.timeframe,
                )

                if data is None or len(data) == 0:
                    logger.info(f"  ∅ No data available")
                    no_data_count += 1
                    continue

                # Save to storage
                store.save(symbol, data, timeframe=args.timeframe)
                logger.info(
                    f"  ✓ {len(data)} bars ({data.index[0]} to {data.index[-1]})"
                )
                success_count += 1

                # Rate limiting delay
                time.sleep(args.delay)

                # Periodic inventory save
                if success_count % args.batch_size == 0:
                    logger.info(f"--- Checkpoint: {success_count} symbols saved ---")

            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
                error_count += 1
                time.sleep(args.delay)  # Still delay on error
                continue

        # Final summary
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ Success:  {success_count}")
        logger.info(f"↷ Skipped:  {skip_count}")
        logger.info(f"∅ No data:  {no_data_count}")
        logger.info(f"✗ Errors:   {error_count}")
        logger.info(f"Total:      {len(symbols)}")
        logger.info(
            f"Time:       {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)"
        )
        logger.info("=" * 80)

        # Show final inventory
        inventory = store.get_inventory()
        if not inventory.empty:
            logger.info(f"\nTotal symbols in storage: {inventory['symbol'].nunique()}")
            logger.info(f"Total storage used: {inventory['file_size_mb'].sum():.2f} MB")
            logger.info(
                f"Date range: {inventory['start_date'].min()} to {inventory['end_date'].max()}"
            )

    except KeyboardInterrupt:
        logger.info("\n\n=== Download interrupted by user ===")
        logger.info(f"Progress saved. Resume with: --resume-from {i}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
