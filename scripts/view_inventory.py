#!/usr/bin/env python3
"""View inventory of stored market data.

Usage:
    python scripts/view_inventory.py --data-dir Z:\\market_data
    python scripts/view_inventory.py --data-dir Z:\\market_data --timeframe 1Day
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading.data import DataStore
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="View data inventory")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data storage directory (e.g., Z:\\market_data)",
    )
    parser.add_argument(
        "--timeframe", help="Filter by specific timeframe (1Day, 1Hour, etc.)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information for each symbol",
    )

    args = parser.parse_args()

    try:
        store = DataStore(data_dir=args.data_dir, organize_by_timeframe=True)
        inventory = store.get_inventory()

        if inventory.empty:
            print(f"No data found in {args.data_dir}")
            print("\nTo download data, run:")
            print(
                f'  .\\venv\\Scripts\\python.exe scripts\\batch_download.py --data-dir "{args.data_dir}"'
            )
            return

        # Filter by timeframe if specified
        if args.timeframe:
            inventory = inventory[inventory["timeframe"] == args.timeframe]
            if inventory.empty:
                print(f"No data found for timeframe: {args.timeframe}")
                return

        print("=" * 100)
        print("DATA INVENTORY")
        print("=" * 100)

        if args.detailed:
            # Show detailed view
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_rows", None)
            print(inventory.to_string(index=False))
        else:
            # Show summary by timeframe
            print("\nSummary by Timeframe:")
            print("-" * 100)
            summary = (
                inventory.groupby("timeframe")
                .agg(
                    {
                        "symbol": "count",
                        "rows": ["sum", "mean"],
                        "file_size_mb": "sum",
                        "start_date": "min",
                        "end_date": "max",
                    }
                )
                .round(2)
            )
            summary.columns = [
                "Symbols",
                "Total Bars",
                "Avg Bars",
                "Storage (MB)",
                "Earliest Date",
                "Latest Date",
            ]
            print(summary)

            print("\n" + "-" * 100)
            print("Symbols by Timeframe:")
            print("-" * 100)
            for tf in sorted(inventory["timeframe"].unique()):
                symbols = inventory[inventory["timeframe"] == tf]["symbol"].tolist()
                print(f"\n{tf} ({len(symbols)} symbols):")
                # Print in columns
                for i in range(0, len(symbols), 10):
                    print("  " + ", ".join(symbols[i : i + 10]))

        print("\n" + "=" * 100)
        print("TOTALS")
        print("=" * 100)
        print(f"Unique symbols: {inventory['symbol'].nunique()}")
        print(f"Total datasets: {len(inventory)}")
        print(f"Total bars: {inventory['rows'].sum():,.0f}")
        print(f"Total storage: {inventory['file_size_mb'].sum():.2f} MB")
        print(
            f"Date range: {inventory['start_date'].min()} to {inventory['end_date'].max()}"
        )
        print(f"Storage location: {args.data_dir}")
        print("=" * 100)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
