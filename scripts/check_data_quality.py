#!/usr/bin/env python3
"""Audit the archive for unadjusted files and implausible one-day moves.

Run after a download/update and before training. Exits non-zero if any
audited symbol is not split/dividend-adjusted, so it can gate a pipeline.

Usage:
    python scripts/check_data_quality.py                         # whole archive
    python scripts/check_data_quality.py --universe universes/liquid_top100_2016.txt
    python scripts/check_data_quality.py --max-abs-return 0.3 --top 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.trading.data.quality import (
    DEFAULT_MAX_ABS_RETURN,
    audit_symbols,
    unadjusted_symbols,
)
from src.trading.data.storage import DataStore
from src.trading.data.universe import resolve_universe


def main():
    parser = argparse.ArgumentParser(description="Audit archive data quality")
    parser.add_argument(
        "--universe",
        default="inventory",
        help="UNIVERSES key, 'inventory' (default: every stored symbol), a universe "
        "file path, or a comma-separated symbol list",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="1Day")
    parser.add_argument("--max-abs-return", type=float, default=DEFAULT_MAX_ABS_RETURN)
    parser.add_argument("--top", type=int, default=25, help="Worst symbols to list")
    args = parser.parse_args()

    store = DataStore(data_dir=args.data_dir)
    symbols = resolve_universe(args.universe, data_store=store, timeframe=args.timeframe)
    audit = audit_symbols(store, symbols, args.timeframe, args.max_abs_return)

    print(f"Audited {len(audit)} of {len(symbols)} symbols in '{args.universe}'")
    print("\nAdjustment recorded in metadata:")
    print(audit["adjustment"].fillna("unknown (pre-adjustment download)").value_counts().to_string())

    suspect = audit[audit["n_suspect"] > 0].sort_values("max_abs_return", ascending=False)
    print(
        f"\n{len(suspect)} symbols with a one-day |return| > {args.max_abs_return:.0%} "
        f"({int(audit['n_suspect'].sum())} bars total)"
    )
    if not suspect.empty:
        with pd.option_context("display.width", 120):
            print(suspect.head(args.top).to_string(index=False))

    unadjusted = unadjusted_symbols(audit)
    if unadjusted:
        print(f"\nFAIL: {len(unadjusted)} symbols are not adjusted — run scripts/update_archive.py")
        sys.exit(1)
    print("\nOK: every audited symbol is split/dividend-adjusted")


if __name__ == "__main__":
    main()
