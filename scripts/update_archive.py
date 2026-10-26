#!/usr/bin/env python3
"""Bring the historical archive up to date with split/dividend-adjusted bars.

For each symbol:
- Not yet stored adjusted (metadata ``adjustment`` != "all", e.g. the old raw
  archive): re-download its full history adjusted.
- Already adjusted: fetch bars from a few days *before* its last stored bar.
  If the overlapping closes still match, append the new bars; if they don't,
  a split or dividend has re-adjusted the past, so re-download the full
  history (see src.trading.data.update.merge_new_bars).

So one plain run converts a raw archive, and re-running after an interruption
resumes where it stopped. Requests are batched across symbols.

Usage:
    python scripts/update_archive.py                      # every symbol in data/1Day
    python scripts/update_archive.py --symbols AAPL MSFT  # just these
    python scripts/update_archive.py --full               # force full re-download
    python scripts/update_archive.py --dry-run
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.trading.data.ingest import DEFAULT_ADJUSTMENT, AlpacaDataFetcher
from src.trading.data.storage import DataStore
from src.trading.data.update import merge_new_bars
from src.trading.utils.logging import setup_logger

logger = setup_logger("update_archive", log_file="logs/update_archive.log")


def fetch_chunked(fetcher, symbols, start, end, batch_size, delay, failed, label):
    """Yield ``(chunk, {symbol: bars})`` per batch; a failed batch is retried symbol by symbol."""
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        try:
            fetched = fetcher.fetch_bars_batch(chunk, start=start, end=end)
        except Exception as e:
            logger.warning(f"{label}: batch failed ({e}); retrying {len(chunk)} singly")
            fetched = {}
            for sym in chunk:
                try:
                    fetched.update(fetcher.fetch_bars_batch([sym], start=start, end=end))
                except Exception as e2:
                    failed.append((sym, str(e2)))
                time.sleep(delay)
        yield chunk, fetched
        logger.info(f"{label}: {i + len(chunk)}/{len(symbols)}")
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Update the archive with adjusted bars")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="1Day")
    parser.add_argument("--symbols", nargs="+", help="Only update these symbols")
    parser.add_argument(
        "--end",
        default=None,
        help="Exclusive end date YYYY-MM-DD (default today, so today's still-forming "
        "bar is never stored)",
    )
    parser.add_argument(
        "--full-start",
        default="2016-01-01",
        help="Start date for full re-downloads (default 2016-01-01, the archive's start)",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=7,
        help="Calendar days of stored history to re-fetch and compare (default 7)",
    )
    parser.add_argument("--full", action="store_true", help="Re-download every symbol in full")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.timeframe != "1Day":
        parser.error("only 1Day is supported")

    store = DataStore(data_dir=args.data_dir)
    fetcher = AlpacaDataFetcher()
    symbols = args.symbols or store.list_symbols(timeframe=args.timeframe)
    end = args.end or datetime.now().strftime("%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    full_queue = []
    by_start = defaultdict(list)
    for symbol in symbols:
        meta = store.get_metadata(symbol, args.timeframe)
        if args.full or not meta or meta.get("adjustment") != DEFAULT_ADJUSTMENT:
            full_queue.append(symbol)
            continue
        last_date = pd.Timestamp(meta["end_date"]).date()
        if last_date + timedelta(days=1) < end_date:
            by_start[last_date - timedelta(days=args.overlap_days)].append(symbol)

    n_incremental = sum(len(v) for v in by_start.values())
    logger.info(
        f"{len(symbols)} symbols: {len(full_queue)} need a full adjusted download, "
        f"{n_incremental} an incremental update (end {end}, exclusive)"
    )
    if args.dry_run:
        return

    failed = []
    appended = rows_added = up_to_date = 0

    for start, syms in sorted(by_start.items()):
        label = f"incremental from {start}"
        for chunk, fetched in fetch_chunked(
            fetcher, syms, str(start), end, args.batch_size, args.delay, failed, label
        ):
            for sym in chunk:
                if sym not in fetched:
                    up_to_date += 1
                    continue
                existing = store.load(sym, args.timeframe)
                merged = merge_new_bars(existing, fetched[sym])
                if merged is None:
                    full_queue.append(sym)  # past re-adjusted, or no overlap to check
                elif len(merged) > len(existing):
                    store.save(sym, merged, timeframe=args.timeframe, adjustment=DEFAULT_ADJUSTMENT)
                    appended += 1
                    rows_added += len(merged) - len(existing)
                else:
                    up_to_date += 1

    logger.info(
        f"Incremental: appended to {appended} symbols (+{rows_added} rows), "
        f"{up_to_date} up to date; {len(full_queue)} queued for full download"
    )

    replaced = 0
    no_data = []
    for chunk, fetched in fetch_chunked(
        fetcher, full_queue, args.full_start, end, args.batch_size, args.delay, failed, "full"
    ):
        for sym in chunk:
            if sym not in fetched:
                no_data.append(sym)
                continue
            store.save(sym, fetched[sym], timeframe=args.timeframe, adjustment=DEFAULT_ADJUSTMENT)
            replaced += 1

    logger.info("=" * 60)
    logger.info(
        f"Done: {appended} appended (+{rows_added} rows), {replaced} fully re-downloaded, "
        f"{up_to_date} already up to date"
    )
    if no_data:
        logger.warning(
            f"{len(no_data)} symbols returned no bars on full download and were left "
            f"as-is (likely delisted): {', '.join(no_data[:30])}"
            + (" ..." if len(no_data) > 30 else "")
        )
    if failed:
        logger.warning(f"{len(failed)} failed:")
        for sym, err in failed[:50]:
            logger.warning(f"  {sym}: {err}")


if __name__ == "__main__":
    main()
