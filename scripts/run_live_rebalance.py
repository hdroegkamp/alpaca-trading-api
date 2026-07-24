#!/usr/bin/env python3
"""Daily live/paper rebalance: score today's universe and adjust Alpaca positions.

Refreshes a small rolling window of bars, scores the latest cross-section
with a previously trained PooledForestModel (see train_pooled_model.py),
builds a long/short target book (src.trading.execution.target_portfolio),
and rebalances the Alpaca account toward it
(src.trading.execution.broker.AlpacaBroker).

Defaults to a dry run — pass --live to actually submit orders. Appends one
row to data/live/equity_log.csv and one row per order considered to
data/live/orders_log.csv on every run (dry or live), so performance can be
tracked and evaluated later regardless of mode.

Deliberately uses its OWN small data directory (default data/live/bars/),
separate from the main data/ historical archive used for research and
training: DataStore.save() fully overwrites a symbol's file, so refreshing
only a ~1-year rolling window into the main archive would silently truncate
its multi-year history. train_pooled_model.py trains against the full
archive; this script only needs enough recent history to compute today's
feature row.

Known simplification: order sizing uses the most recent *completed* daily
bar's close as the reference price (there is no partial bar for "today"
until the market closes), so live market-order fills will differ slightly
from the price used to size them — the same kind of simplification the
backtest's own docs call out for slippage.

Usage:
    python scripts/run_live_rebalance.py --universe starter
    python scripts/run_live_rebalance.py --universe starter --live
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.trading.data.ingest import AlpacaDataFetcher
from src.trading.data.storage import DataStore
from src.trading.data.universe import resolve_universe
from src.trading.execution.broker import AlpacaBroker
from src.trading.execution.target_portfolio import build_target_weights
from src.trading.ml.panel import build_panel
from src.trading.ml.pooled_model import PooledForestModel
from src.trading.risk.rules import RiskManager
from src.trading.utils.logging import setup_logger

logger = setup_logger("run_live_rebalance", log_file="logs/run_live_rebalance.log")

EQUITY_LOG = Path("data/live/equity_log.csv")
ORDERS_LOG = Path("data/live/orders_log.csv")


def refresh_data(symbols, store, timeframe, lookback_days):
    """Re-fetch a rolling window of bars for each symbol into ``store``."""
    fetcher = AlpacaDataFetcher()
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    for symbol in symbols:
        try:
            data = fetcher.fetch_bars(symbol, start=start, timeframe=timeframe)
            store.save(symbol, data, timeframe=timeframe)
        except Exception as e:
            logger.warning(f"Could not refresh {symbol}: {e}")


def append_log(path: Path, row: dict):
    """Append one row to a CSV log, creating it (with header) if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    parser = argparse.ArgumentParser(description="Daily live/paper cross-sectional rebalance")
    parser.add_argument(
        "--universe",
        default="starter",
        help="UNIVERSES key, 'inventory', a universe file path, or a comma-separated "
        "symbol list (default: starter)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/live/bars",
        help="Rolling-window data directory for live scoring (default: data/live/bars, "
        "kept separate from the main research archive — see module docstring)",
    )
    parser.add_argument("--timeframe", default="1Day")
    parser.add_argument("--model-path", default="data/live/pooled_model")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=250,
        help="Calendar days of history to keep refreshed (default 250, comfortably "
        "clears the 60-day rolling-volatility feature's warmup)",
    )
    parser.add_argument("--top-pct", type=float, default=0.2)
    parser.add_argument("--bottom-pct", type=float, default=0.2)
    parser.add_argument(
        "--capital-fraction",
        type=float,
        default=0.5,
        help="Fraction of account equity to deploy (default 0.5 for this first live pass)",
    )
    parser.add_argument("--min-trade-dollars", type=float, default=50.0)
    parser.add_argument("--max-position-pct", type=float, default=0.2)
    parser.add_argument(
        "--live", action="store_true", help="Actually submit orders (default: dry run)"
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip re-downloading bars and use whatever is already in --data-dir",
    )
    args = parser.parse_args()

    mode = "LIVE" if args.live else "DRY RUN"
    logger.info(f"=== run_live_rebalance ({mode}) ===")

    store = DataStore(data_dir=args.data_dir)
    symbols = resolve_universe(args.universe, data_store=store, timeframe=args.timeframe)
    logger.info(f"Universe '{args.universe}': {len(symbols)} symbols")

    broker = AlpacaBroker()

    if not broker.is_market_open():
        logger.info("Market is closed — skipping this run.")
        return

    if not args.skip_refresh:
        refresh_data(symbols, store, args.timeframe, args.lookback_days)

    panel = build_panel(symbols, store, timeframe=args.timeframe, min_history=70)
    latest_date = panel["date"].max()
    today_slice = panel[panel["date"] == latest_date].copy()
    logger.info(f"Scoring {len(today_slice)} symbols for {latest_date.date()}")

    model = PooledForestModel()
    model.load(args.model_path)
    scored = model.predict_scores(today_slice)
    scored = scored.merge(today_slice[["symbol", "close"]], on="symbol", how="left")

    target_weights = build_target_weights(
        scored, top_pct=args.top_pct, bottom_pct=args.bottom_pct
    )
    scaled_weights = {sym: w * args.capital_fraction for sym, w in target_weights.items()}
    prices = dict(zip(scored["symbol"], scored["close"]))

    broker_equity = broker.get_equity()
    risk_manager = RiskManager(max_position_size=args.max_position_pct)

    orders = broker.submit_target_weights(
        scaled_weights,
        prices,
        dry_run=not args.live,
        min_trade_dollars=args.min_trade_dollars,
        risk_manager=risk_manager,
    )

    n_long = sum(1 for w in target_weights.values() if w > 0)
    n_short = sum(1 for w in target_weights.values() if w < 0)
    timestamp = datetime.now().isoformat(timespec="seconds")

    append_log(
        EQUITY_LOG,
        {
            "timestamp": timestamp,
            "signal_date": str(latest_date.date()),
            "equity": broker_equity,
            "cash": broker.get_cash(),
            "n_long": n_long,
            "n_short": n_short,
            "mode": mode,
        },
    )
    for order in orders:
        append_log(ORDERS_LOG, {"timestamp": timestamp, "signal_date": str(latest_date.date()), **order})

    logger.info(f"Equity: ${broker_equity:,.2f} | long={n_long} short={n_short} | mode={mode}")
    for order in orders:
        if order["status"] != "no_change":
            logger.info(f"  {order['symbol']}: {order['status']} — {order['detail']}")


if __name__ == "__main__":
    main()
