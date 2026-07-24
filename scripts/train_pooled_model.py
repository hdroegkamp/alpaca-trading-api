#!/usr/bin/env python3
"""Train and save a production PooledForestModel for live scoring.

Wraps the existing panel/label/model pieces only — no new ML logic. Run this
manually or on a periodic cadence (e.g. weekly) to refresh the model that
scripts/run_live_rebalance.py loads and scores every trading day. Decoupled
from the daily rebalance cadence so that job stays fast.

Usage:
    python scripts/train_pooled_model.py --universe starter
    python scripts/train_pooled_model.py --universe AAPL,MSFT,GOOGL --model-path data/live/pooled_model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.trading.data.storage import DataStore
from src.trading.data.universe import resolve_universe
from src.trading.ml.cross_sectional_labels import add_cross_sectional_labels
from src.trading.ml.feature_engineering import FeatureEngineering
from src.trading.ml.panel import build_panel
from src.trading.ml.pooled_model import PooledForestModel
from src.trading.utils.logging import setup_logger

logger = setup_logger("train_pooled_model")


def main():
    parser = argparse.ArgumentParser(
        description="Train and save a pooled cross-sectional model for live scoring"
    )
    parser.add_argument(
        "--universe",
        default="starter",
        help="UNIVERSES key, 'inventory', a universe file path, or a comma-separated "
        "symbol list (default: starter)",
    )
    parser.add_argument("--data-dir", default="data", help="Local data directory")
    parser.add_argument("--timeframe", default="1Day", help="Bar timeframe")
    parser.add_argument(
        "--model-path",
        default="data/live/pooled_model",
        help="Output path prefix for the saved model (default: data/live/pooled_model)",
    )
    parser.add_argument(
        "--panel-cache-path", default=None, help="Optional parquet cache path for the panel"
    )
    parser.add_argument("--min-history", type=int, default=70)
    parser.add_argument("--top-quantile", type=float, default=0.2)
    parser.add_argument("--bottom-quantile", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=30)
    args = parser.parse_args()

    store = DataStore(data_dir=args.data_dir)
    symbols = resolve_universe(args.universe, data_store=store, timeframe=args.timeframe)
    if not symbols:
        logger.error(f"Universe '{args.universe}' resolved to no symbols.")
        sys.exit(1)
    logger.info(f"Universe '{args.universe}': {len(symbols)} symbols")

    panel = build_panel(
        symbols,
        store,
        timeframe=args.timeframe,
        min_history=args.min_history,
        cache_path=args.panel_cache_path,
    )
    logger.info(
        f"Panel: {panel.shape[0]:,} rows, {panel['symbol'].nunique()} symbols, "
        f"{panel['date'].min().date()} -> {panel['date'].max().date()}"
    )

    labeled = add_cross_sectional_labels(
        panel, top_quantile=args.top_quantile, bottom_quantile=args.bottom_quantile
    )
    logger.info(f"Labeled panel: {labeled.shape[0]:,} rows")

    feature_cols = FeatureEngineering.FEATURE_COLUMNS
    model = PooledForestModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    model.fit_full(labeled, feature_cols=feature_cols, target_col="target")

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    logger.info(f"Model saved to {args.model_path}_model.pkl / {args.model_path}_metadata.pkl")


if __name__ == "__main__":
    main()
