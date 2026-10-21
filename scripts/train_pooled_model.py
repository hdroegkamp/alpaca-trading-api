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
from datetime import datetime
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
from src.trading.ml.promotion import (
    validate_holdout_promotion,
    validate_walk_forward_promotion,
)
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
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=60,
        help="Most-recent unique dates held out from training and backtested "
        "to validate the exact artifact before it's saved (default 60)",
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.0,
        help="Promotion bar: the holdout backtest's Sharpe ratio must exceed "
        "this or the model is not saved (default 0.0 — must be profitable "
        "out-of-sample)",
    )
    parser.add_argument(
        "--wf-splits",
        type=int,
        default=10,
        help="Expanding-window folds for the walk-forward method gate (default 10)",
    )
    parser.add_argument(
        "--wf-min-sharpe",
        type=float,
        default=0.5,
        help="Method gate: stitched walk-forward out-of-sample Sharpe, net of "
        "--commission, must exceed this (default 0.5)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Flat commission fraction on turnover for both gates' backtests (default 0.001)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Save the model even if it fails either promotion gate",
    )
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
    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
    }

    # Both gates train on the labeled panel but backtest on the full panel —
    # see src/trading/ml/promotion.py for why that distinction matters.
    wf_gate = validate_walk_forward_promotion(
        labeled,
        panel,
        feature_cols=feature_cols,
        model_params=model_params,
        n_splits=args.wf_splits,
        top_pct=args.top_quantile,
        bottom_pct=args.bottom_quantile,
        commission=args.commission,
        min_sharpe=args.wf_min_sharpe,
    )
    wf_sharpe = wf_gate["summary"]["sharpe_ratio"]
    logger.info(
        f"Walk-forward method gate ({args.wf_splits} folds, {wf_gate['n_days']} OOS dates): "
        f"sharpe={wf_sharpe:.3f} (bar: >{args.wf_min_sharpe}), "
        f"max_drawdown={wf_gate['summary']['max_drawdown']:.3%}"
    )

    gate = validate_holdout_promotion(
        labeled,
        panel,
        feature_cols=feature_cols,
        model_params=model_params,
        holdout_days=args.holdout_days,
        top_pct=args.top_quantile,
        bottom_pct=args.bottom_quantile,
        commission=args.commission,
        min_sharpe=args.min_sharpe,
    )
    sharpe = gate["summary"]["sharpe_ratio"]
    holdout_dates = gate["holdout_dates"]
    logger.info(
        f"Holdout artifact gate ({holdout_dates.min().date()} -> {holdout_dates.max().date()}, "
        f"{len(holdout_dates)} dates): sharpe={sharpe:.3f} "
        f"(bar: >{args.min_sharpe}), max_drawdown={gate['summary']['max_drawdown']:.3%}"
    )

    failures = []
    if not wf_gate["passed"]:
        failures.append(f"walk-forward Sharpe {wf_sharpe:.3f} <= {args.wf_min_sharpe}")
    if not gate["passed"]:
        failures.append(f"holdout Sharpe {sharpe:.3f} <= {args.min_sharpe}")

    if failures and not args.force:
        logger.error(
            f"Promotion gate failed ({'; '.join(failures)}) — NOT saving. "
            "Re-run with --force to override."
        )
        sys.exit(1)
    elif failures:
        logger.warning(
            f"Promotion gate failed ({'; '.join(failures)}) — saving anyway "
            "because --force was passed."
        )

    model = PooledForestModel(**model_params)
    model.fit_full(labeled, feature_cols=feature_cols, target_col="target")

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(
        args.model_path,
        extra_metadata={
            "holdout_sharpe": sharpe,
            "holdout_summary": gate["summary"],
            "holdout_start": str(holdout_dates.min().date()),
            "holdout_end": str(holdout_dates.max().date()),
            "holdout_passed": gate["passed"],
            "wf_sharpe": wf_sharpe,
            "wf_summary": wf_gate["summary"],
            "wf_splits": args.wf_splits,
            "wf_oos_days": wf_gate["n_days"],
            "wf_passed": wf_gate["passed"],
            "commission": args.commission,
            "validated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    logger.info(f"Model saved to {args.model_path}_model.pkl / {args.model_path}_metadata.pkl")


if __name__ == "__main__":
    main()
