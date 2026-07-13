"""Smoke test for the pooled cross-sectional stock-picking pipeline.

Runs the full pooled pipeline end-to-end on a small local subset: panel
assembly, cross-sectional labeling, walk-forward validation, and the
portfolio backtest.

Usage:
    python test_pooled_ml.py [N_SYMBOLS]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.trading.data.storage import DataStore
from src.trading.ml.panel import build_panel
from src.trading.ml.cross_sectional_labels import add_cross_sectional_labels
from src.trading.ml.pooled_model import PooledForestModel
from src.trading.backtest.portfolio import PortfolioBacktest

DATA_DIR = "data"
DEFAULT_N_SYMBOLS = 30


def load_symbols(n_symbols):
    """Pick a small, fast-to-iterate subset of locally available symbols."""
    data_store = DataStore(data_dir=DATA_DIR)
    symbols = data_store.list_symbols(timeframe="1Day")
    if not symbols:
        print("No data found. Run the download scripts first.")
        return None, None
    return data_store, sorted(symbols)[:n_symbols]


def build_and_label(data_store, symbols):
    print("\n" + "=" * 60)
    print(f"Panel Assembly ({len(symbols)} symbols)")
    print("=" * 60)

    panel = build_panel(symbols, data_store, timeframe="1Day", min_history=70)
    print(f"Panel shape: {panel.shape}")
    print(f"Date range: {panel['date'].min()} -> {panel['date'].max()}")
    print(f"Symbols retained: {panel['symbol'].nunique()} / {len(symbols)}")

    labeled = add_cross_sectional_labels(panel, top_quantile=0.2, bottom_quantile=0.2)
    print(f"Labeled panel shape: {labeled.shape}")
    dist = labeled["target"].value_counts()
    print(f"Target balance — top: {dist.get(1, 0)} / bottom: {dist.get(0, 0)}")
    return panel, labeled


def train_and_validate(labeled):
    print("\n" + "=" * 60)
    print("Pooled Model — Walk-Forward Validation")
    print("=" * 60)

    from src.trading.ml.feature_engineering import FeatureEngineering

    feature_cols = FeatureEngineering.FEATURE_COLUMNS
    model = PooledForestModel(n_estimators=200, max_depth=5, min_samples_leaf=30)
    wf = model.walk_forward_validate_panel(
        labeled, feature_cols=feature_cols, target_col="target", n_splits=5
    )
    print(
        f"Mean accuracy: {wf['mean']['accuracy']:.4f} ± {wf['std']['accuracy_std']:.4f}"
    )
    print(f"Mean F1:       {wf['mean']['f1_score']:.4f}")
    for f in wf["fold_metrics"]:
        print(
            f"  Fold {f['fold']}: train={f['train_size']:6d} test={f['test_size']:5d} "
            f"acc={f['accuracy']:.4f}"
        )
    return model, wf, feature_cols


def run_portfolio_backtest(panel, wf, feature_cols):
    print("\n" + "=" * 60)
    print("Portfolio Backtest")
    print("=" * 60)

    bt = PortfolioBacktest(
        panel=panel,
        fold_models=wf["fold_metrics"],
        feature_columns=feature_cols,
        top_pct=0.2,
        bottom_pct=0.2,
        commission=0.001,
    )
    results = bt.run()
    print(f"Rebalance days: {len(results)}")
    print(f"Final equity: ${results['equity'].iloc[-1]:,.2f}")

    summary = bt.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


def main():
    n_symbols = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_SYMBOLS

    print("\n" + "=" * 60)
    print("Pooled Cross-Sectional Pipeline — Smoke Test")
    print("=" * 60)

    data_store, symbols = load_symbols(n_symbols)
    if data_store is None:
        return

    panel, labeled = build_and_label(data_store, symbols)
    model, wf, feature_cols = train_and_validate(labeled)
    run_portfolio_backtest(panel, wf, feature_cols)

    print("\n" + "=" * 60)
    print("Done. Next: streamlit run streamlit_apps/main.py -> Portfolio ML")
    print("=" * 60)


if __name__ == "__main__":
    main()
