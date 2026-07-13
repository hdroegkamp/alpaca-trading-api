"""Smoke test for the naive Random Forest baseline.

Runs the full pipeline on one symbol: feature generation, a leakage-safe
train/val/test split, walk-forward validation, feature importance, and the
long/short strategy signal distribution.

Usage:
    python test_ml.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.trading.data.storage import DataStore
from src.trading.ml.feature_engineering import FeatureEngineering
from src.trading.ml.random_forest_analyzer import RandomForestAnalyzer
from src.trading.ml.ml_strategy import RandomForestStrategy

DATA_DIR = "data"


def load_sample():
    """Load the first available daily symbol from the data store."""
    data_store = DataStore(data_dir=DATA_DIR)
    inventory = data_store.get_inventory()
    if len(inventory) == 0:
        print("No data found. Run the download scripts first.")
        return None, None

    inv_1day = inventory[inventory["timeframe"] == "1Day"]
    if len(inv_1day) > 0:
        symbol, timeframe = inv_1day.iloc[0]["symbol"], "1Day"
    else:
        symbol = inventory.iloc[0]["symbol"]
        timeframe = inventory.iloc[0]["timeframe"]

    print(f"Loading {symbol} ({timeframe})...")
    df = data_store.load(symbol, timeframe)
    if df is None or len(df) == 0:
        print(f"No data available for {symbol}")
        return None, None
    print(f"Loaded {len(df)} rows")
    return df, symbol


def test_features(df):
    """Generate baseline features and report shape and NaN coverage."""
    print("\n" + "=" * 60)
    print("Feature Engineering")
    print("=" * 60)
    fe = FeatureEngineering()
    df_features = fe.generate_features(df)
    feature_names = fe.get_feature_names(df_features)
    print(f"Generated {len(feature_names)} features: {feature_names}")

    nan_pct = (
        df_features[feature_names].isna().sum().sum()
        / (len(df_features) * len(feature_names))
        * 100
    )
    print(f"NaN in feature block: {nan_pct:.2f}% (expected from rolling windows)")
    return df_features, fe


def test_random_forest(df_features, fe):
    """Train the RF baseline and run walk-forward validation."""
    print("\n" + "=" * 60)
    print("Random Forest")
    print("=" * 60)

    df_ml = fe.prepare_for_ml(df_features, forward_periods=1, return_threshold=0.01)
    feature_cols = fe.get_feature_names(df_ml)
    print(f"Dataset size after labelling: {len(df_ml)} rows")

    dist = df_ml["target"].value_counts()
    print(f"Target balance — up: {dist.get(1, 0)} / down: {dist.get(0, 0)}")

    rf_model = RandomForestAnalyzer(n_estimators=200, max_depth=5, min_samples_leaf=30)
    data_dict = rf_model.prepare_data(
        df_ml, feature_cols=feature_cols, test_size=0.2, val_size=0.1
    )
    print(
        f"Splits — train: {len(data_dict['y_train'])}, "
        f"val: {len(data_dict['y_val'])}, test: {len(data_dict['y_test'])}"
    )

    metrics = rf_model.train(data_dict, verbose=True)

    print("\nWalk-forward validation (honest out-of-sample estimate):")
    wf = rf_model.walk_forward_validate(
        df_ml, feature_cols=feature_cols, target_col="target", n_splits=5
    )
    print(
        f"  Mean accuracy: {wf['mean']['accuracy']:.4f} "
        f"± {wf['std']['accuracy_std']:.4f}"
    )
    print(f"  Mean F1:       {wf['mean']['f1_score']:.4f}")

    print("\nFeature importance:")
    for _, row in rf_model.get_feature_importance(top_n=10).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")

    return rf_model


def test_strategy(df_features, fe, rf_model):
    """Check the long/short strategy produces a sensible signal mix."""
    print("\n" + "=" * 60)
    print("Random Forest Strategy Signals")
    print("=" * 60)
    strategy = RandomForestStrategy(
        rf_model=rf_model,
        feature_engineer=fe,
        long_threshold=0.55,
        short_threshold=0.45,
    )
    signals = strategy.generate_signals(df_features)
    counts = signals["position"].value_counts()
    total = len(signals)
    print(
        f"  Long (1):   {counts.get(1.0, 0)} ({counts.get(1.0, 0) / total * 100:.1f}%)"
    )
    print(
        f"  Flat (0):   {counts.get(0.0, 0)} ({counts.get(0.0, 0) / total * 100:.1f}%)"
    )
    print(
        f"  Short (-1): {counts.get(-1.0, 0)} ({counts.get(-1.0, 0) / total * 100:.1f}%)"
    )
    print(f"  Mean confidence: {signals['confidence'].mean():.3f}")


def main():
    print("\n" + "=" * 60)
    print("Naive Random Forest Baseline — Smoke Test")
    print("=" * 60)

    df, symbol = load_sample()
    if df is None:
        return

    df_features, fe = test_features(df)
    rf_model = test_random_forest(df_features, fe)
    test_strategy(df_features, fe, rf_model)

    print("\n" + "=" * 60)
    print("Done. Next: streamlit run streamlit_apps/main.py -> ML Analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()
