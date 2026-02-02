"""Test script for ML models with sample data.

Tests LSTM and Random Forest models on a sample stock.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.trading.data.storage import DataStore
from src.trading.ml.feature_engineering import FeatureEngineering
from src.trading.ml.lstm_predictor import LSTMPredictor
from src.trading.ml.random_forest_analyzer import RandomForestAnalyzer
from src.trading.ml.ml_strategy import MLPredictionStrategy


def test_feature_engineering():
    """Test feature engineering on sample data."""
    print("\n" + "=" * 60)
    print("Testing Feature Engineering")
    print("=" * 60)

    # Load sample data
    data_store = DataStore(data_dir="Z:\\market_data")
    inventory = data_store.get_inventory()

    if len(inventory) == 0:
        print("No data found. Please run download scripts first.")
        return None

    # Use first available symbol (prefer 1Day timeframe)
    inventory_1day = inventory[inventory["timeframe"] == "1Day"]
    if len(inventory_1day) == 0:
        symbol = inventory.iloc[0]["symbol"]
        timeframe = inventory.iloc[0]["timeframe"]
    else:
        symbol = inventory_1day.iloc[0]["symbol"]
        timeframe = "1Day"
    print(f"\nLoading data for {symbol} ({timeframe})...")

    df = data_store.load(symbol, timeframe)

    if df is None or len(df) == 0:
        print(f"No data available for {symbol}")
        return None

    print(f"Loaded {len(df)} rows")

    # Generate features
    print("\nGenerating features...")
    fe = FeatureEngineering()
    df_features = fe.generate_features(df)

    feature_names = fe.get_feature_names(df_features)
    print(f"Generated {len(feature_names)} features")
    print(f"Sample features: {feature_names[:10]}")

    # Check for NaN
    nan_count = df_features[feature_names].isna().sum().sum()
    print(
        f"\nNaN values: {nan_count} ({nan_count / (len(df_features) * len(feature_names)) * 100:.2f}%)"
    )

    return df_features, fe


def test_random_forest(df_features, fe):
    """Test Random Forest model."""
    print("\n" + "=" * 60)
    print("Testing Random Forest")
    print("=" * 60)

    # Prepare data
    print("\nPreparing data for classification...")
    df_ml = fe.prepare_for_ml(df_features, forward_periods=1)
    print(f"Dataset size: {len(df_ml)} rows")

    # Target distribution
    target_dist = df_ml["target"].value_counts()
    print(f"\nTarget distribution:")
    print(
        f"  Down (0): {target_dist.get(0, 0)} ({target_dist.get(0, 0) / len(df_ml) * 100:.1f}%)"
    )
    print(
        f"  Up (1):   {target_dist.get(1, 0)} ({target_dist.get(1, 0) / len(df_ml) * 100:.1f}%)"
    )

    # Initialize and train
    print("\nInitializing Random Forest...")
    rf_model = RandomForestAnalyzer(
        n_estimators=100, max_depth=10, random_state=42  # Reduced for speed
    )

    print("Preparing train/val/test splits...")
    data_dict = rf_model.prepare_data(df_ml, test_size=0.2, val_size=0.1)

    print(f"  Train: {len(data_dict['y_train'])} samples")
    print(f"  Val:   {len(data_dict['y_val'])} samples")
    print(f"  Test:  {len(data_dict['y_test'])} samples")

    print("\nTraining Random Forest (this may take a minute)...")
    metrics = rf_model.train(data_dict, verbose=True)

    # Feature importance
    print("\nTop 10 Features by Importance:")
    importance_df = rf_model.get_feature_importance(top_n=10)
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    return rf_model, data_dict


def test_lstm(df_features, fe):
    """Test LSTM model."""
    print("\n" + "=" * 60)
    print("Testing LSTM")
    print("=" * 60)

    # Initialize LSTM with smaller architecture for testing
    print("\nInitializing LSTM...")
    lstm_model = LSTMPredictor(
        sequence_length=30,  # Reduced for speed
        lstm_units=[64, 32],  # Smaller network
        dropout_rate=0.2,
    )

    # Select subset of features
    all_features = fe.get_feature_names(df_features)
    important_features = [
        f
        for f in all_features
        if any(
            key in f for key in ["sma", "ema", "rsi", "macd", "returns", "volume_ratio"]
        )
    ][
        :15
    ]  # Limit to 15 features

    print(f"Using {len(important_features)} features for LSTM")

    # Drop rows with NaN in selected features or target
    df_clean = df_features[important_features + ["close"]].dropna()
    print(
        f"After removing NaN: {len(df_clean)} rows ({len(df_features) - len(df_clean)} removed)"
    )

    # Prepare data
    print("\nPreparing sequences...")
    data_dict = lstm_model.prepare_data(
        df_clean,
        target_col="close",
        feature_cols=important_features,
        test_size=0.2,
        val_size=0.1,
    )

    print(f"  Train: {len(data_dict['y_train'])} sequences")
    print(f"  Val:   {len(data_dict['y_val'])} sequences")
    print(f"  Test:  {len(data_dict['y_test'])} sequences")

    # Train model
    print("\nTraining LSTM (this will take a few minutes)...")
    print("Epochs: 20 (reduced for testing)")

    history = lstm_model.train(
        data_dict,
        epochs=20,  # Reduced for testing
        batch_size=32,
        early_stopping_patience=5,
        verbose=1,
    )

    # Evaluate
    print("\nEvaluating on test set...")
    eval_metrics = lstm_model.evaluate(data_dict["X_test"], data_dict["y_test"])

    print("\nTest Set Metrics:")
    print(f"  RMSE: {eval_metrics['rmse']:.4f}")
    print(f"  MAE:  {eval_metrics['mae']:.4f}")
    print(f"  MAPE: {eval_metrics['mape']:.2f}%")
    print(f"  Directional Accuracy: {eval_metrics['directional_accuracy']:.2%}")

    return lstm_model


def test_ml_strategy(df_features, fe, rf_model, lstm_model):
    """Test ML-based trading strategy."""
    print("\n" + "=" * 60)
    print("Testing ML Strategy")
    print("=" * 60)

    # Create ensemble strategy
    print("\nCreating ensemble strategy...")
    ml_strategy = MLPredictionStrategy(
        lstm_model=lstm_model,
        rf_model=rf_model,
        feature_engineer=fe,
        confidence_threshold=0.6,
        lstm_weight=0.5,
        rf_weight=0.5,
    )

    # Generate signals
    print("\nGenerating trading signals...")
    signals = ml_strategy.generate_signals(df_features)

    # Signal statistics
    position_counts = signals["position"].value_counts()
    print(f"\nSignal Distribution:")
    print(
        f"  Long (1):  {position_counts.get(1.0, 0)} ({position_counts.get(1.0, 0) / len(signals) * 100:.1f}%)"
    )
    print(
        f"  Flat (0):  {position_counts.get(0.0, 0)} ({position_counts.get(0.0, 0) / len(signals) * 100:.1f}%)"
    )
    print(
        f"  Short (-1): {position_counts.get(-1.0, 0)} ({position_counts.get(-1.0, 0) / len(signals) * 100:.1f}%)"
    )

    print(f"\nAverage Confidence: {signals['confidence'].mean():.3f}")
    print(f"Max Confidence: {signals['confidence'].max():.3f}")

    print("\nStrategy test complete!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ML Implementation Test Suite")
    print("=" * 60)

    try:
        # Test 1: Feature Engineering
        result = test_feature_engineering()
        if result is None:
            print("\nFailed to load data. Exiting.")
            return

        df_features, fe = result

        # Test 2: Random Forest
        rf_model, rf_data_dict = test_random_forest(df_features, fe)

        # Test 3: LSTM
        lstm_model = test_lstm(df_features, fe)

        # Test 4: ML Strategy
        test_ml_strategy(df_features, fe, rf_model, lstm_model)

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Open Streamlit app: streamlit run streamlit_apps/main.py")
        print("2. Navigate to ML Analysis page")
        print("3. Train models on your preferred symbols")
        print("4. Compare ML strategies vs traditional approaches")

    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
