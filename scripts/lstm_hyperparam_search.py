"""Hyperparameter search for LSTM on MSFT 1Day data.

Tests combinations of sequence_length, learning_rate, dropout_rate, lstm_units,
and batch_size.  Prints results sorted by directional accuracy on the test set.
"""

import sys
import os
import itertools
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, r"c:\Users\hdroe\OneDrive\Documents\Coursera\alpaca-trading-api")

from src.trading.data.storage import DataStore
from src.trading.ml.feature_engineering import FeatureEngineering
from src.trading.ml.lstm_predictor import LSTMPredictor

# ── Load & feature-engineer MSFT data ──────────────────────────────────────
ds = DataStore(data_dir=r"Z:\market_data")
df_raw = ds.load("MSFT", "1Day")
if df_raw is None:
    raise RuntimeError("No data found for MSFT 1Day in the DataStore.")

fe = FeatureEngineering()
df = fe.generate_features(df_raw)

# Use the same normalised feature set as the Streamlit app
all_feature_cols = fe.get_feature_names(df)
feature_cols = [
    col
    for col in all_feature_cols
    if any(
        key in col
        for key in [
            "returns",
            "log_returns",
            "rsi",
            "macd",
            "bb_percent",
            "bb_bandwidth",
            "atr_percent",
            "volume_ratio",
            "volume_change",
            "_ratio",
            "_distance",
            "bullish_candle",
            "candle_body",
            "high_low_range",
            "close_open_range",
        ]
    )
    and col != "returns"
][:30]

print(f"Features ({len(feature_cols)}): {feature_cols}\n")

# ── Hyperparameter grid ─────────────────────────────────────────────────────
# Focused 96-combo search.  Full 640-combo sweep would take several hours;
# these ranges cover the highest-leverage axes identified from prior analysis.
GRID = {
    "sequence_length": [20, 40, 60, 90],
    "lstm_units": [[64, 32], [128, 64]],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "dropout_rate": [0.3, 0.4],
    "batch_size": [16, 32],
}

keys = list(GRID.keys())
combos = list(itertools.product(*GRID.values()))
total = len(combos)
print(f"Total combinations: {total}\n")

results = []
for idx, combo in enumerate(combos, 1):
    params = dict(zip(keys, combo))
    seq = params["sequence_length"]
    units = params["lstm_units"]
    lr = params["learning_rate"]
    drop = params["dropout_rate"]
    bs = params["batch_size"]

    try:
        model = LSTMPredictor(
            sequence_length=seq,
            lstm_units=units,
            dropout_rate=drop,
            learning_rate=lr,
        )
        data_dict = model.prepare_data(
            df,
            target_col="returns",
            target_type="return",
            feature_cols=feature_cols,
            test_size=0.2,
            val_size=0.1,
        )
        model.train(
            data_dict, epochs=60, batch_size=bs, early_stopping_patience=8, verbose=0
        )
        metrics = model.evaluate(data_dict["X_test"], data_dict["y_test"])
        da = metrics["directional_accuracy"]

        result = {**params, "directional_accuracy": da, "mae": metrics["mae"]}
        results.append(result)

        marker = (
            " *** BEST" if da == max(r["directional_accuracy"] for r in results) else ""
        )
        print(
            f"[{idx:3d}/{total}] seq={seq:3d} units={str(units):12s} "
            f"lr={lr:.0e} drop={drop:.1f} bs={bs:2d}  "
            f"DA={da:.4f}{marker}"
        )
    except Exception as e:
        print(f"[{idx:3d}/{total}] FAILED: {params} — {e}")

# ── Summary ─────────────────────────────────────────────────────────────────
results.sort(key=lambda r: r["directional_accuracy"], reverse=True)
print("\n" + "=" * 80)
print("TOP 15 RESULTS BY DIRECTIONAL ACCURACY")
print("=" * 80)
print(
    f"{'seq':>5} {'units':>14} {'lr':>8} {'drop':>6} {'bs':>4}  {'DA':>8}  {'MAE':>10}"
)
print("-" * 80)
for r in results[:15]:
    print(
        f"{r['sequence_length']:5d} {str(r['lstm_units']):>14} "
        f"{r['learning_rate']:8.0e} {r['dropout_rate']:6.2f} {r['batch_size']:4d}  "
        f"{r['directional_accuracy']:8.4f}  {r['mae']:10.6f}"
    )
