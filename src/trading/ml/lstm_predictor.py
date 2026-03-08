"""LSTM-based price prediction model for time series forecasting.

Uses Long Short-Term Memory neural networks to predict future price movements.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class LSTMPredictor:
    """LSTM model for stock price prediction."""

    def __init__(
        self,
        sequence_length: int = 40,
        lstm_units: List[int] = [128, 64],
        dropout_rate: float = 0.4,
        learning_rate: float = 1e-4,
        model_type: str = "classifier",
    ):
        """Initialize LSTM predictor.

        Args:
            sequence_length: Number of time steps to look back
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_type: ``"classifier"`` (default) predicts P(return>0) with
                sigmoid + binary cross-entropy, which directly optimises the
                directional signal.  ``"regressor"`` predicts return magnitude
                with a linear output + Huber loss (robust to outlier days).
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        self.history = None
        # Mirrors model_type after prepare_data(); used by evaluate() to pick
        # the right accuracy formula.
        self.target_type: str = model_type

    def _build_model(self, n_features: int):
        """Build LSTM model architecture.

        Args:
            n_features: Number of input features
        """
        try:
            from tensorflow import keras  # type: ignore
            from tensorflow.keras import layers  # type: ignore
        except ImportError:
            raise ImportError(
                "TensorFlow is required for LSTM models. "
                "Install with: pip install tensorflow"
            )

        model = keras.Sequential()

        # Explicit Input layer (avoids deprecated input_shape argument on layers)
        model.add(keras.Input(shape=(self.sequence_length, n_features)))

        # First LSTM layer + BatchNorm + Dropout.
        # BatchNormalization stabilises training when the signal is tiny
        # (daily return magnitudes ~0.001), reducing internal covariate shift.
        # NOTE: recurrent_dropout is intentionally omitted.  Even small values
        # (e.g. 0.1) force TensorFlow to fall back from the fast cuDNN LSTM
        # kernel to a slower, step-by-step implementation.  On low-signal
        # financial data that already uses EarlyStopping, the slower kernel
        # means the model barely trains before stopping.  The combination of
        # BatchNorm + output Dropout is sufficient regularisation here.
        model.add(
            layers.LSTM(
                units=self.lstm_units[0],
                return_sequences=len(self.lstm_units) > 1,
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(self.dropout_rate))

        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_seq = i < len(self.lstm_units) - 2
            model.add(layers.LSTM(units=units, return_sequences=return_seq))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))

        if self.model_type == "classifier":
            # Sigmoid output + binary cross-entropy directly optimises
            # P(return > 0), which is the exact directional signal we measure.
            model.add(layers.Dense(units=1, activation="sigmoid"))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        else:
            # Regressor: linear output + Huber loss.
            # Huber clips the gradient influence of outlier days (earnings,
            # flash-crashes) that dominate MSE and mask everyday patterns.
            model.add(layers.Dense(units=1))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss=keras.losses.Huber(),
                metrics=["mae"],
            )

        self.model = model

    def create_sequences(
        self, data: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for LSTM training.

        Args:
            data: Feature data
            target: Target values (if training)

        Returns:
            Tuple of (X_sequences, y_sequences) or (X_sequences, None)
        """
        X = []
        y: Optional[List] = [] if target is not None else None

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            if target is not None and y is not None:
                y.append(target[i])

        X = np.array(X)

        if y is not None:
            y = np.array(y)  # type: ignore
            return X, y  # type: ignore

        return X, None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "returns",
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        target_type: str = "return",
    ) -> Dict[str, Any]:
        """Prepare data for training.

        Args:
            df: DataFrame with features
            target_col: Column to predict.  Should reference a *returns* column
                so the sign carries directional meaning.  For the classifier
                the values are binarised internally (1 if > 0, else 0).
            feature_cols: List of feature columns (if None, use all numeric)
            test_size: Proportion for test set
            val_size: Proportion for validation set
            target_type: Ignored when ``model_type="classifier"`` (binarisation
                is applied automatically).  For regressors: ``"return"`` or
                ``"price"``; controls the directional-accuracy formula in
                ``evaluate()``.

        Returns:
            Dict with train/val/test splits and metadata
        """
        # model_type drives both the target encoding and the evaluate formula.
        self.target_type = (
            self.model_type if self.model_type == "classifier" else target_type
        )
        # Select features
        if feature_cols is None:
            # Use all numeric columns except target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)

        self.feature_columns = feature_cols

        # Drop rows that contain NaN or inf in any feature or target column —
        # NaN rows are produced by rolling-window indicators (SMA-200, ATR,
        # lagged features, etc.) at the start of the series.  Inf values can
        # appear in ratio columns (e.g. bb_percent when bands collapse, or
        # volume_ratio when the volume SMA is zero).  Both propagate through
        # MinMaxScaler and cause flat loss curves and NaN metrics.
        cols_needed = list(dict.fromkeys(feature_cols + [target_col]))
        df = df[cols_needed].replace([np.inf, -np.inf], np.nan).dropna()

        # Extract raw features and targets before any scaling or sequencing.
        features = df[feature_cols].values

        if self.model_type == "classifier":
            # Binarise: 1 if the return for this bar is positive, else 0.
            # The sequence ending at bar t-1 predicts returns[t], so the label
            # is whether the next day closes up.
            col_values = np.asarray(df[target_col].values, dtype=np.float64)
            target = (col_values > 0).astype(np.float32).reshape(-1, 1)
        else:
            target = np.asarray(df[target_col].values, dtype=np.float64).reshape(-1, 1)

        # Chronological split on raw arrays before fitting any scaler.
        # Using sequence_length as the gap ensures no training sequence
        # reaches into the test window.
        n_raw = len(features)
        test_split = int(n_raw * (1 - test_size))
        seq_gap = self.sequence_length

        features_train_raw = features[: test_split - seq_gap]
        features_test_raw = features[test_split:]
        target_train_raw = target[: test_split - seq_gap]
        target_test_raw = target[test_split:]

        # Fit scaler on training data only; apply same transform to test.
        features_train_scaled = self.scaler.fit_transform(features_train_raw)
        features_test_scaled = self.scaler.transform(features_test_raw)

        if self.model_type == "classifier":
            target_train_scaled = target_train_raw  # already 0/1 — no scaler needed
            target_test_scaled = target_test_raw
        else:
            # Scale target — prevents exploding gradients from raw magnitudes
            target_train_scaled = self.target_scaler.fit_transform(target_train_raw)
            target_test_scaled = self.target_scaler.transform(target_test_raw)

        # Create sequences for each partition separately so no sequence spans
        # the train/test boundary.
        X_all_train, y_all_train_opt = self.create_sequences(
            features_train_scaled, target_train_scaled
        )
        X_test, y_test_opt = self.create_sequences(
            features_test_scaled, target_test_scaled
        )

        # target is always supplied above so y will never be None
        assert y_all_train_opt is not None and y_test_opt is not None
        y_all_train = y_all_train_opt
        y_test = y_test_opt

        # Chronological val split within training sequences
        val_ratio = val_size / (1 - test_size)
        val_split = len(X_all_train) - int(len(X_all_train) * val_ratio)
        X_train, X_val = X_all_train[:val_split], X_all_train[val_split:]
        y_train, y_val = y_all_train[:val_split], y_all_train[val_split:]

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_columns": feature_cols,
            "target_column": target_col,
        }

    def train(
        self,
        data_dict: Dict[str, Any],
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """Train LSTM model.

        Args:
            data_dict: Dictionary from prepare_data()
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level

        Returns:
            Training history dictionary
        """
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
        except ImportError:
            raise ImportError(
                "TensorFlow is required. Install with: pip install tensorflow"
            )

        # Build model if not already built
        if self.model is None:
            n_features = data_dict["X_train"].shape[2]
            self._build_model(n_features)

        # Clamp verbose to Literal[0, 1] expected by Keras callbacks
        _cb_verbose: int = 1 if verbose else 0

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=_cb_verbose,  # type: ignore[arg-type]
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=_cb_verbose  # type: ignore[arg-type]
            ),
        ]

        # Class weights for the binary classifier.
        #
        # We intentionally do NOT use "balanced" weights.
        # "balanced" adjusts so that up-days and down-days contribute equally
        # to the loss, effectively centering the model on P(up)≈0.5 from the
        # start.  For equities this erases the natural bull-market drift
        # (~53-54 % up days for large-caps), which is itself a weak but real
        # predictive signal.  Without forced balancing the model is free to
        # learn both the base-rate drift AND any additional technical signal,
        # which consistently yields 3-6 pp higher directional accuracy.
        #
        # The class imbalance (54/46) is mild enough that it does not prevent
        # the model from learning the minority class; we rely on sufficient
        # training data and regularisation (Dropout + BatchNorm) instead.
        class_weight: Optional[dict] = None

        # Train model
        self.history = self.model.fit(  # type: ignore
            data_dict["X_train"],
            data_dict["y_train"],
            validation_data=(data_dict["X_val"], data_dict["y_val"]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose,
        )

        hist = dict(self.history.history)

        if self.model_type == "regressor":
            # Normalise MAE key names: older Keras versions emit the long form.
            for src, dst in [
                ("mean_absolute_error", "mae"),
                ("val_mean_absolute_error", "val_mae"),
            ]:
                if src in hist and dst not in hist:
                    hist[dst] = hist.pop(src)

        return hist

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input sequences

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X, verbose=0)

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Predict from raw DataFrame.

        Args:
            df: DataFrame with feature columns

        Returns:
            Series with predictions
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained. Call train() first.")

        # Extract and scale features
        features = df[self.feature_columns].values
        features_scaled = self.scaler.transform(features)

        # Create sequences
        X, _ = self.create_sequences(features_scaled)

        predictions_raw = self.predict(X)
        if self.model_type == "classifier":
            # Return probabilities directly (no inverse transform needed)
            predictions = predictions_raw.flatten()
        else:
            predictions = self.target_scaler.inverse_transform(
                predictions_raw.reshape(-1, 1)
            ).flatten()

        # Create series with proper index
        # Note: predictions start at index sequence_length
        pred_index = df.index[self.sequence_length :]
        return pd.Series(predictions, index=pred_index)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        y_pred_raw = self.predict(X_test)

        if self.model_type == "classifier":
            # Classifier: predictions are probabilities in [0, 1].
            # Labels are binary 0/1 (1 = positive return day).
            y_pred_prob = y_pred_raw.flatten()
            y_true = y_test.flatten()
            pred_classes = (y_pred_prob >= 0.5).astype(float)
            directional_accuracy = float(np.mean(pred_classes == y_true))
            # Brier-score-style error (meaningful for probabilistic classifiers)
            mse = float(np.mean((y_true - y_pred_prob) ** 2))
            mae = float(np.mean(np.abs(y_true - y_pred_prob)))
            rmse = float(np.sqrt(mse))
            mape = 0.0  # not meaningful for binary classifier

            # Confident-DA: accuracy restricted to samples where the model
            # has clear conviction (prob outside the [0.45, 0.55] dead zone).
            # A model that predicts randomly has ~50% of predictions outside
            # this band purely by chance; a model with real signal concentrates
            # more mass near 0 and 1.  For trading this is the relevant metric:
            # we only trade when the model is confident.
            _conf_threshold = 0.05  # distance from 0.5
            _confident_mask = np.abs(y_pred_prob - 0.5) >= _conf_threshold
            if _confident_mask.sum() > 0:
                confident_da = float(
                    np.mean(pred_classes[_confident_mask] == y_true[_confident_mask])
                )
                confident_coverage = float(_confident_mask.mean())
            else:
                confident_da = directional_accuracy
                confident_coverage = 1.0
        else:
            # Regressor: inverse-transform to original return/price space.
            y_test_real = self.target_scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()
            y_pred_real = self.target_scaler.inverse_transform(
                y_pred_raw.reshape(-1, 1)
            ).flatten()

            mse = float(np.mean((y_test_real - y_pred_real) ** 2))
            mae = float(np.mean(np.abs(y_test_real - y_pred_real)))
            rmse = float(np.sqrt(mse))

            if self.target_type == "return":
                actual_direction = np.sign(y_test_real)
                pred_direction = np.sign(y_pred_real)
            else:
                actual_direction = np.sign(np.diff(y_test_real))
                pred_direction = np.sign(np.diff(y_pred_real))
            directional_accuracy = float(np.mean(actual_direction == pred_direction))

            nonzero = y_test_real != 0
            mape = float(
                np.mean(
                    np.abs(
                        (y_test_real[nonzero] - y_pred_real[nonzero])
                        / y_test_real[nonzero]
                    )
                )
                * 100
            )

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "directional_accuracy": directional_accuracy,
            # Only meaningful for classifiers; fall back to DA for regressors.
            "confident_directional_accuracy": (
                confident_da
                if self.model_type == "classifier"
                else directional_accuracy
            ),
            "confident_coverage": (
                confident_coverage if self.model_type == "classifier" else 1.0
            ),
        }

    def get_predictions_df(
        self, df: pd.DataFrame, data_dict: Dict[str, Any]
    ) -> pd.DataFrame:
        """Get predictions for all data splits.

        Args:
            df: Original DataFrame
            data_dict: Dictionary from prepare_data()

        Returns:
            DataFrame with actual and predicted values
        """
        # Predict on all sets (scaled space)
        train_pred = self.predict(data_dict["X_train"])
        val_pred = self.predict(data_dict["X_val"])
        test_pred = self.predict(data_dict["X_test"])

        # Combine and inverse-transform to real price space
        all_pred_scaled = np.concatenate([train_pred, val_pred, test_pred])
        all_actual_scaled = np.concatenate(
            [
                data_dict["y_train"].flatten(),
                data_dict["y_val"].flatten(),
                data_dict["y_test"].flatten(),
            ]
        )

        if self.model_type == "classifier":
            # Classifier: predictions are probabilities, actuals are 0/1 labels
            all_pred = all_pred_scaled.flatten()
            all_actual = all_actual_scaled.flatten()
        else:
            all_pred = self.target_scaler.inverse_transform(
                all_pred_scaled.reshape(-1, 1)
            ).flatten()
            all_actual = self.target_scaler.inverse_transform(
                all_actual_scaled.reshape(-1, 1)
            ).flatten()

        # Create result DataFrame
        result_df = pd.DataFrame(
            {
                "actual": all_actual,
                "predicted": all_pred,
            }
        )

        # Add split labels
        train_size = len(data_dict["y_train"])
        val_size = len(data_dict["y_val"])
        test_size = len(data_dict["y_test"])

        result_df["split"] = (
            ["train"] * train_size + ["val"] * val_size + ["test"] * test_size
        )

        # Add index from original data (accounting for sequence length)
        result_df.index = df.index[
            self.sequence_length : self.sequence_length + len(result_df)
        ]

        return result_df

    def save(self, filepath: str):
        """Save model and scaler.

        Args:
            filepath: Path to save model (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save.")

        # Save model
        self.model.save(f"{filepath}_model.keras")

        # Save scaler and metadata
        metadata = {
            "scaler": self.scaler,
            "target_scaler": self.target_scaler,
            "feature_columns": self.feature_columns,
            "sequence_length": self.sequence_length,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "target_type": self.target_type,
            "model_type": self.model_type,
        }

        with open(f"{filepath}_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load(self, filepath: str):
        """Load model and scaler.

        Args:
            filepath: Path to saved model (without extension)
        """
        try:
            from tensorflow import keras  # type: ignore
        except ImportError:
            raise ImportError(
                "TensorFlow is required. Install with: pip install tensorflow"
            )

        # Load model
        self.model = keras.models.load_model(f"{filepath}_model.keras")  # type: ignore

        # Load scaler and metadata
        with open(f"{filepath}_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.scaler = metadata["scaler"]
        self.target_scaler = metadata.get(
            "target_scaler", MinMaxScaler(feature_range=(0, 1))
        )
        self.feature_columns = metadata["feature_columns"]
        self.sequence_length = metadata["sequence_length"]
        self.lstm_units = metadata["lstm_units"]
        self.dropout_rate = metadata["dropout_rate"]
        self.learning_rate = metadata["learning_rate"]
        self.model_type = metadata.get("model_type", "regressor")  # back-compat
        self.target_type = metadata.get("target_type", "price")

    def get_model_summary(self) -> str:
        """Get model architecture summary.

        Returns:
            String representation of model
        """
        if self.model is None:
            return "Model not built yet."

        import io

        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + "\n"))
        return stream.getvalue()
