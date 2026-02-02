"""LSTM-based price prediction model for time series forecasting.

Uses Long Short-Term Memory neural networks to predict future price movements.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os


class LSTMPredictor:
    """LSTM model for stock price prediction."""

    def __init__(
        self,
        sequence_length: int = 60,
        lstm_units: List[int] = [128, 64],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        """Initialize LSTM predictor.

        Args:
            sequence_length: Number of time steps to look back
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        self.history = None

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

        # First LSTM layer
        model.add(
            layers.LSTM(
                units=self.lstm_units[0],
                return_sequences=len(self.lstm_units) > 1,
                input_shape=(self.sequence_length, n_features),
            )
        )
        model.add(layers.Dropout(self.dropout_rate))

        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_seq = i < len(self.lstm_units) - 2
            model.add(layers.LSTM(units=units, return_sequences=return_seq))
            model.add(layers.Dropout(self.dropout_rate))

        # Dense output layer
        model.add(layers.Dense(units=1))

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
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
        target_col: str = "close",
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Dict[str, Any]:
        """Prepare data for training.

        Args:
            df: DataFrame with features
            target_col: Column to predict
            feature_cols: List of feature columns (if None, use all numeric)
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Dict with train/val/test splits and metadata
        """
        # Select features
        if feature_cols is None:
            # Use all numeric columns except target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols and target_col != "close":
                feature_cols.remove(target_col)

        self.feature_columns = feature_cols

        # Extract features and target
        features = df[feature_cols].values
        target = df[target_col].values.reshape(-1, 1)  # type: ignore

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Create sequences
        X, y = self.create_sequences(features_scaled, target)

        # Split into train/val/test
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Second split: separate train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, shuffle=False
        )

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

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=verbose
            ),
        ]

        # Train model
        self.history = self.model.fit(  # type: ignore
            data_dict["X_train"],
            data_dict["y_train"],
            validation_data=(data_dict["X_val"], data_dict["y_val"]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        return self.history.history

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

        # Predict
        predictions = self.predict(X)

        # Create series with proper index
        # Note: predictions start at index sequence_length
        pred_index = df.index[self.sequence_length :]
        return pd.Series(predictions.flatten(), index=pred_index)

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

        # Get predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        mse = float(np.mean((y_test - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(mse))

        # Calculate directional accuracy
        actual_direction = np.sign(np.diff(y_test.flatten()))
        pred_direction = np.sign(np.diff(y_pred.flatten()))
        directional_accuracy = float(np.mean(actual_direction == pred_direction))

        # MAPE
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "directional_accuracy": directional_accuracy,
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
        # Predict on all sets
        train_pred = self.predict(data_dict["X_train"])
        val_pred = self.predict(data_dict["X_val"])
        test_pred = self.predict(data_dict["X_test"])

        # Combine predictions
        all_pred = np.concatenate([train_pred, val_pred, test_pred])

        # Create result DataFrame
        result_df = pd.DataFrame(
            {
                "actual": np.concatenate(
                    [
                        data_dict["y_train"].flatten(),
                        data_dict["y_val"].flatten(),
                        data_dict["y_test"].flatten(),
                    ]
                ),
                "predicted": all_pred.flatten(),
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
            "feature_columns": self.feature_columns,
            "sequence_length": self.sequence_length,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
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
        self.feature_columns = metadata["feature_columns"]
        self.sequence_length = metadata["sequence_length"]
        self.lstm_units = metadata["lstm_units"]
        self.dropout_rate = metadata["dropout_rate"]
        self.learning_rate = metadata["learning_rate"]

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
