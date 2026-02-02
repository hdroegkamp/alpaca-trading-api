"""ML-based trading strategy using ensemble predictions.

Combines LSTM and Random Forest predictions to generate trading signals.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from src.trading.strategy.base import Strategy
from src.trading.ml.lstm_predictor import LSTMPredictor
from src.trading.ml.random_forest_analyzer import RandomForestAnalyzer
from src.trading.ml.feature_engineering import FeatureEngineering


class MLPredictionStrategy(Strategy):
    """Trading strategy using ML model predictions.

    Uses ensemble of LSTM and Random Forest to predict price direction
    and generate trading signals.
    """

    def __init__(
        self,
        lstm_model: Optional[LSTMPredictor] = None,
        rf_model: Optional[RandomForestAnalyzer] = None,
        feature_engineer: Optional[FeatureEngineering] = None,
        confidence_threshold: float = 0.6,
        lstm_weight: float = 0.5,
        rf_weight: float = 0.5,
        position_sizing: str = "fixed",
        **params,
    ):
        """Initialize ML prediction strategy.

        Args:
            lstm_model: Trained LSTM predictor
            rf_model: Trained Random Forest analyzer
            feature_engineer: Feature engineering instance
            confidence_threshold: Minimum confidence to enter position
            lstm_weight: Weight for LSTM predictions in ensemble
            rf_weight: Weight for RF predictions in ensemble
            position_sizing: 'fixed' or 'proportional' (to confidence)
            **params: Additional strategy parameters
        """
        super().__init__(**params)

        self.lstm_model = lstm_model
        self.rf_model = rf_model
        self.feature_engineer = feature_engineer or FeatureEngineering()

        self.confidence_threshold = confidence_threshold
        self.lstm_weight = lstm_weight
        self.rf_weight = rf_weight
        self.position_sizing = position_sizing

        # Normalize weights
        total_weight = lstm_weight + rf_weight
        self.lstm_weight = lstm_weight / total_weight
        self.rf_weight = rf_weight / total_weight

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from ML predictions.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with signals column (-1, 0, 1) and metadata
        """
        # Generate features
        data_with_features = self.feature_engineer.generate_features(data)

        # Get predictions from models
        signals = data.copy()
        signals["position"] = 0.0
        signals["confidence"] = 0.0
        signals["lstm_pred"] = 0.0
        signals["rf_pred"] = 0.0

        # LSTM predictions (if model provided)
        if self.lstm_model is not None:
            try:
                lstm_predictions = self.lstm_model.predict_from_dataframe(
                    data_with_features
                )

                # Convert price predictions to direction
                current_price = data_with_features.loc[lstm_predictions.index, "close"]
                lstm_direction = (lstm_predictions > current_price).astype(float)
                lstm_direction = lstm_direction * 2 - 1  # Convert 0/1 to -1/1

                # Calculate confidence from prediction magnitude
                price_change_pct = (lstm_predictions - current_price) / current_price
                lstm_confidence = np.abs(price_change_pct).clip(0, 1)

                signals.loc[lstm_predictions.index, "lstm_pred"] = lstm_direction
                signals.loc[lstm_predictions.index, "lstm_confidence"] = lstm_confidence

            except Exception as e:
                print(f"Warning: LSTM prediction failed: {e}")

        # Random Forest predictions (if model provided)
        if self.rf_model is not None:
            try:
                rf_predictions = self.rf_model.predict_from_dataframe(
                    data_with_features
                )

                # Convert 0/1 predictions to -1/1
                rf_direction = rf_predictions["prediction"].values * 2 - 1  # type: ignore
                rf_confidence = rf_predictions["probability_up"].values  # type: ignore
                rf_confidence = np.where(  # type: ignore
                    rf_direction == 1, rf_confidence, 1 - rf_confidence  # type: ignore
                )

                signals.loc[rf_predictions.index, "rf_pred"] = rf_direction
                signals.loc[rf_predictions.index, "rf_confidence"] = rf_confidence

            except Exception as e:
                print(f"Warning: RF prediction failed: {e}")

        # Ensemble predictions
        if self.lstm_model is not None and self.rf_model is not None:
            # Weighted average of directions
            ensemble_signal = (
                signals["lstm_pred"] * self.lstm_weight
                + signals["rf_pred"] * self.rf_weight
            )

            # Weighted average of confidences
            ensemble_confidence = (
                signals.get("lstm_confidence", 0) * self.lstm_weight
                + signals.get("rf_confidence", 0) * self.rf_weight
            )

        elif self.lstm_model is not None:
            ensemble_signal = signals["lstm_pred"]
            ensemble_confidence = signals.get("lstm_confidence", 0.5)

        elif self.rf_model is not None:
            ensemble_signal = signals["rf_pred"]
            ensemble_confidence = signals.get("rf_confidence", 0.5)

        else:
            raise ValueError("At least one model (LSTM or RF) must be provided")

        # Apply confidence threshold
        signals["confidence"] = ensemble_confidence

        # Generate position based on signal strength and confidence
        if self.position_sizing == "proportional":
            # Position size proportional to confidence
            signals["position"] = np.where(
                ensemble_confidence >= self.confidence_threshold,
                np.sign(ensemble_signal) * ensemble_confidence,
                0,
            )
        else:
            # Fixed position size
            signals["position"] = np.where(
                ensemble_confidence >= self.confidence_threshold,
                np.sign(ensemble_signal),
                0,
            )

        # Smooth signals to avoid excessive trading
        signals["position"] = self._smooth_signals(signals["position"])

        return signals[["position", "confidence", "lstm_pred", "rf_pred"]]

    def _smooth_signals(self, positions: pd.Series, window: int = 3) -> pd.Series:
        """Smooth position signals to reduce noise.

        Args:
            positions: Raw position signals
            window: Smoothing window size

        Returns:
            Smoothed positions
        """
        # Use rolling mean and round to nearest valid position
        smoothed = positions.rolling(window=window, min_periods=1).mean()

        # Convert to discrete positions: -1, 0, 1
        result = pd.Series(0, index=positions.index)
        result[smoothed > 0.3] = 1
        result[smoothed < -0.3] = -1

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models.

        Returns:
            Dictionary with model configuration
        """
        return {
            "lstm_loaded": self.lstm_model is not None,
            "rf_loaded": self.rf_model is not None,
            "lstm_weight": self.lstm_weight,
            "rf_weight": self.rf_weight,
            "confidence_threshold": self.confidence_threshold,
            "position_sizing": self.position_sizing,
        }


class LSTMStrategy(Strategy):
    """Trading strategy using only LSTM predictions."""

    def __init__(
        self,
        lstm_model: LSTMPredictor,
        feature_engineer: Optional[FeatureEngineering] = None,
        confidence_threshold: float = 0.02,
        **params,
    ):
        """Initialize LSTM strategy.

        Args:
            lstm_model: Trained LSTM predictor
            feature_engineer: Feature engineering instance
            confidence_threshold: Minimum price change % to enter position
            **params: Additional strategy parameters
        """
        super().__init__(**params)
        self.lstm_model = lstm_model
        self.feature_engineer = feature_engineer or FeatureEngineering()
        self.confidence_threshold = confidence_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals from LSTM predictions."""
        # Generate features
        data_with_features = self.feature_engineer.generate_features(data)

        # Get LSTM predictions
        predictions = self.lstm_model.predict_from_dataframe(data_with_features)

        # Create signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals["position"] = 0.0
        signals["confidence"] = 0.0
        signals["predicted_price"] = 0.0

        # Calculate position based on predicted price change
        current_price = data.loc[predictions.index, "close"]
        price_change_pct = (predictions - current_price) / current_price

        # Generate positions
        signals.loc[predictions.index, "predicted_price"] = predictions
        signals.loc[predictions.index, "confidence"] = np.abs(price_change_pct)
        signals.loc[predictions.index, "position"] = np.where(
            np.abs(price_change_pct) >= self.confidence_threshold,
            np.sign(price_change_pct),
            0,
        )

        return signals[["position", "confidence", "predicted_price"]]


class RandomForestStrategy(Strategy):
    """Trading strategy using only Random Forest predictions."""

    def __init__(
        self,
        rf_model: RandomForestAnalyzer,
        feature_engineer: Optional[FeatureEngineering] = None,
        confidence_threshold: float = 0.6,
        **params,
    ):
        """Initialize Random Forest strategy.

        Args:
            rf_model: Trained Random Forest analyzer
            feature_engineer: Feature engineering instance
            confidence_threshold: Minimum probability to enter position
            **params: Additional strategy parameters
        """
        super().__init__(**params)
        self.rf_model = rf_model
        self.feature_engineer = feature_engineer or FeatureEngineering()
        self.confidence_threshold = confidence_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals from Random Forest predictions."""
        # Generate features
        data_with_features = self.feature_engineer.generate_features(data)

        # Get RF predictions
        predictions = self.rf_model.predict_from_dataframe(data_with_features)

        # Create signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals["position"] = 0.0
        signals["confidence"] = 0.0
        signals["probability_up"] = 0.0

        # Generate positions based on probability
        signals.loc[predictions.index, "probability_up"] = predictions["probability_up"]
        signals.loc[predictions.index, "confidence"] = np.maximum(
            predictions["probability_up"], predictions["probability_down"]
        )

        # Position: 1 if prob_up > threshold, -1 if prob_down > threshold, else 0
        signals.loc[predictions.index, "position"] = np.where(
            predictions["probability_up"] >= self.confidence_threshold,
            1,
            np.where(
                predictions["probability_down"] >= self.confidence_threshold, -1, 0
            ),
        )

        return signals[["position", "confidence", "probability_up"]]
