"""Trading strategy wrapping the Random Forest baseline.

Turns the model's predicted probability of an up move into a discrete
position: long when confidence in "up" is high, short when confidence in
"down" is high, otherwise flat.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from src.trading.strategy.base import Strategy
from src.trading.ml.random_forest_analyzer import RandomForestAnalyzer
from src.trading.ml.feature_engineering import FeatureEngineering


class RandomForestStrategy(Strategy):
    """Long/short strategy driven by Random Forest direction probabilities."""

    def __init__(
        self,
        rf_model: RandomForestAnalyzer,
        feature_engineer: Optional[FeatureEngineering] = None,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        **params,
    ):
        """Initialize the strategy.

        Args:
            rf_model: A trained :class:`RandomForestAnalyzer`.
            feature_engineer: Feature pipeline (defaults to a fresh instance).
            long_threshold: Go long when P(up) >= this value.
            short_threshold: Go short when P(up) <= this value.
            **params: Additional strategy parameters passed to the base class.
        """
        super().__init__(**params)
        if long_threshold <= short_threshold:
            raise ValueError("long_threshold must be greater than short_threshold")
        self.rf_model = rf_model
        self.feature_engineer = feature_engineer or FeatureEngineering()
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate -1/0/1 positions from model probabilities.

        Args:
            data: OHLCV DataFrame indexed by timestamp.

        Returns:
            DataFrame with columns: position, confidence, probability_up.
        """
        data_with_features = self.feature_engineer.generate_features(data)
        predictions = self.rf_model.predict_from_dataframe(data_with_features)

        signals = pd.DataFrame(index=data.index)
        signals["position"] = 0.0
        signals["confidence"] = 0.0
        signals["probability_up"] = 0.0

        prob_up = predictions["probability_up"]
        signals.loc[predictions.index, "probability_up"] = prob_up
        signals.loc[predictions.index, "confidence"] = np.maximum(
            predictions["probability_up"], predictions["probability_down"]
        )
        signals.loc[predictions.index, "position"] = np.where(
            prob_up >= self.long_threshold,
            1.0,
            np.where(prob_up <= self.short_threshold, -1.0, 0.0),
        )

        return signals[["position", "confidence", "probability_up"]]

    def get_model_info(self) -> Dict[str, Any]:
        """Return a summary of the strategy configuration."""
        return {
            "long_threshold": self.long_threshold,
            "short_threshold": self.short_threshold,
            "feature_columns": self.rf_model.feature_columns,
        }
