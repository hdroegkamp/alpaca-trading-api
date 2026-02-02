"""Machine learning module for algorithmic trading.

This module provides ML-based trading strategies and predictive models:
- Feature engineering from OHLCV data
- LSTM time series forecasting
- Random Forest classification
- Model evaluation and interpretability
"""

from src.trading.ml.feature_engineering import FeatureEngineering
from src.trading.ml.lstm_predictor import LSTMPredictor
from src.trading.ml.random_forest_analyzer import RandomForestAnalyzer
from src.trading.ml.ml_strategy import (
    MLPredictionStrategy,
    LSTMStrategy,
    RandomForestStrategy,
)

__all__ = [
    "FeatureEngineering",
    "LSTMPredictor",
    "RandomForestAnalyzer",
    "MLPredictionStrategy",
    "LSTMStrategy",
    "RandomForestStrategy",
]
