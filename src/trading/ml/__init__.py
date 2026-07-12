"""Machine learning module for algorithmic trading.

A deliberately simple, testable baseline:
- Curated stationary feature engineering from OHLCV data
- A small, regularised Random Forest classifier for price direction
- A long/short trading strategy driven by the model's probabilities
"""

from src.trading.ml.feature_engineering import FeatureEngineering
from src.trading.ml.random_forest_analyzer import RandomForestAnalyzer
from src.trading.ml.ml_strategy import RandomForestStrategy

__all__ = [
    "FeatureEngineering",
    "RandomForestAnalyzer",
    "RandomForestStrategy",
]
