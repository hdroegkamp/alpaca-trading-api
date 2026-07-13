"""Machine learning module for algorithmic trading.

A deliberately simple, testable baseline:
- Curated stationary feature engineering from OHLCV data
- A small, regularised Random Forest classifier for price direction
- A long/short trading strategy driven by the model's probabilities

Plus a pooled, cross-sectional path for ranking a whole universe on a given
day (see ``panel``, ``cross_sectional_labels``, ``panel_split``, and
``pooled_model``), which the single-symbol classes above can't do.
"""

from src.trading.ml.feature_engineering import FeatureEngineering
from src.trading.ml.random_forest_analyzer import RandomForestAnalyzer
from src.trading.ml.ml_strategy import RandomForestStrategy
from src.trading.ml.panel import build_panel
from src.trading.ml.cross_sectional_labels import add_cross_sectional_labels
from src.trading.ml.panel_split import (
    assert_no_leakage,
    panel_train_val_test_split,
    panel_walk_forward_splits,
)
from src.trading.ml.pooled_model import PooledForestModel

__all__ = [
    "FeatureEngineering",
    "RandomForestAnalyzer",
    "RandomForestStrategy",
    "build_panel",
    "add_cross_sectional_labels",
    "assert_no_leakage",
    "panel_train_val_test_split",
    "panel_walk_forward_splits",
    "PooledForestModel",
]
