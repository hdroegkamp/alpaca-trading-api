"""Feature engineering for ML-based trading strategies.

Extracts technical indicators and derived features from OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


class FeatureEngineering:
    """Extract technical indicators and features from price data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineering.

        Args:
            config: Configuration dict with feature parameters
                - sma_periods: List of SMA window sizes
                - ema_periods: List of EMA window sizes
                - rsi_period: RSI lookback period
                - macd_config: MACD parameters (fast, slow, signal)
                - bb_period: Bollinger Bands period
                - bb_std: Bollinger Bands standard deviations
        """
        self.config = config or self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default feature configuration."""
        return {
            "sma_periods": [10, 20, 50, 100, 200],
            "ema_periods": [9, 12, 20, 26, 50],
            "rsi_period": 14,
            "macd_config": {"fast": 12, "slow": 26, "signal": 9},
            "bb_period": 20,
            "bb_std": 2.0,
            "volume_sma_period": 20,
        }

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all technical indicator features.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            DataFrame with original data plus all technical features
        """
        result = df.copy()

        # Price-based features
        result = self._add_price_features(result)

        # Moving averages
        result = self._add_sma_features(result)
        result = self._add_ema_features(result)

        # Momentum indicators
        result = self._add_rsi(result)
        result = self._add_macd(result)

        # Volatility indicators
        result = self._add_bollinger_bands(result)
        result = self._add_atr(result)

        # Volume indicators
        result = self._add_volume_features(result)

        # Price patterns
        result = self._add_pattern_features(result)

        # Lagged features
        result = self._add_lagged_features(result)

        return result

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features."""
        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Price ranges
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        df["close_open_range"] = (df["close"] - df["open"]) / df["open"]

        # Typical price
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        return df

    def _add_sma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple Moving Average features."""
        for period in self.config["sma_periods"]:
            col_name = f"sma_{period}"
            df[col_name] = df["close"].rolling(window=period).mean()

            # Price relative to SMA
            df[f"{col_name}_ratio"] = df["close"] / df[col_name]

            # Distance from SMA
            df[f"{col_name}_distance"] = (df["close"] - df[col_name]) / df["close"]

        # SMA crossover signals
        if 50 in self.config["sma_periods"] and 200 in self.config["sma_periods"]:
            df["sma_50_200_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)

        return df

    def _add_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Exponential Moving Average features."""
        for period in self.config["ema_periods"]:
            col_name = f"ema_{period}"
            df[col_name] = df["close"].ewm(span=period, adjust=False).mean()

            # Price relative to EMA
            df[f"{col_name}_ratio"] = df["close"] / df[col_name]

        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Relative Strength Index."""
        period = self.config["rsi_period"]

        # Calculate price changes
        delta = df["close"].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # RSI zones
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator."""
        config = self.config["macd_config"]

        # Calculate MACD
        ema_fast = df["close"].ewm(span=config["fast"], adjust=False).mean()
        ema_slow = df["close"].ewm(span=config["slow"], adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=config["signal"], adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # MACD crossover signal
        df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)

        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands."""
        period = self.config["bb_period"]
        std_dev = self.config["bb_std"]

        # Calculate middle band (SMA)
        df["bb_middle"] = df["close"].rolling(window=period).mean()

        # Calculate standard deviation
        rolling_std = df["close"].rolling(window=period).std()

        # Calculate upper and lower bands
        df["bb_upper"] = df["bb_middle"] + (rolling_std * std_dev)
        df["bb_lower"] = df["bb_middle"] - (rolling_std * std_dev)

        # Bandwidth and %B
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_percent"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        return df

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range (volatility indicator)."""
        # True Range calculation
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR
        df["atr"] = true_range.rolling(window=period).mean()
        df["atr_percent"] = df["atr"] / df["close"]

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume SMA
        vol_period = self.config["volume_sma_period"]
        df["volume_sma"] = df["volume"].rolling(window=vol_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Volume changes
        df["volume_change"] = df["volume"].pct_change()

        # On-Balance Volume (OBV)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()

        # Volume-weighted average price
        df["vwap"] = (df["typical_price"] * df["volume"]).cumsum() / df[
            "volume"
        ].cumsum()

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        # Body size
        df["candle_body"] = abs(df["close"] - df["open"]) / df["open"]

        # Upper and lower shadows
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df[
            "open"
        ]
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df[
            "open"
        ]

        # Candle direction
        df["bullish_candle"] = (df["close"] > df["open"]).astype(int)

        # Doji detection (small body relative to range)
        df["is_doji"] = (df["candle_body"] < 0.001).astype(int)

        return df

    def _add_lagged_features(
        self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """Add lagged features for time series context."""
        # Lagged returns
        for lag in lags:
            df[f"returns_lag_{lag}"] = df["returns"].shift(lag)

        # Lagged RSI
        for lag in [1, 2, 3]:
            df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag)

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of generated feature column names.

        Args:
            df: DataFrame with features generated

        Returns:
            List of feature column names (excludes OHLCV and timestamp)
        """
        base_cols = ["open", "high", "low", "close", "volume", "timestamp", "symbol"]
        return [col for col in df.columns if col not in base_cols]

    def prepare_for_ml(
        self, df: pd.DataFrame, target_col: str = "target", forward_periods: int = 1
    ) -> pd.DataFrame:
        """Prepare dataset for ML training.

        Args:
            df: DataFrame with features
            target_col: Name for target column
            forward_periods: Periods ahead to predict

        Returns:
            DataFrame ready for train/test split with target column
        """
        result = df.copy()

        # Create target: 1 if price goes up, 0 if down
        result[target_col] = (
            result["close"].shift(-forward_periods) > result["close"]
        ).astype(int)

        # Drop rows with NaN (from rolling windows and target shift)
        result = result.dropna()

        return result

    def get_feature_importance_names(self) -> Dict[str, str]:
        """Get human-readable names for features.

        Returns:
            Dict mapping feature names to descriptions
        """
        return {
            "returns": "Daily Returns",
            "rsi": "RSI (14)",
            "macd": "MACD Line",
            "macd_histogram": "MACD Histogram",
            "bb_percent": "Bollinger %B",
            "bb_bandwidth": "Bollinger Bandwidth",
            "atr_percent": "ATR %",
            "volume_ratio": "Volume Ratio",
            "sma_50_distance": "Distance from SMA(50)",
            "ema_12_ratio": "Price/EMA(12) Ratio",
        }
