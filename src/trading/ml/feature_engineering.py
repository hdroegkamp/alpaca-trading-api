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
        # Columns that are raw price / volume levels (non-stationary) or
        # intermediate building blocks.  They must NOT be used as ML features
        # because their absolute values drift over time and cause overfitting.
        self._non_stationary_cols: set = set()

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

        # Multi-horizon momentum and volatility features
        result = self._add_momentum_features(result)

        return result

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features."""
        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Price ranges
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        df["close_open_range"] = (df["close"] - df["open"]) / df["open"]

        # Typical price (intermediate — used by VWAP, not a direct feature)
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        self._non_stationary_cols.add("typical_price")

        return df

    def _add_sma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple Moving Average features."""
        for period in self.config["sma_periods"]:
            col_name = f"sma_{period}"
            df[col_name] = df["close"].rolling(window=period).mean()
            self._non_stationary_cols.add(col_name)

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
            self._non_stationary_cols.add(col_name)

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
        # Raw MACD / signal scale with price level — mark as non-stationary.
        self._non_stationary_cols.update(["macd", "macd_signal", "macd_histogram"])

        # Normalised versions (percentage of close) — these are stationary.
        df["macd_pct"] = df["macd"] / df["close"]
        df["macd_signal_pct"] = df["macd_signal"] / df["close"]
        df["macd_histogram_pct"] = df["macd_histogram"] / df["close"]

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
        # Raw band levels are non-stationary — only used to derive %B / bandwidth
        self._non_stationary_cols.update(["bb_middle", "bb_upper", "bb_lower"])

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
        self._non_stationary_cols.add("atr")  # absolute value — use atr_percent
        df["atr_percent"] = df["atr"] / df["close"]

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume SMA
        vol_period = self.config["volume_sma_period"]
        df["volume_sma"] = df["volume"].rolling(window=vol_period).mean()
        self._non_stationary_cols.add("volume_sma")
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Volume changes
        df["volume_change"] = df["volume"].pct_change()

        # On-Balance Volume — use rate-of-change instead of cumulative sum,
        # because the raw cumsum grows monotonically and leaks time information.
        obv_direction = df["close"].diff().apply(np.sign)
        obv_raw = (obv_direction * df["volume"]).cumsum()
        df["obv_roc"] = obv_raw.pct_change(periods=vol_period)

        # Volume-weighted average price — use price/VWAP ratio instead of raw
        # VWAP, which is a cumulative running average that drifts with price.
        vwap = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["vwap_ratio"] = df["close"] / vwap

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

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-horizon momentum and rolling volatility features.

        These capture price momentum and volatility regime at multiple
        timescales, which are among the strongest stationary predictors
        of forward returns in equity markets.
        """
        for window in [5, 10, 20, 60, 120]:
            # Rolling cumulative return over window
            df[f"momentum_{window}d"] = df["close"].pct_change(periods=window)

            # Rolling realized volatility (annualized)
            df[f"volatility_{window}d"] = df["returns"].rolling(
                window=window
            ).std() * np.sqrt(252)

        # Z-score of price relative to rolling mean (mean-reversion signal)
        for window in [5, 10, 20, 60]:
            rolling_mean = df["close"].rolling(window=window).mean()
            rolling_std = df["close"].rolling(window=window).std()
            df[f"zscore_{window}d"] = (df["close"] - rolling_mean) / rolling_std

        # Rate of change of RSI (momentum of momentum)
        df["rsi_roc"] = df["rsi"].diff(periods=5)

        # Volatility ratio: short-term / long-term realized vol
        vol_5 = df["returns"].rolling(window=5).std()
        vol_20 = df["returns"].rolling(window=20).std()
        vol_60 = df["returns"].rolling(window=60).std()
        df["vol_ratio_5_20"] = vol_5 / vol_20
        df["vol_ratio_20_60"] = vol_20 / vol_60

        # Sharpe-like ratio: risk-adjusted momentum
        for window in [20, 60]:
            ret_mean = df["returns"].rolling(window=window).mean()
            ret_std = df["returns"].rolling(window=window).std()
            df[f"sharpe_{window}d"] = (ret_mean / ret_std) * np.sqrt(252)

        # Drawdown from rolling max (regime indicator)
        rolling_max_60 = df["close"].rolling(window=60).max()
        df["drawdown_60d"] = (df["close"] - rolling_max_60) / rolling_max_60

        rolling_max_120 = df["close"].rolling(window=120).max()
        df["drawdown_120d"] = (df["close"] - rolling_max_120) / rolling_max_120

        # Up-day ratio over window (directional persistence)
        for window in [10, 20]:
            df[f"up_ratio_{window}d"] = (
                (df["returns"] > 0).rolling(window=window).mean()
            )

        return df

    def get_feature_names(
        self, df: pd.DataFrame, target_col: str = "target"
    ) -> List[str]:
        """Get list of generated feature column names.

        Args:
            df: DataFrame with features generated
            target_col: Target column name to exclude

        Returns:
            List of feature column names (excludes OHLCV, timestamp, target,
            and non-stationary columns)
        """
        exclude = {"open", "high", "low", "close", "volume", "timestamp", "symbol"}
        exclude |= self._non_stationary_cols
        exclude.add(target_col)
        return [col for col in df.columns if col not in exclude]

    def prepare_for_ml(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        forward_periods: int = 1,
        return_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """Prepare dataset for ML training.

        Args:
            df: DataFrame with features
            target_col: Name for target column
            forward_periods: Periods ahead to predict
            return_threshold: Minimum absolute return to keep a row.
                Rows where the forward return is between -threshold and
                +threshold are dropped (ambiguous / noise). Set to 0.0
                to keep all rows (original behaviour).

        Returns:
            DataFrame ready for train/test split with target column.
            Use train_test_split() from this class — naive shuffled splits
            leak future prices into training labels because each row's target
            references a price forward_periods bars ahead.
        """
        result = df.copy()
        self._last_forward_periods = forward_periods

        # Forward return
        fwd_return = result["close"].shift(-forward_periods) / result["close"] - 1

        # Create target: 1 if price goes up, 0 if down
        result[target_col] = (fwd_return > 0).astype(int)

        # Drop rows with NaN (from rolling windows and target shift)
        result = result.dropna()

        # Drop ambiguous rows near zero return
        if return_threshold > 0:
            mask = fwd_return.reindex(result.index).abs() >= return_threshold
            result = result.loc[mask]

        return result

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        forward_periods: Optional[int] = None,
    ) -> tuple:
        """Chronological train/test split with a gap to prevent leakage.

        Because each row's target looks forward_periods bars ahead, the last
        forward_periods rows of the training slice have targets that reference
        prices in the test window. This method removes that overlap by
        inserting a gap of forward_periods rows at the train/test boundary.

        Args:
            df: Time-ordered DataFrame produced by prepare_for_ml().
            test_size: Fraction of rows reserved for the test set.
            forward_periods: Gap size at the boundary. Defaults to the value
                from the last prepare_for_ml() call, or 1.

        Returns:
            (train_df, test_df) — chronologically ordered with no target
            leakage across the boundary. Fit any scalers/encoders on
            train_df only, then transform test_df without refitting.
        """
        gap = (
            forward_periods
            if forward_periods is not None
            else getattr(self, "_last_forward_periods", 1)
        )
        n = len(df)
        split_idx = int(n * (1 - test_size))

        if split_idx - gap < 1:
            raise ValueError(
                f"Dataset too small: split_idx={split_idx}, gap={gap}. "
                "Reduce test_size or forward_periods."
            )

        train = df.iloc[: split_idx - gap].copy()
        test = df.iloc[split_idx:].copy()
        return train, test

    def select_features(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        max_correlation: float = 0.85,
        min_target_correlation: float = 0.0,
    ) -> List[str]:
        """Select features by removing highly correlated pairs and low-signal columns.

        Uses a greedy approach: among correlated pairs, keep the feature with
        higher absolute correlation to the target.

        Args:
            df: DataFrame with features and target column.
            target_col: Target column name.
            max_correlation: Drop one of any pair with |corr| above this.
            min_target_correlation: Drop features with |corr to target| below this.

        Returns:
            Filtered list of feature column names.
        """
        feature_cols = self.get_feature_names(df, target_col=target_col)
        sub = df[feature_cols + [target_col]].dropna()

        if len(sub) < 30:
            return feature_cols

        # Correlation with target
        target_corr = sub[feature_cols].corrwith(sub[target_col]).abs()

        # Drop features with negligible target correlation
        keep = target_corr[target_corr >= min_target_correlation].index.tolist()
        if not keep:
            return feature_cols  # fallback: keep all

        # Pairwise feature correlation matrix
        corr_matrix = sub[keep].corr().abs()
        corr_values = corr_matrix.to_numpy()

        # Greedy removal: for each highly-correlated pair, drop the one with
        # lower absolute target correlation
        to_drop: set = set()
        for i in range(len(keep)):
            if keep[i] in to_drop:
                continue
            for j in range(i + 1, len(keep)):
                if keep[j] in to_drop:
                    continue
                if corr_values[i, j] > max_correlation:
                    # Drop the feature less correlated with target
                    if target_corr[keep[i]] < target_corr[keep[j]]:
                        to_drop.add(keep[i])
                        break  # feature i is dropped, move on
                    else:
                        to_drop.add(keep[j])

        selected = [f for f in keep if f not in to_drop]
        return selected

    def get_feature_importance_names(self) -> Dict[str, str]:
        """Get human-readable names for features.

        Returns:
            Dict mapping feature names to descriptions
        """
        return {
            "returns": "Daily Returns",
            "rsi": "RSI (14)",
            "macd_pct": "MACD % of Close",
            "macd_histogram_pct": "MACD Histogram % of Close",
            "bb_percent": "Bollinger %B",
            "bb_bandwidth": "Bollinger Bandwidth",
            "atr_percent": "ATR %",
            "volume_ratio": "Volume Ratio",
            "obv_roc": "OBV Rate-of-Change",
            "vwap_ratio": "Price/VWAP Ratio",
            "sma_50_distance": "Distance from SMA(50)",
            "ema_12_ratio": "Price/EMA(12) Ratio",
        }
