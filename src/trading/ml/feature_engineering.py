"""Feature engineering for the naive Random Forest baseline.

Extracts a small, curated set of *stationary* technical features from OHLCV
data. The goal is a compact, explainable feature set that is easy to test and
does not leak future information.

The 10 baseline features (all stationary — no raw price/volume levels):
    momentum_5d      5-day cumulative return
    momentum_20d     20-day cumulative return
    rsi              14-period Relative Strength Index
    macd_signal_pct  MACD signal line normalised by close
    bb_percent       position of close within Bollinger Bands (%B)
    bb_bandwidth     Bollinger Band width relative to the middle band
    atr_percent      Average True Range as a fraction of close
    volatility_20d   annualised realised volatility, 20-day window
    volatility_60d   annualised realised volatility, 60-day window
    vwap_ratio       close divided by rolling VWAP
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


class FeatureEngineering:
    """Extract a curated set of stationary technical features from price data."""

    #: The fixed baseline feature set produced by :meth:`generate_features`.
    FEATURE_COLUMNS: List[str] = [
        "momentum_5d",
        "momentum_20d",
        "rsi",
        "macd_signal_pct",
        "bb_percent",
        "bb_bandwidth",
        "atr_percent",
        "volatility_20d",
        "volatility_60d",
        "vwap_ratio",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineering.

        Args:
            config: Optional overrides for indicator parameters:
                - rsi_period (default 14)
                - macd_fast / macd_slow / macd_signal (default 12 / 26 / 9)
                - bb_period / bb_std (default 20 / 2.0)
                - atr_period (default 14)
                - vwap_period (default 20)
        """
        self.config = {**self._default_config(), **(config or {})}

    def _default_config(self) -> Dict[str, Any]:
        return {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            "vwap_period": 20,
        }

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate the 10 baseline features from OHLCV data.

        Args:
            df: DataFrame with columns open, high, low, close, volume.

        Returns:
            Copy of ``df`` with the baseline feature columns appended.
        """
        result = df.copy()
        cfg = self.config

        returns = result["close"].pct_change()

        # Momentum (cumulative returns over horizon)
        result["momentum_5d"] = result["close"].pct_change(periods=5)
        result["momentum_20d"] = result["close"].pct_change(periods=20)

        # RSI
        result["rsi"] = self._rsi(result["close"], cfg["rsi_period"])

        # MACD signal line, normalised by price (stationary)
        ema_fast = result["close"].ewm(span=cfg["macd_fast"], adjust=False).mean()
        ema_slow = result["close"].ewm(span=cfg["macd_slow"], adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=cfg["macd_signal"], adjust=False).mean()
        result["macd_signal_pct"] = macd_signal / result["close"]

        # Bollinger Bands: %B and bandwidth (both stationary)
        bb_mid = result["close"].rolling(window=cfg["bb_period"]).mean()
        bb_std = result["close"].rolling(window=cfg["bb_period"]).std()
        bb_upper = bb_mid + cfg["bb_std"] * bb_std
        bb_lower = bb_mid - cfg["bb_std"] * bb_std
        band_range = (bb_upper - bb_lower).replace(0, np.nan)
        result["bb_percent"] = (result["close"] - bb_lower) / band_range
        result["bb_bandwidth"] = (bb_upper - bb_lower) / bb_mid

        # ATR as a fraction of price
        result["atr_percent"] = self._atr(result, cfg["atr_period"]) / result["close"]

        # Annualised realised volatility at two horizons
        result["volatility_20d"] = returns.rolling(window=20).std() * np.sqrt(252)
        result["volatility_60d"] = returns.rolling(window=60).std() * np.sqrt(252)

        # Close relative to rolling VWAP (stationary; uses a rolling, not
        # cumulative, window to avoid drift over long histories)
        result["vwap_ratio"] = self._vwap_ratio(result, cfg["vwap_period"])

        return result

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def _vwap_ratio(df: pd.DataFrame, period: int) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        pv = (typical_price * df["volume"]).rolling(window=period).sum()
        vol = df["volume"].rolling(window=period).sum().replace(0, np.nan)
        vwap = pv / vol
        return df["close"] / vwap

    def get_feature_names(
        self, df: Optional[pd.DataFrame] = None, target_col: str = "target"
    ) -> List[str]:
        """Return the baseline feature column names.

        Args:
            df: Optional DataFrame; if given, only columns actually present
                are returned. ``target_col`` is always excluded.
            target_col: Target column name to exclude.

        Returns:
            List of feature column names.
        """
        names = [c for c in self.FEATURE_COLUMNS if c != target_col]
        if df is not None:
            names = [c for c in names if c in df.columns]
        return names

    def prepare_for_ml(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        forward_periods: int = 1,
        return_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """Attach a binary direction target and drop unusable rows.

        The target is 1 if the close ``forward_periods`` bars ahead is higher
        than the current close, else 0. Rows with NaN features (from rolling
        windows) or a NaN target (end of series) are dropped.

        Args:
            df: DataFrame with features generated.
            target_col: Name for the created target column.
            forward_periods: Bars ahead to predict.
            return_threshold: If > 0, drop rows whose forward return is within
                +/- this threshold (ambiguous / noise labels).

        Returns:
            DataFrame ready for a leakage-safe split. Never use a shuffled
            split — use :meth:`train_test_split` or
            ``RandomForestAnalyzer.prepare_data``.
        """
        result = df.copy()
        self._last_forward_periods = forward_periods

        fwd_return = result["close"].shift(-forward_periods) / result["close"] - 1
        result[target_col] = (fwd_return > 0).astype(int)

        feature_cols = self.get_feature_names(result, target_col=target_col)
        result = result.dropna(subset=feature_cols + [target_col])

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

        Because each row's target looks ``forward_periods`` bars ahead, the
        last ``forward_periods`` rows of the training slice would reference
        prices inside the test window. A gap of that size is removed at the
        boundary.

        Args:
            df: Time-ordered DataFrame from :meth:`prepare_for_ml`.
            test_size: Fraction of rows reserved for the test set.
            forward_periods: Gap size; defaults to the value from the last
                :meth:`prepare_for_ml` call, or 1.

        Returns:
            ``(train_df, test_df)`` with no target leakage across the boundary.
            Fit scalers on ``train_df`` only.
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
