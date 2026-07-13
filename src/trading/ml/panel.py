"""Cross-sectional panel assembly: a long (date, symbol, features, ...) frame
built across a whole universe of symbols.

A per-symbol model can't rank a universe — its scores were calibrated on that
symbol alone and aren't comparable to any other symbol's. Pooling every
symbol's rows into one long panel, keyed by ``date`` and ``symbol``, is what
makes a single model's scores directly comparable across the universe on a
given day (see :mod:`src.trading.ml.cross_sectional_labels` and
:mod:`src.trading.ml.pooled_model`).

Survivorship-bias caveat: ``symbols`` typically reflects data available
*today* — delisted names are absent, so training/backtesting on this panel is
optimistically biased relative to a true point-in-time universe. This is
documented, not fixed, here.
"""

from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from ..data.storage import DataStore
from .feature_engineering import FeatureEngineering

#: Called as ``callback(done, total, symbol)`` after each symbol is processed.
ProgressCallback = Callable[[int, int, str], None]


def build_panel(
    symbols: List[str],
    data_store: DataStore,
    timeframe: str = "1Day",
    feature_engineer: Optional[FeatureEngineering] = None,
    forward_periods: int = 1,
    min_history: int = 70,
    cache_path: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Build a long cross-sectional panel from per-symbol OHLCV data.

    For each symbol: load OHLCV, generate the stationary feature set
    (:meth:`FeatureEngineering.generate_features`, unchanged), attach a raw
    ``forward_return`` column, then concatenate every symbol into one frame.
    Labeling is a separate step — see
    :func:`src.trading.ml.cross_sectional_labels.add_cross_sectional_labels`.

    Feature and return columns are downcast to float32 and the panel can be
    cached to parquet: with thousands of symbols across years of history the
    assembled panel is tens of millions of rows, and re-running feature
    engineering from raw OHLCV on every call is wasteful once the inputs are
    fixed.

    Args:
        symbols: Universe to include (see ``src.trading.data.universe``).
        data_store: Source of per-symbol OHLCV data.
        timeframe: Bar timeframe to load.
        feature_engineer: Reused across all symbols; a default instance is
            created if omitted.
        forward_periods: Bars ahead for the ``forward_return`` column.
        min_history: Skip symbols with fewer than this many rows remaining
            after dropping NaN feature rows (too little history to be
            useful).
        cache_path: If given and the file already exists, load and return
            the cached panel instead of rebuilding (``symbols`` is then
            ignored). Otherwise the assembled panel is written there as a
            parquet cache for later reuse.
        progress_callback: Optional ``(done, total, symbol)`` hook, e.g. to
            drive a UI progress bar.

    Returns:
        Long DataFrame sorted by ``(date, symbol)`` with columns: date,
        symbol, <FeatureEngineering.FEATURE_COLUMNS>, close, forward_return.

    Raises:
        ValueError: If no symbol produced enough usable history.
    """
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_parquet(cache_path)

    fe = feature_engineer or FeatureEngineering()
    frames = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, start=1):
        raw = data_store.load(symbol, timeframe)
        if raw is None or len(raw) == 0:
            if progress_callback:
                progress_callback(i, total, symbol)
            continue

        features = fe.generate_features(raw)
        feature_cols = fe.get_feature_names(features)
        features["forward_return"] = FeatureEngineering.forward_return(
            features["close"], forward_periods
        )
        features = features.dropna(subset=feature_cols)

        if len(features) < min_history:
            if progress_callback:
                progress_callback(i, total, symbol)
            continue

        date_col = features.index.name or "index"
        rows = features[feature_cols + ["close", "forward_return"]].reset_index()
        rows = rows.rename(columns={date_col: "date"})
        rows[feature_cols + ["forward_return"]] = rows[
            feature_cols + ["forward_return"]
        ].astype("float32")
        rows.insert(1, "symbol", symbol)
        frames.append(rows)

        if progress_callback:
            progress_callback(i, total, symbol)

    if not frames:
        raise ValueError(
            "No symbol produced at least min_history usable rows for the panel."
        )

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache_path, compression="gzip")

    return panel
