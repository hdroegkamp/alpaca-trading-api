"""Unit tests for cross-sectional panel assembly and labeling."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.data.storage import DataStore
from src.trading.ml.panel import build_panel
from src.trading.ml.cross_sectional_labels import add_cross_sectional_labels


def _make_ohlcv(n_days=150, start_price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    returns = rng.normal(0, 0.02, n_days)
    close = start_price * (1 + returns).cumprod()
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.005, n_days)),
            "high": close * (1 + abs(rng.normal(0, 0.01, n_days))),
            "low": close * (1 - abs(rng.normal(0, 0.01, n_days))),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n_days),
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


@pytest.fixture
def data_store(tmp_path):
    store = DataStore(data_dir=str(tmp_path))
    for i, sym in enumerate(SYMBOLS):
        store.save(sym, _make_ohlcv(seed=i), timeframe="1Day")
    return store


class TestBuildPanel:
    def test_shape_and_columns(self, data_store):
        panel = build_panel(SYMBOLS, data_store, timeframe="1Day", min_history=10)

        assert {"date", "symbol", "close", "forward_return"}.issubset(panel.columns)
        assert set(SYMBOLS) == set(panel["symbol"].unique())
        assert panel.sort_values(["date", "symbol"]).reset_index(drop=True).equals(panel)

    def test_skips_missing_symbol(self, data_store):
        panel = build_panel(
            SYMBOLS + ["ZZZ"], data_store, timeframe="1Day", min_history=10
        )
        assert "ZZZ" not in panel["symbol"].unique()

    def test_cache_roundtrip(self, data_store, tmp_path):
        cache_path = tmp_path / "panel_cache.parquet"
        panel1 = build_panel(
            SYMBOLS, data_store, timeframe="1Day", min_history=10, cache_path=str(cache_path)
        )
        assert cache_path.exists()

        panel2 = build_panel(
            ["should-not-be-read"],
            data_store,
            timeframe="1Day",
            cache_path=str(cache_path),
        )
        pd.testing.assert_frame_equal(panel1, panel2)


class TestCrossSectionalLabels:
    def test_top_bottom_counts_by_date(self, data_store):
        panel = build_panel(SYMBOLS, data_store, timeframe="1Day", min_history=10)
        labeled = add_cross_sectional_labels(
            panel, top_quantile=1 / 3, bottom_quantile=1 / 3, min_symbols_per_date=6
        )
        counts = labeled.groupby("date")["target"].value_counts().unstack(fill_value=0)
        # 6 symbols, 1/3 quantile => exactly 2 per leg per date.
        assert (counts[1] == 2).all()
        assert (counts[0] == 2).all()

    def test_drop_middle_false_keeps_nan(self, data_store):
        panel = build_panel(SYMBOLS, data_store, timeframe="1Day", min_history=10)
        labeled = add_cross_sectional_labels(
            panel,
            top_quantile=1 / 3,
            bottom_quantile=1 / 3,
            drop_middle=False,
            min_symbols_per_date=6,
        )
        assert labeled["target"].isna().any()

    def test_min_symbols_per_date_drops_thin_dates(self, data_store):
        panel = build_panel(SYMBOLS, data_store, timeframe="1Day", min_history=10)
        labeled = add_cross_sectional_labels(panel, min_symbols_per_date=10_000)
        assert labeled.empty

    def test_rejects_quantiles_over_one(self, data_store):
        panel = build_panel(SYMBOLS, data_store, timeframe="1Day", min_history=10)
        with pytest.raises(ValueError):
            add_cross_sectional_labels(panel, top_quantile=0.6, bottom_quantile=0.6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
