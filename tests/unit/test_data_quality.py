"""Unit tests for adjusted-history merging and archive data-quality checks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.data.quality import (
    audit_symbols,
    summarize_audit,
    suspect_moves,
    unadjusted_symbols,
)
from src.trading.data.storage import DataStore
from src.trading.data.update import merge_new_bars


def _bars(start, n, start_price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B", tz="UTC")
    close = start_price * (1 + rng.normal(0, 0.01, n)).cumprod()
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


class TestMergeNewBars:
    def test_appends_only_bars_after_existing_when_overlap_matches(self):
        full = _bars("2024-01-01", 30)
        existing, fetched = full.iloc[:20], full.iloc[15:]

        merged = merge_new_bars(existing, fetched)

        assert merged is not None
        pd.testing.assert_frame_equal(merged, full)

    def test_rescaled_overlap_requires_full_redownload(self):
        # A 2:1 split after the stored window: the provider halves every past price.
        full = _bars("2024-01-01", 30)
        existing = full.iloc[:20]
        fetched = full.iloc[15:].copy()
        fetched[["open", "high", "low", "close"]] /= 2

        assert merge_new_bars(existing, fetched) is None

    def test_small_dividend_readjustment_is_detected(self):
        full = _bars("2024-01-01", 30)
        existing = full.iloc[:20]
        fetched = full.iloc[15:].copy()
        fetched["close"] *= 0.995  # ~0.5% dividend adjustment

        assert merge_new_bars(existing, fetched) is None

    def test_no_overlap_requires_full_redownload(self):
        full = _bars("2024-01-01", 30)
        assert merge_new_bars(full.iloc[:20], full.iloc[22:]) is None

    def test_nothing_new_returns_existing(self):
        full = _bars("2024-01-01", 30)
        merged = merge_new_bars(full, full.iloc[25:])
        assert merged is full

    def test_empty_fetch_returns_existing(self):
        full = _bars("2024-01-01", 30)
        assert merge_new_bars(full, full.iloc[0:0]) is full

    def test_new_rows_take_existing_dtypes(self):
        full = _bars("2024-01-01", 30)
        existing = full.iloc[:20]
        fetched = full.iloc[15:].copy()
        fetched["volume"] = fetched["volume"].astype("int64")

        merged = merge_new_bars(existing, fetched)

        assert merged["volume"].dtype == existing["volume"].dtype


class TestSuspectMoves:
    def test_flags_split_shaped_drop(self):
        data = _bars("2024-01-01", 30)
        data.iloc[20:, data.columns.get_loc("close")] /= 10  # unadjusted 10:1 split

        moves = suspect_moves(data, max_abs_return=0.5)

        assert list(moves.index) == [data.index[20]]
        assert moves.iloc[0] == pytest.approx(-0.9, abs=0.05)

    def test_normal_series_has_none(self):
        assert suspect_moves(_bars("2024-01-01", 100)).empty


@pytest.fixture
def store(tmp_path):
    store = DataStore(data_dir=str(tmp_path))
    store.save("ADJ", _bars("2024-01-01", 50), timeframe="1Day", adjustment="all")
    raw = _bars("2024-01-01", 50, seed=1)
    raw.iloc[30:, raw.columns.get_loc("close")] /= 4
    store.save("RAW", raw, timeframe="1Day")  # no adjustment recorded
    return store


class TestAuditSymbols:
    def test_reports_adjustment_and_suspect_moves(self, store):
        audit = audit_symbols(store, ["ADJ", "RAW", "MISSING"], "1Day").set_index("symbol")

        assert list(audit.index) == ["ADJ", "RAW"]
        assert audit.loc["ADJ", "adjustment"] == "all"
        assert audit.loc["ADJ", "n_suspect"] == 0
        assert pd.isna(audit.loc["RAW", "adjustment"])
        assert audit.loc["RAW", "n_suspect"] == 1
        assert audit.loc["RAW", "max_abs_return"] == pytest.approx(0.75, abs=0.05)

    def test_unadjusted_symbols(self, store):
        audit = audit_symbols(store, ["ADJ", "RAW"], "1Day")
        assert unadjusted_symbols(audit) == ["RAW"]

    def test_summary_is_empty_when_clean(self, store):
        assert summarize_audit(audit_symbols(store, ["ADJ"], "1Day")) == []

    def test_summary_names_problem_symbols(self, store):
        lines = summarize_audit(audit_symbols(store, ["ADJ", "RAW"], "1Day"))
        assert len(lines) == 2
        assert all("RAW" in line for line in lines)
