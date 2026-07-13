"""Unit tests for date-based purged walk-forward panel splitting."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.ml.panel_split import (
    assert_no_leakage,
    panel_train_val_test_split,
    panel_walk_forward_splits,
)


def _make_panel(n_dates=60, symbols=("AAA", "BBB", "CCC")):
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="B")
    rows = [{"date": d, "symbol": s} for d in dates for s in symbols]
    return pd.DataFrame(rows)


class TestPanelWalkForwardSplits:
    def test_every_symbol_in_each_fold(self):
        panel = _make_panel()
        symbols = set(panel["symbol"].unique())
        folds = panel_walk_forward_splits(panel, n_splits=4, embargo_periods=2)

        assert len(folds) == 4
        for train_dates, test_dates in folds:
            train_symbols = set(panel.loc[panel["date"].isin(train_dates), "symbol"])
            test_symbols = set(panel.loc[panel["date"].isin(test_dates), "symbol"])
            assert train_symbols == symbols
            assert test_symbols == symbols

    def test_no_leakage_across_folds(self):
        panel = _make_panel()
        all_dates = np.sort(panel["date"].unique())
        embargo = 2
        folds = panel_walk_forward_splits(panel, n_splits=4, embargo_periods=embargo)

        for train_dates, test_dates in folds:
            assert_no_leakage(all_dates, train_dates, test_dates, embargo)

    def test_leakage_detected_when_embargo_too_large(self):
        panel = _make_panel()
        all_dates = np.sort(panel["date"].unique())
        folds = panel_walk_forward_splits(panel, n_splits=4, embargo_periods=2)
        train_dates, test_dates = folds[0]

        with pytest.raises(ValueError):
            assert_no_leakage(all_dates, train_dates, test_dates, embargo_periods=999)

    def test_folds_are_chronological(self):
        panel = _make_panel()
        folds = panel_walk_forward_splits(panel, n_splits=3, embargo_periods=1)
        for train_dates, test_dates in folds:
            assert train_dates.max() < test_dates.min()


class TestPanelTrainValTestSplit:
    def test_no_overlap_and_ordering(self):
        panel = _make_panel()
        splits = panel_train_val_test_split(
            panel, test_size=0.2, val_size=0.1, embargo_periods=2
        )
        assert splits["train_dates"].max() < splits["val_dates"].min()
        assert splits["val_dates"].max() < splits["test_dates"].min()

    def test_covers_every_symbol(self):
        panel = _make_panel()
        symbols = set(panel["symbol"].unique())
        splits = panel_train_val_test_split(
            panel, test_size=0.2, val_size=0.1, embargo_periods=2
        )
        for key in ("train_dates", "val_dates", "test_dates"):
            present = set(panel.loc[panel["date"].isin(splits[key]), "symbol"])
            assert present == symbols

    def test_leakage_guard_raises_on_too_few_dates(self):
        panel = _make_panel(n_dates=5)
        with pytest.raises(ValueError):
            panel_train_val_test_split(
                panel, test_size=0.2, val_size=0.1, embargo_periods=3
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
