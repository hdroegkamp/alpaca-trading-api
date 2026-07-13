"""Date-based purged walk-forward splitting for a pooled cross-sectional panel.

Row position in a multi-symbol panel does not correspond to a fixed time
offset the way it does in a single symbol's chronological frame (the pattern
``RandomForestAnalyzer`` relies on), so every split here operates on the
panel's *unique dates*, not row indices, and maps fold boundaries back to row
masks. This keeps every symbol's rows for a given date in the same fold and
applies the leakage-preventing gap in date-space, generalizing the existing
``TimeSeriesSplit(gap=forward_periods)`` pattern
(``random_forest_analyzer.py``) from one series to the whole panel.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def panel_walk_forward_splits(
    panel: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 5,
    embargo_periods: int = 1,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Expanding-window walk-forward folds over a panel's unique dates.

    Args:
        panel: Long cross-sectional panel.
        date_col: Column identifying each cross-section.
        n_splits: Number of walk-forward folds.
        embargo_periods: Gap, in unique dates, enforced between each fold's
            train and test range — size to the label's ``forward_periods``
            so no label at the boundary straddles the split.

    Returns:
        List of ``(train_dates, test_dates)`` tuples, one per fold, in
        chronological order. Map a fold back to rows with
        ``panel[date_col].isin(train_dates)`` / ``.isin(test_dates)``.
    """
    dates = np.sort(panel[date_col].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=embargo_periods)

    folds = []
    for train_idx, test_idx in tscv.split(dates):
        folds.append(
            (pd.DatetimeIndex(dates[train_idx]), pd.DatetimeIndex(dates[test_idx]))
        )
    return folds


def panel_train_val_test_split(
    panel: pd.DataFrame,
    date_col: str = "date",
    test_size: float = 0.2,
    val_size: float = 0.1,
    embargo_periods: int = 1,
) -> Dict[str, pd.DatetimeIndex]:
    """One-shot chronological train/val/test split over a panel's unique dates.

    Date-based analog of ``RandomForestAnalyzer.prepare_data``'s split: a gap
    of ``embargo_periods`` unique dates is removed at each boundary so no
    label straddles a split.

    Args:
        panel: Long cross-sectional panel.
        date_col: Column identifying each cross-section.
        test_size: Fraction of unique dates reserved for testing.
        val_size: Fraction of unique dates reserved for validation.
        embargo_periods: Gap size, in unique dates, at each boundary.

    Returns:
        Dict with ``train_dates``, ``val_dates``, ``test_dates``
        ``DatetimeIndex`` entries.

    Raises:
        ValueError: If too few unique dates remain for the requested split.
    """
    dates = np.sort(panel[date_col].unique())
    n = len(dates)
    test_split = int(n * (1 - test_size))
    val_ratio = val_size / (1 - test_size) if test_size < 1 else 0.0
    val_split = int(test_split * (1 - val_ratio))

    train_end = val_split - embargo_periods
    val_end = test_split - embargo_periods

    if train_end < 1 or val_end <= val_split:
        raise ValueError(
            f"Too few unique dates ({n}) for the requested test_size={test_size}, "
            f"val_size={val_size}, embargo_periods={embargo_periods}."
        )

    return {
        "train_dates": pd.DatetimeIndex(dates[:train_end]),
        "val_dates": pd.DatetimeIndex(dates[val_split:val_end]),
        "test_dates": pd.DatetimeIndex(dates[test_split:]),
    }


def assert_no_leakage(
    all_dates: np.ndarray,
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    embargo_periods: int,
) -> None:
    """Verify at least ``embargo_periods`` unique dates separate train and test.

    Args:
        all_dates: The full universe of unique dates the split was drawn
            from (not just ``train_dates``/``test_dates``) — needed to count
            how many dates actually fall between the two ranges.
        train_dates: Unique dates used for training.
        test_dates: Unique dates used for testing.
        embargo_periods: Minimum number of unique dates required strictly
            between the last training date and the first test date.

    Raises:
        ValueError: If the embargo is violated (including any overlap).
    """
    if len(train_dates) == 0 or len(test_dates) == 0:
        return

    sorted_dates = np.sort(np.unique(all_dates))
    between = (sorted_dates > train_dates.max()) & (sorted_dates < test_dates.min())
    gap = int(between.sum())

    if gap < embargo_periods:
        raise ValueError(
            f"Only {gap} unique date(s) separate train (ends {train_dates.max()}) "
            f"from test (starts {test_dates.min()}); embargo_periods="
            f"{embargo_periods} requires at least that many."
        )
