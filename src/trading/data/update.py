"""Incremental updates for adjusted price histories.

An adjusted history is rewritten every time a split or dividend occurs: all
bars before the event are rescaled. So "fetch only bars after the last stored
one and append" silently stitches two different price scales together the
first time a corporate action lands in the new window.

The fix used here needs no corporate-actions feed: re-fetch a few bars that
overlap the stored history along with the new ones. If the overlapping closes
still match, nothing was re-adjusted and the new bars can be appended; if they
don't (or there is no overlap to check), the caller must re-download the
symbol's full history.
"""

from typing import Optional

import pandas as pd

#: Relative tolerance for "the overlap still matches". Alpaca's adjusted
#: prices are rounded to a few decimals; a real re-adjustment (even a small
#: dividend, ~0.1%+) is well above this.
DEFAULT_RTOL = 1e-4


def merge_new_bars(
    existing: pd.DataFrame,
    fetched: pd.DataFrame,
    rtol: float = DEFAULT_RTOL,
) -> Optional[pd.DataFrame]:
    """Append ``fetched`` bars to ``existing`` if their overlap still agrees.

    Args:
        existing: Stored OHLCV history (DatetimeIndex, sorted).
        fetched: Freshly fetched bars that should start *before* the end of
            ``existing`` so the two overlap.
        rtol: Max relative difference between overlapping closes.

    Returns:
        ``existing`` + the fetched bars after its last date (cast to
        ``existing``'s dtypes), or ``existing`` unchanged if there is nothing
        new. ``None`` when the history must be re-downloaded in full: the
        overlapping closes disagree (a split/dividend re-adjusted the past),
        or there is no overlap to verify against.
    """
    if fetched.empty:
        return existing

    overlap = existing.index.intersection(fetched.index)
    if len(overlap) == 0:
        return None

    stored = existing.loc[overlap, "close"].astype(float)
    fresh = fetched.loc[overlap, "close"].astype(float)
    rel_diff = ((fresh - stored).abs() / stored.abs()).max()
    if not rel_diff <= rtol:  # also catches NaN
        return None

    new = fetched[fetched.index > existing.index.max()]
    if new.empty:
        return existing
    new = new[existing.columns].astype(existing.dtypes.to_dict())
    return pd.concat([existing, new]).sort_index()
