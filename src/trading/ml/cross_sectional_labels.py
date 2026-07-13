"""Cross-sectional labeling: rank each date's forward returns and assign
top/bottom targets.

A per-symbol binary target ("will this stock go up") isn't the right label
for a picker — what matters is how a stock ranks against the rest of the
universe that day. This module turns the panel's raw ``forward_return`` into
a per-date rank and a top/bottom classification target.
"""

from typing import Optional

import numpy as np
import pandas as pd


def add_cross_sectional_labels(
    panel: pd.DataFrame,
    date_col: str = "date",
    return_col: str = "forward_return",
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    drop_middle: bool = True,
    target_col: str = "target",
    min_symbols_per_date: int = 5,
) -> pd.DataFrame:
    """Label each date's cross-section by rank of ``return_col``.

    For each date, symbols are ranked by ``return_col`` (rows with a NaN
    return are excluded from ranking and never labeled). The top
    ``round(n * top_quantile)`` names (by count, not a percentile threshold —
    this avoids ambiguous boundary ties) are labeled 1, the bottom
    ``round(n * bottom_quantile)`` are labeled 0. Dates with fewer than
    ``min_symbols_per_date`` valid returns are dropped entirely — too few
    names to form a meaningful cross-section.

    Args:
        panel: Long panel from ``src.trading.ml.panel.build_panel``.
        date_col: Column identifying each cross-section.
        return_col: Column ranked within each date.
        top_quantile: Fraction of each date's cross-section labeled 1.
        bottom_quantile: Fraction of each date's cross-section labeled 0.
        drop_middle: If True (default), rows outside both bands are dropped.
            If False, they are kept with ``target_col`` set to ``pd.NA``.
        target_col: Name for the created binary label column.
        min_symbols_per_date: Minimum valid returns required to label a date.

    Returns:
        Copy of ``panel`` with ``target_col`` (nullable ``Int64``, or plain
        ``int64`` if ``drop_middle`` is True) and a continuous ``cs_rank``
        percentile-rank column (1.0 = best return that date, near 0 = worst)
        added.

    Raises:
        ValueError: If ``top_quantile + bottom_quantile > 1.0``.
    """
    if top_quantile + bottom_quantile > 1.0:
        raise ValueError("top_quantile + bottom_quantile must not exceed 1.0")

    result = panel.copy()
    valid_counts = result.groupby(date_col)[return_col].transform("count")
    result = result.loc[valid_counts >= min_symbols_per_date].copy()

    grouped_return = result.groupby(date_col)[return_col]
    result["cs_rank"] = grouped_return.rank(pct=True, method="first")

    group_size = grouped_return.transform("count")
    n_top = np.maximum(1, np.round(group_size * top_quantile)).astype(int)
    n_bottom = np.maximum(1, np.round(group_size * bottom_quantile)).astype(int)

    rank_desc = grouped_return.rank(ascending=False, method="first")
    rank_asc = grouped_return.rank(ascending=True, method="first")

    result[target_col] = pd.array([pd.NA] * len(result), dtype="Int64")
    result.loc[rank_desc <= n_top, target_col] = 1
    result.loc[rank_asc <= n_bottom, target_col] = 0

    if drop_middle:
        result = result.dropna(subset=[target_col]).copy()
        result[target_col] = result[target_col].astype("int64")

    return result
