"""Target portfolio construction: turn one day's ranked scores into weights.

Single-cross-section counterpart to ``PortfolioBacktest._build_book``
(``src.trading.backtest.portfolio``) — same top/bottom-quantile,
dollar-neutral, equal-weight convention, applied to one day's scored universe
instead of a whole panel of rebalance dates.
"""

from typing import Dict

import pandas as pd


def build_target_weights(
    scored: pd.DataFrame,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    symbol_col: str = "symbol",
    score_col: str = "probability_up",
) -> Dict[str, float]:
    """Rank one day's scored universe into a long/short target-weight book.

    Args:
        scored: One cross-section's worth of rows (already filtered to a
            single date), with ``symbol_col`` and ``score_col`` populated.
        top_pct: Fraction of the universe held long.
        bottom_pct: Fraction of the universe held short.
        symbol_col: Column identifying each row's symbol.
        score_col: Column ranked to form the book (higher = more bullish).

    Returns:
        Dict mapping symbol -> weight. Long legs get ``0.5 / n_long``, short
        legs get ``-0.5 / n_short`` (dollar-neutral, equal-weight within each
        leg) — same convention as ``PortfolioBacktest._build_book``. Symbols
        not selected are absent from the dict.

    Raises:
        ValueError: If ``top_pct + bottom_pct > 1.0``, or the universe is too
            small to form both legs without overlap.
    """
    if top_pct + bottom_pct > 1.0:
        raise ValueError("top_pct + bottom_pct must not exceed 1.0")

    n = len(scored)
    n_long = max(1, round(n * top_pct))
    n_short = max(1, round(n * bottom_pct))
    if n_long + n_short > n:
        raise ValueError(
            f"Universe too small ({n} symbols) to form both legs "
            f"(n_long={n_long}, n_short={n_short}) without overlap."
        )

    ranked = scored.sort_values(score_col, ascending=False)
    longs = ranked.iloc[:n_long]
    shorts = ranked.iloc[-n_short:]

    weights: Dict[str, float] = {}
    for sym in longs[symbol_col]:
        weights[sym] = 0.5 / n_long
    for sym in shorts[symbol_col]:
        weights[sym] = -0.5 / n_short
    return weights
