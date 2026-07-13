"""Portfolio-level backtest: a daily cross-sectional long/short book driven by
walk-forward fold models.

Unlike ``VectorizedBacktest`` (one instrument, one position, one equity
curve), this scores the whole universe on each rebalance date with the fold
model trained only on data before that date, forms a long/short book from
the top/bottom of the ranked scores, and aggregates into a single portfolio
equity curve.

Known limitations (documented here, not fixed):
- Survivorship bias: the panel's universe reflects symbols with data
  available today; delisted names are absent, which optimistically biases
  results relative to a true point-in-time universe.
- Costs are a flat commission fraction on turnover only — no slippage or
  market-impact model. Daily long/short rebalancing across many names
  generates substantial turnover, so results are sensitive to
  ``top_pct``/``bottom_pct`` and ``rebalance_every_n_days``.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import PerformanceMetrics


class PortfolioBacktest:
    """Daily cross-sectional long/short backtest driven by walk-forward fold models."""

    def __init__(
        self,
        panel: pd.DataFrame,
        fold_models: List[Dict[str, Any]],
        feature_columns: List[str],
        date_col: str = "date",
        symbol_col: str = "symbol",
        close_col: str = "close",
        top_pct: float = 0.2,
        bottom_pct: float = 0.2,
        commission: float = 0.001,
        rebalance_every_n_days: int = 1,
        initial_capital: float = 100000.0,
    ):
        """Configure the portfolio backtest.

        Args:
            panel: The full panel used to train ``fold_models`` (must include
                ``close_col`` for realized returns and the columns in
                ``feature_columns``).
            fold_models: The ``fold_metrics`` list from
                ``PooledForestModel.walk_forward_validate_panel`` — each
                fold's model/scaler is only ever used to score that fold's
                own ``test_dates``, so a rebalance date is never scored by a
                model trained on data at or after that date.
            feature_columns: Feature columns the fold models were trained on.
            date_col: Column identifying each cross-section.
            symbol_col: Symbol column.
            close_col: Raw close-price column used for realized P&L (not the
                label's ``forward_return``, so realized results stay
                independent of the label horizon and easy to audit).
            top_pct: Fraction of each rebalance date's eligible universe
                held long.
            bottom_pct: Fraction of each rebalance date's eligible universe
                held short.
            commission: Flat commission fraction applied to turnover, same
                convention as ``VectorizedBacktest``.
            rebalance_every_n_days: Rebalance cadence in trading days within
                each fold's test range (1 = daily); the book is held between
                rebalances.
            initial_capital: Starting portfolio value.
        """
        self.panel = panel
        self.fold_models = fold_models
        self.feature_columns = feature_columns
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.close_col = close_col
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.commission = commission
        self.rebalance_every_n_days = max(1, rebalance_every_n_days)
        self.initial_capital = initial_capital
        self.results: Optional[pd.DataFrame] = None
        self.book: Optional[pd.DataFrame] = None

    def _score_fold(self, fold: Dict[str, Any]) -> pd.DataFrame:
        mask = self.panel[self.date_col].isin(fold["test_dates"])
        rows = self.panel.loc[mask].dropna(subset=self.feature_columns)
        if rows.empty:
            return rows

        X = fold["scaler"].transform(rows[self.feature_columns].values)
        rows = rows.copy()
        rows["score"] = fold["model"].predict_proba(X)[:, 1]
        return rows

    def _build_book(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Rank each date's cross-section and assign equal-weight, dollar-neutral legs."""
        books = []
        for date, group in scored.groupby(self.date_col):
            n = len(group)
            n_long = max(1, int(round(n * self.top_pct)))
            n_short = max(1, int(round(n * self.bottom_pct)))
            if n_long + n_short > n:
                continue  # universe too small to form both legs without overlap

            ranked = group.sort_values("score", ascending=False)
            longs = ranked.iloc[:n_long]
            shorts = ranked.iloc[-n_short:]

            books.append(
                pd.concat(
                    [
                        pd.DataFrame(
                            {
                                self.date_col: date,
                                self.symbol_col: longs[self.symbol_col].values,
                                "side": "long",
                                "weight": 0.5 / n_long,
                            }
                        ),
                        pd.DataFrame(
                            {
                                self.date_col: date,
                                self.symbol_col: shorts[self.symbol_col].values,
                                "side": "short",
                                "weight": -0.5 / n_short,
                            }
                        ),
                    ]
                )
            )

        if not books:
            raise ValueError(
                "No rebalance date produced a valid long/short book — the "
                "universe may be too small for the requested top_pct/bottom_pct."
            )
        return pd.concat(books, ignore_index=True)

    def run(self) -> pd.DataFrame:
        """Run the backtest across every fold's out-of-sample test range.

        Returns:
            DataFrame with ``date, portfolio_return, cumulative_return,
            equity, drawdown``, stitched chronologically across all folds.
            Also populates ``self.book`` (``date, symbol, side, weight``) for
            turnover/auditability.

        Raises:
            ValueError: If no fold produced any scoreable rows.
        """
        scored_folds = [self._score_fold(f) for f in self.fold_models]
        scored = pd.concat(
            [s for s in scored_folds if not s.empty], ignore_index=True
        )
        if scored.empty:
            raise ValueError("No fold produced scoreable rows.")

        all_dates = np.sort(scored[self.date_col].unique())
        rebalance_dates = all_dates[:: self.rebalance_every_n_days]

        book = self._build_book(scored[scored[self.date_col].isin(rebalance_dates)])
        self.book = book

        # Weights held between rebalances: forward-fill the sparse rebalance
        # grid across every scored date, treating pre-first-rebalance as flat.
        weight_wide = (
            book.pivot_table(
                index=self.date_col,
                columns=self.symbol_col,
                values="weight",
                fill_value=0.0,
            )
            .reindex(all_dates, fill_value=np.nan)
            .ffill()
            .fillna(0.0)
        )

        # Next-period return per symbol from raw close, computed over the
        # full panel date range (not just all_dates) so the last scored date
        # still has a "next" price to realize a return against.
        symbols_in_scope = scored[self.symbol_col].unique()
        panel_dates = np.sort(self.panel[self.date_col].unique())
        close_wide = (
            self.panel[self.panel[self.symbol_col].isin(symbols_in_scope)]
            .pivot_table(index=self.date_col, columns=self.symbol_col, values=self.close_col)
            .reindex(panel_dates)
        )
        next_return = (close_wide.shift(-1) / close_wide - 1).reindex(all_dates)

        aligned_weights, aligned_returns = weight_wide.align(
            next_return, join="inner", axis=1
        )
        daily_gross = (aligned_weights * aligned_returns).sum(axis=1, skipna=True)

        # diff() at row 0 is NaN (no prior book) — commission still applies
        # when the initial positions are entered from flat.
        turnover = weight_wide.diff().abs().sum(axis=1)
        turnover.iloc[0] = weight_wide.iloc[0].abs().sum()
        daily_net = daily_gross - turnover * self.commission

        result = pd.DataFrame(
            {self.date_col: daily_net.index, "portfolio_return": daily_net.values}
        )
        result = result.dropna(subset=["portfolio_return"]).reset_index(drop=True)
        result["cumulative_return"] = (1 + result["portfolio_return"]).cumprod()
        result["equity"] = self.initial_capital * result["cumulative_return"]
        result["cumulative_max"] = result["equity"].cummax()
        result["drawdown"] = (
            result["equity"] - result["cumulative_max"]
        ) / result["cumulative_max"]

        self.results = result
        return result

    def get_summary(self) -> Dict[str, float]:
        """Summary stats for the portfolio equity curve.

        Returns:
            Dict from ``PerformanceMetrics.calculate_all``.

        Raises:
            ValueError: If :meth:`run` hasn't been called yet.
        """
        if self.results is None:
            raise ValueError("Run backtest first by calling run()")
        return PerformanceMetrics.calculate_all(
            self.results["portfolio_return"], self.results["equity"]
        )
