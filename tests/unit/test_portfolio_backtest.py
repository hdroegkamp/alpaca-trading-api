"""Unit tests for the cross-sectional portfolio backtest."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.backtest.portfolio import PortfolioBacktest


class _IdentityScaler:
    """No-op scaler stand-in — the test feature IS the signal, unscaled."""

    def transform(self, X):
        return X


class _PerfectForesightModel:
    """Stub model whose predict_proba reveals the true forward-return sign."""

    def predict_proba(self, X):
        up = (X[:, 0] > 0).astype(float)
        return np.column_stack([1 - up, up])


N_SYMBOLS = 6


def _make_panel(n_dates=20, n_symbols=N_SYMBOLS, seed=0):
    rng = np.random.default_rng(seed)
    # One extra trailing date so the last *scored* date still has a "next"
    # close price to realize a return against.
    dates = pd.date_range("2022-01-01", periods=n_dates + 1, freq="B")
    symbols = [f"SYM{i}" for i in range(n_symbols)]

    rows = []
    for sym_idx, sym in enumerate(symbols):
        # Deterministic per-symbol drift so ranking is stable and testable.
        drift = (sym_idx - n_symbols / 2) * 0.01
        noise = rng.normal(0, 0.0005, len(dates))
        close = 100 * np.cumprod(1 + drift + noise)
        for i, d in enumerate(dates):
            rows.append({"date": d, "symbol": sym, "close": close[i]})

    panel = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    panel["true_forward_return"] = panel.groupby("symbol")["close"].transform(
        lambda s: s.shift(-1) / s - 1
    )
    return panel, dates[:-1]  # usable (scoreable) dates exclude the trailing one


def _fold_over(test_dates):
    return {
        "model": _PerfectForesightModel(),
        "scaler": _IdentityScaler(),
        "test_dates": pd.DatetimeIndex(test_dates),
    }


class TestPortfolioBacktest:
    def test_long_leg_beats_short_leg_under_perfect_foresight(self):
        panel, usable_dates = _make_panel()

        bt = PortfolioBacktest(
            panel=panel,
            fold_models=[_fold_over(usable_dates)],
            feature_columns=["true_forward_return"],
            top_pct=1 / 3,
            bottom_pct=1 / 3,
            commission=0.0,
        )
        results = bt.run()

        # A perfect-foresight score should make almost every rebalance day
        # profitable (long picks that go up, short picks that go down).
        assert (results["portfolio_return"] > 0).mean() > 0.9

    def test_commission_reduces_return(self):
        panel, usable_dates = _make_panel()

        bt_no_comm = PortfolioBacktest(
            panel=panel,
            fold_models=[_fold_over(usable_dates)],
            feature_columns=["true_forward_return"],
            commission=0.0,
        )
        bt_comm = PortfolioBacktest(
            panel=panel,
            fold_models=[_fold_over(usable_dates)],
            feature_columns=["true_forward_return"],
            commission=0.05,
        )
        total_no_comm = bt_no_comm.run()["portfolio_return"].sum()
        total_comm = bt_comm.run()["portfolio_return"].sum()

        assert total_comm < total_no_comm

    def test_one_row_per_rebalance_date_no_gaps(self):
        panel, usable_dates = _make_panel()

        bt = PortfolioBacktest(
            panel=panel,
            fold_models=[_fold_over(usable_dates)],
            feature_columns=["true_forward_return"],
        )
        results = bt.run()

        assert results["date"].is_monotonic_increasing
        assert results["date"].nunique() == len(results)

    def test_book_is_dollar_neutral_per_date(self):
        panel, usable_dates = _make_panel()

        bt = PortfolioBacktest(
            panel=panel,
            fold_models=[_fold_over(usable_dates)],
            feature_columns=["true_forward_return"],
            top_pct=1 / 3,
            bottom_pct=1 / 3,
        )
        bt.run()

        net_by_date = bt.book.groupby("date")["weight"].sum()
        assert np.allclose(net_by_date.values, 0.0, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
