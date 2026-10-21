"""Unit tests for the holdout model-promotion gate (src.trading.ml.promotion)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.ml.cross_sectional_labels import add_cross_sectional_labels
from src.trading.ml.panel_split import panel_holdout_split
from src.trading.ml.promotion import (
    validate_holdout_promotion,
    validate_walk_forward_promotion,
)

N_SYMBOLS = 8
N_DATES = 150
HOLDOUT_DAYS = 30

# Keep the model tiny so tests run fast — the gate logic, not model quality,
# is what's under test.
_SMALL_MODEL_PARAMS = {
    "n_estimators": 20,
    "max_depth": 3,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
}


def _make_panels(signal_strength, seed=0):
    """Build a (full panel, labeled panel) pair where one feature predicts next-day return.

    ``signal_strength`` scales how strongly ``signal`` feature correlates
    with ``forward_return`` — strong and positive makes the model's ranking
    genuinely profitable out-of-sample (should pass the gate); zero/negative
    makes it noise or anti-correlated (should fail).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=N_DATES + 1, freq="B")
    symbols = [f"SYM{i}" for i in range(N_SYMBOLS)]

    rows = []
    for d_idx, d in enumerate(dates):
        for sym in symbols:
            rows.append({"date": d, "symbol": sym, "signal": rng.normal()})
    panel = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)

    # Close price walks so that each day's forward return is driven by that
    # day's "signal" feature (plus noise), scaled by signal_strength.
    noise = rng.normal(0, 0.001, len(panel))
    panel["forward_return"] = signal_strength * panel["signal"] * 0.01 + noise

    close = []
    for sym in symbols:
        sym_mask = panel["symbol"] == sym
        sym_returns = panel.loc[sym_mask, "forward_return"].shift(1).fillna(0.0).values
        close.append(pd.Series(100 * np.cumprod(1 + sym_returns), index=panel.loc[sym_mask].index))
    panel["close"] = pd.concat(close).sort_index()

    labeled = add_cross_sectional_labels(panel, top_quantile=0.25, bottom_quantile=0.25)
    return panel, labeled


class TestPanelHoldoutSplit:
    def test_train_precedes_holdout_with_embargo_gap(self):
        _, panel = _make_panels(signal_strength=1.0)
        split = panel_holdout_split(panel, holdout_days=HOLDOUT_DAYS, embargo_periods=1)

        assert split["train_dates"].max() < split["holdout_dates"].min()
        assert len(split["holdout_dates"]) == HOLDOUT_DAYS

    def test_raises_when_too_few_dates(self):
        _, panel = _make_panels(signal_strength=1.0)
        with pytest.raises(ValueError):
            panel_holdout_split(panel, holdout_days=10_000, embargo_periods=1)


class TestValidateHoldoutPromotion:
    def test_strong_signal_passes_the_gate(self):
        panel, labeled = _make_panels(signal_strength=5.0, seed=1)

        result = validate_holdout_promotion(
            labeled,
            panel,
            feature_cols=["signal"],
            model_params=_SMALL_MODEL_PARAMS,
            holdout_days=HOLDOUT_DAYS,
            top_pct=0.25,
            bottom_pct=0.25,
            min_sharpe=0.0,
        )

        assert result["passed"] is True
        assert result["summary"]["sharpe_ratio"] > 0.0
        assert len(result["holdout_dates"]) == HOLDOUT_DAYS

    def test_pure_noise_does_not_reliably_pass(self):
        # No real relationship between "signal" and forward_return.
        panel, labeled = _make_panels(signal_strength=0.0, seed=2)

        result = validate_holdout_promotion(
            labeled,
            panel,
            feature_cols=["signal"],
            model_params=_SMALL_MODEL_PARAMS,
            holdout_days=HOLDOUT_DAYS,
            top_pct=0.25,
            bottom_pct=0.25,
            min_sharpe=0.0,
        )

        # With a high enough bar, pure noise should fail; this is the gate's
        # core promise — it must be possible to fail, not just always pass.
        strict = validate_holdout_promotion(
            labeled,
            panel,
            feature_cols=["signal"],
            model_params=_SMALL_MODEL_PARAMS,
            holdout_days=HOLDOUT_DAYS,
            top_pct=0.25,
            bottom_pct=0.25,
            min_sharpe=100.0,
        )
        assert strict["passed"] is False
        assert result["summary"]["sharpe_ratio"] == strict["summary"]["sharpe_ratio"]

    def test_inverted_signal_fails_a_zero_bar(self):
        # Train on history where "signal" predicts the OPPOSITE of what the
        # holdout window will show — an obviously broken retrain.
        panel, labeled = _make_panels(signal_strength=5.0, seed=3)
        split = panel_holdout_split(labeled, holdout_days=HOLDOUT_DAYS, embargo_periods=1)

        inverted = labeled.copy()
        train_mask = inverted["date"].isin(split["train_dates"])
        inverted.loc[train_mask, "signal"] = -inverted.loc[train_mask, "signal"]

        result = validate_holdout_promotion(
            inverted,
            panel,
            feature_cols=["signal"],
            model_params=_SMALL_MODEL_PARAMS,
            holdout_days=HOLDOUT_DAYS,
            top_pct=0.25,
            bottom_pct=0.25,
            min_sharpe=0.0,
        )

        assert result["passed"] is False
        assert result["summary"]["sharpe_ratio"] < 0.0

    def test_holdout_book_is_drawn_from_the_full_cross_section(self):
        # Regression: the gate used to backtest on the labeled panel, which
        # (drop_middle=True) keeps only each date's hindsight top/bottom
        # forward-return names — 4 of 8 here — so the book was picked from
        # future winners/losers. On the full panel every date has all 8
        # names, giving round(8 * 0.25) = 2 per leg.
        panel, labeled = _make_panels(signal_strength=0.0, seed=4)

        result = validate_holdout_promotion(
            labeled,
            panel,
            feature_cols=["signal"],
            model_params=_SMALL_MODEL_PARAMS,
            holdout_days=HOLDOUT_DAYS,
            top_pct=0.25,
            bottom_pct=0.25,
        )

        names_per_date = result["book"].groupby("date").size()
        assert (names_per_date == 4).all()

    def test_rejects_labeled_panel_as_backtest_panel(self):
        _, labeled = _make_panels(signal_strength=1.0)
        with pytest.raises(ValueError, match="unlabeled"):
            validate_holdout_promotion(
                labeled, labeled, feature_cols=["signal"], model_params=_SMALL_MODEL_PARAMS
            )
        with pytest.raises(ValueError, match="unlabeled"):
            validate_holdout_promotion(
                labeled, labeled.copy(), feature_cols=["signal"], model_params=_SMALL_MODEL_PARAMS
            )


class TestValidateWalkForwardPromotion:
    def test_strong_signal_passes(self):
        panel, labeled = _make_panels(signal_strength=5.0, seed=5)

        result = validate_walk_forward_promotion(
            labeled,
            panel,
            feature_cols=["signal"],
            model_params=_SMALL_MODEL_PARAMS,
            n_splits=3,
            top_pct=0.25,
            bottom_pct=0.25,
            min_sharpe=0.5,
        )

        assert result["passed"] is True
        assert result["summary"]["sharpe_ratio"] > 0.5
        assert len(result["folds"]) == 3
        assert result["n_days"] > 0

    def test_pure_noise_fails_net_of_costs(self):
        # No signal: daily long/short turnover just pays commission.
        panel, labeled = _make_panels(signal_strength=0.0, seed=6)

        result = validate_walk_forward_promotion(
            labeled,
            panel,
            feature_cols=["signal"],
            model_params=_SMALL_MODEL_PARAMS,
            n_splits=3,
            top_pct=0.25,
            bottom_pct=0.25,
            commission=0.001,
            min_sharpe=0.5,
        )

        assert result["passed"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
