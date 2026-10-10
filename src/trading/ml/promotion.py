"""Promotion gate: validate the exact model artifact about to be deployed.

``PooledForestModel.walk_forward_validate_panel`` (used for the research
backtest in ``6_Portfolio_ML.py``) validates the *method* — repeatedly
retrained fold models scored on their own held-out dates. It never touches
the literal pickle that ``train_pooled_model.py`` saves to
``data/live/pooled_model_*.pkl`` via ``fit_full()``, which trains on 100% of
history with no held-out check at all.

This module closes that gap: train a candidate on history only, holding out
a recent tail of dates, score that holdout with ``PortfolioBacktest`` (the
same accounting code the method was already validated with), and gate
promotion on a simple out-of-sample profitability bar. Only on a pass should
the caller go on to retrain on 100% of history and overwrite the live model.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from ..backtest.portfolio import PortfolioBacktest
from .panel_split import assert_no_leakage, panel_holdout_split
from .pooled_model import PooledForestModel


def validate_holdout_promotion(
    labeled_panel: pd.DataFrame,
    feature_cols: List[str],
    model_params: Optional[Dict[str, Any]] = None,
    holdout_days: int = 60,
    embargo_periods: int = 1,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    min_sharpe: float = 0.0,
    target_col: str = "target",
    date_col: str = "date",
) -> Dict[str, Any]:
    """Train on history only, backtest the holdout tail, and decide pass/fail.

    Args:
        labeled_panel: Output of ``add_cross_sectional_labels`` — must
            include ``feature_cols``, ``target_col``, and ``close``.
        feature_cols: Feature columns to train on (same set used for the
            production model).
        model_params: Constructor kwargs for ``PooledForestModel`` — pass
            the same hyperparameters the production model will use, so the
            holdout result reflects the artifact that will actually ship.
        holdout_days: Number of most-recent unique dates held out from
            training and used as the out-of-sample backtest window.
        embargo_periods: Gap, in unique dates, between training and the
            holdout window (see ``panel_holdout_split``).
        top_pct: Long-leg fraction for the holdout backtest's book.
        bottom_pct: Short-leg fraction for the holdout backtest's book.
        min_sharpe: Promotion bar — the holdout Sharpe ratio must exceed
            this to pass. Default 0.0 (must be profitable out-of-sample;
            rejects only an obviously broken retrain).
        target_col: Binary target column.
        date_col: Column identifying each cross-section.

    Returns:
        Dict with ``passed`` (bool), ``summary`` (the backtest's
        ``PerformanceMetrics.calculate_all`` dict), ``holdout_dates``
        (``DatetimeIndex``), and ``min_sharpe``.
    """
    split = panel_holdout_split(
        labeled_panel,
        date_col=date_col,
        holdout_days=holdout_days,
        embargo_periods=embargo_periods,
    )
    all_dates = labeled_panel[date_col].unique()
    assert_no_leakage(all_dates, split["train_dates"], split["holdout_dates"], embargo_periods)

    train_panel = labeled_panel[labeled_panel[date_col].isin(split["train_dates"])]

    holdout_model = PooledForestModel(**(model_params or {}))
    holdout_model.fit_full(train_panel, feature_cols=feature_cols, target_col=target_col)

    fold = {
        "model": holdout_model.model,
        "scaler": holdout_model.scaler,
        "test_dates": split["holdout_dates"],
    }
    backtest = PortfolioBacktest(
        panel=labeled_panel,
        fold_models=[fold],
        feature_columns=feature_cols,
        date_col=date_col,
        top_pct=top_pct,
        bottom_pct=bottom_pct,
    )
    backtest.run()
    summary = backtest.get_summary()

    return {
        "passed": summary["sharpe_ratio"] > min_sharpe,
        "summary": summary,
        "holdout_dates": split["holdout_dates"],
        "min_sharpe": min_sharpe,
    }
