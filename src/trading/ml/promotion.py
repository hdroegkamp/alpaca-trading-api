"""Promotion gates: validate the method and the exact artifact before deploying.

``PooledForestModel.walk_forward_validate_panel`` (used for the research
backtest in ``6_Portfolio_ML.py``) validates the *method* — repeatedly
retrained fold models scored on their own held-out dates. It never touches
the literal pickle that ``train_pooled_model.py`` saves to
``data/live/pooled_model_*.pkl`` via ``fit_full()``, which trains on 100% of
history with no held-out check at all.

This module provides two gates, and ``train_pooled_model.py`` requires both:

- :func:`validate_walk_forward_promotion` — does the *method* hold up across
  many expanding-window folds (years of out-of-sample dates)? A single short
  holdout can't answer this: a 60-day Sharpe has a standard error of ~2, so
  a model with no edge clears a zero bar about half the time.
- :func:`validate_holdout_promotion` — is the *literal artifact* about to be
  deployed (same hyperparameters, trained on everything but a recent tail)
  not obviously broken on the most recent data?

Every backtest here scores the **full, unlabeled** panel. The labeled panel
(``add_cross_sectional_labels`` with ``drop_middle=True``) keeps only rows
whose *forward* return landed in the top/bottom band — backtesting on it
would pick each day's book from that day's hindsight winners and losers,
and drop realized returns whenever a held name isn't an extreme mover the
next day. Models train on the labeled panel; they are judged on the panel the
live rebalance actually sees.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from ..backtest.portfolio import PortfolioBacktest
from .panel_split import assert_no_leakage, panel_holdout_split
from .pooled_model import PooledForestModel


def _check_backtest_panel(
    labeled_panel: pd.DataFrame, panel: pd.DataFrame, date_col: str, target_col: str
) -> None:
    """Guard against passing the labeled panel (or a subset of it) as ``panel``."""
    # build_panel output has no target column; a fully-populated one means the
    # middle band was dropped (drop_middle=False keeps those rows as NA).
    if target_col in panel.columns and panel[target_col].notna().all():
        raise ValueError(
            "panel must be the full, unlabeled panel (build_panel output), not "
            "labeled_panel — backtesting on labeled rows leaks forward returns."
        )
    if panel.groupby(date_col).size().median() < labeled_panel.groupby(date_col).size().median():
        raise ValueError(
            "panel has fewer rows per date than labeled_panel — pass the full, "
            "unlabeled panel (build_panel output) for backtesting."
        )


def walk_forward_backtest(
    labeled_panel: pd.DataFrame,
    panel: pd.DataFrame,
    feature_cols: List[str],
    model_params: Optional[Dict[str, Any]] = None,
    n_splits: int = 10,
    embargo_periods: int = 1,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    commission: float = 0.001,
    rebalance_every_n_days: int = 1,
    target_col: str = "target",
    date_col: str = "date",
) -> Dict[str, Any]:
    """Train walk-forward fold models on ``labeled_panel``, backtest on ``panel``.

    Args:
        labeled_panel: Output of ``add_cross_sectional_labels`` — training rows.
        panel: Full, unlabeled panel (``build_panel`` output) — every symbol
            on every date, as the live rebalance sees it. Must include
            ``feature_cols`` and ``close``.
        feature_cols: Feature columns to train on.
        model_params: Constructor kwargs for ``PooledForestModel``.
        n_splits: Number of expanding-window folds.
        embargo_periods: Gap, in unique dates, between each fold's train and
            test range — size to the label's ``forward_periods``.
        top_pct: Long-leg fraction of each date's cross-section.
        bottom_pct: Short-leg fraction of each date's cross-section.
        commission: Flat commission fraction on turnover.
        rebalance_every_n_days: Rebalance cadence within each fold.
        target_col: Binary target column.
        date_col: Column identifying each cross-section.

    Returns:
        Dict with ``folds`` (the ``fold_metrics`` list, each carrying its
        model/scaler/test_dates), ``results`` (stitched daily OOS curve) and
        ``summary`` (its ``PerformanceMetrics.calculate_all`` dict).
    """
    _check_backtest_panel(labeled_panel, panel, date_col, target_col)

    model = PooledForestModel(**(model_params or {}))
    wf = model.walk_forward_validate_panel(
        labeled_panel,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col=date_col,
        n_splits=n_splits,
        embargo_periods=embargo_periods,
    )
    backtest = PortfolioBacktest(
        panel=panel,
        fold_models=wf["fold_metrics"],
        feature_columns=feature_cols,
        date_col=date_col,
        top_pct=top_pct,
        bottom_pct=bottom_pct,
        commission=commission,
        rebalance_every_n_days=rebalance_every_n_days,
    )
    results = backtest.run()
    return {
        "folds": wf["fold_metrics"],
        "results": results,
        "summary": backtest.get_summary(),
    }


def validate_walk_forward_promotion(
    labeled_panel: pd.DataFrame,
    panel: pd.DataFrame,
    feature_cols: List[str],
    model_params: Optional[Dict[str, Any]] = None,
    n_splits: int = 10,
    embargo_periods: int = 1,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    commission: float = 0.001,
    min_sharpe: float = 0.5,
    target_col: str = "target",
    date_col: str = "date",
) -> Dict[str, Any]:
    """Gate on the method: stitched walk-forward OOS Sharpe, net of costs.

    Args:
        labeled_panel: Training rows (``add_cross_sectional_labels`` output).
        panel: Full, unlabeled panel to backtest on.
        feature_cols: Feature columns to train on.
        model_params: Constructor kwargs for ``PooledForestModel``.
        n_splits: Number of expanding-window folds.
        embargo_periods: Gap between each fold's train and test range.
        top_pct: Long-leg fraction for the backtest's book.
        bottom_pct: Short-leg fraction for the backtest's book.
        commission: Flat commission fraction on turnover.
        min_sharpe: Promotion bar — the stitched out-of-sample Sharpe,
            net of ``commission``, must exceed this. Default 0.5: over years
            of OOS dates the Sharpe's standard error is ~0.3, so this asks for
            an edge that is unlikely to be noise, not merely positive.
        target_col: Binary target column.
        date_col: Column identifying each cross-section.

    Returns:
        Dict with ``passed`` (bool), ``summary``, ``results``, ``folds``,
        ``n_days`` (stitched OOS length), and ``min_sharpe``.
    """
    wf = walk_forward_backtest(
        labeled_panel,
        panel,
        feature_cols=feature_cols,
        model_params=model_params,
        n_splits=n_splits,
        embargo_periods=embargo_periods,
        top_pct=top_pct,
        bottom_pct=bottom_pct,
        commission=commission,
        target_col=target_col,
        date_col=date_col,
    )
    return {
        **wf,
        "passed": wf["summary"]["sharpe_ratio"] > min_sharpe,
        "n_days": len(wf["results"]),
        "min_sharpe": min_sharpe,
    }


def validate_holdout_promotion(
    labeled_panel: pd.DataFrame,
    panel: pd.DataFrame,
    feature_cols: List[str],
    model_params: Optional[Dict[str, Any]] = None,
    holdout_days: int = 60,
    embargo_periods: int = 1,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    commission: float = 0.001,
    min_sharpe: float = 0.0,
    target_col: str = "target",
    date_col: str = "date",
) -> Dict[str, Any]:
    """Train on history only, backtest the holdout tail, and decide pass/fail.

    Args:
        labeled_panel: Output of ``add_cross_sectional_labels`` — must
            include ``feature_cols`` and ``target_col``. Used for training.
        panel: Full, unlabeled panel (``build_panel`` output) used for the
            holdout backtest — see the module docstring for why it must not
            be ``labeled_panel``.
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
        commission: Flat commission fraction on turnover.
        min_sharpe: Promotion bar — the holdout Sharpe ratio must exceed
            this to pass. Default 0.0 (must be profitable out-of-sample;
            rejects only an obviously broken retrain — too short a window to
            certify an edge, which is :func:`validate_walk_forward_promotion`'s job).
        target_col: Binary target column.
        date_col: Column identifying each cross-section.

    Returns:
        Dict with ``passed`` (bool), ``summary`` (the backtest's
        ``PerformanceMetrics.calculate_all`` dict), ``holdout_dates``
        (``DatetimeIndex``), ``book`` (the backtest's ``date, symbol, side,
        weight`` frame), and ``min_sharpe``.
    """
    _check_backtest_panel(labeled_panel, panel, date_col, target_col)

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
        panel=panel,
        fold_models=[fold],
        feature_columns=feature_cols,
        date_col=date_col,
        top_pct=top_pct,
        bottom_pct=bottom_pct,
        commission=commission,
    )
    backtest.run()
    summary = backtest.get_summary()

    return {
        "passed": summary["sharpe_ratio"] > min_sharpe,
        "summary": summary,
        "holdout_dates": split["holdout_dates"],
        "book": backtest.book,
        "min_sharpe": min_sharpe,
    }
