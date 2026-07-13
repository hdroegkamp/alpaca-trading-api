"""Pooled panel-native Random Forest: one model trained across the whole universe.

Date-native counterpart to ``RandomForestAnalyzer`` — every split is by
unique date (via ``src.trading.ml.panel_split``) rather than row position, so
a day's whole cross-section always lands in the same fold. See
``src.trading.ml.panel`` for how the training panel is assembled and
``src.trading.ml.cross_sectional_labels`` for how ``target_col`` is produced.
"""

import pickle
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from .panel_split import panel_train_val_test_split, panel_walk_forward_splits


def _split_metrics(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
    }


class PooledForestModel:
    """Regularised Random Forest trained on a pooled cross-sectional panel."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 50,
        min_samples_leaf: int = 30,
        max_features: Union[Literal["sqrt", "log2"], float] = "sqrt",
        random_state: int = 42,
    ):
        """Initialize the pooled Random Forest.

        Same conservative defaults as ``RandomForestAnalyzer`` (shallow
        trees, large leaves) to limit overfitting on noisy financial data —
        see that class for rationale. This class differs only in how data is
        split (by date across the whole panel) and in what a walk-forward
        fold returns (the fitted model itself, for the portfolio backtest).

        Args:
            n_estimators: Number of trees.
            max_depth: Maximum tree depth (lower = less overfitting).
            min_samples_split: Minimum samples to split an internal node.
            min_samples_leaf: Minimum samples in a leaf.
            max_features: Features considered per split.
            random_state: Random seed for reproducibility.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.feature_columns: Optional[List[str]] = None

    def prepare_panel_data(
        self,
        panel: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        date_col: str = "date",
        test_size: float = 0.2,
        val_size: float = 0.1,
        embargo_periods: int = 1,
    ) -> Dict[str, Any]:
        """Date-based train/val/test split, scaler fit on train only.

        Args:
            panel: Labeled panel (output of ``add_cross_sectional_labels``).
            feature_cols: Feature columns to use.
            target_col: Binary target column.
            date_col: Column identifying each cross-section.
            test_size: Fraction of unique dates reserved for testing.
            val_size: Fraction of unique dates reserved for validation.
            embargo_periods: Gap, in unique dates, at each split boundary.

        Returns:
            Dict with X_/y_ train, val, test arrays plus column metadata and
            the underlying date ranges (``date_splits``).

        Raises:
            ValueError: If the resulting training set is too small.
        """
        self.feature_columns = feature_cols
        clean = panel.dropna(subset=feature_cols + [target_col])

        date_splits = panel_train_val_test_split(
            clean,
            date_col=date_col,
            test_size=test_size,
            val_size=val_size,
            embargo_periods=embargo_periods,
        )

        def subset(dates):
            mask = clean[date_col].isin(dates)
            return (
                clean.loc[mask, feature_cols].values,
                clean.loc[mask, target_col].values,
            )

        X_train, y_train = subset(date_splits["train_dates"])
        X_val, y_val = subset(date_splits["val_dates"])
        X_test, y_test = subset(date_splits["test_dates"])

        if len(X_train) < 50:
            raise ValueError(
                f"Training set too small after split ({len(X_train)} rows). "
                "Reduce test_size, val_size, embargo_periods, or widen the universe."
            )

        return {
            "X_train": self.scaler.fit_transform(X_train),
            "y_train": y_train,
            "X_val": self.scaler.transform(X_val),
            "y_val": y_val,
            "X_test": self.scaler.transform(X_test),
            "y_test": y_test,
            "feature_columns": feature_cols,
            "target_column": target_col,
            "date_splits": date_splits,
        }

    def train(self, data_dict: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """Fit the model and report metrics on all splits.

        Same contract as ``RandomForestAnalyzer.train``.

        Args:
            data_dict: Output of :meth:`prepare_panel_data`.
            verbose: Print the test-set summary.

        Returns:
            Dict of accuracy/precision/recall/f1 for train, val, and test.
        """
        self.model.fit(data_dict["X_train"], data_dict["y_train"])

        metrics = {
            split: _split_metrics(
                self.model, data_dict[f"X_{split}"], data_dict[f"y_{split}"]
            )
            for split in ["train", "val", "test"]
        }

        if verbose:
            t = metrics["test"]
            print("\nTest Set Performance:")
            print(f"Accuracy:  {t['accuracy']:.4f}")
            print(f"Precision: {t['precision']:.4f}")
            print(f"Recall:    {t['recall']:.4f}")
            print(f"F1 Score:  {t['f1_score']:.4f}")

        return metrics

    def walk_forward_validate_panel(
        self,
        panel: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        date_col: str = "date",
        n_splits: int = 5,
        embargo_periods: int = 1,
    ) -> Dict[str, Any]:
        """Expanding-window walk-forward validation over the panel's unique dates.

        Unlike ``RandomForestAnalyzer.walk_forward_validate``, each fold's
        result also carries the fitted model, scaler, and test date range —
        ``PortfolioBacktest`` needs a fold's "trained only on the past" model
        to score that fold's own out-of-sample dates, never a date at or
        after its own training window.

        Args:
            panel: Labeled panel.
            feature_cols: Feature columns to use.
            target_col: Binary target column.
            date_col: Column identifying each cross-section.
            n_splits: Number of walk-forward folds.
            embargo_periods: Gap, in unique dates, between train and test in
                each fold.

        Returns:
            Dict with ``fold_metrics`` (list of per-fold dicts including
            ``model``, ``scaler``, ``test_dates``, and the usual accuracy/
            precision/recall/f1) plus aggregate ``mean``/``std``.
        """
        self.feature_columns = feature_cols
        clean = panel.dropna(subset=feature_cols + [target_col])

        folds = panel_walk_forward_splits(
            clean, date_col=date_col, n_splits=n_splits, embargo_periods=embargo_periods
        )

        fold_metrics = []
        for fold_idx, (train_dates, test_dates) in enumerate(folds):
            train_mask = clean[date_col].isin(train_dates)
            test_mask = clean[date_col].isin(test_dates)

            X_train_raw = clean.loc[train_mask, feature_cols].values
            X_test_raw = clean.loc[test_mask, feature_cols].values
            y_train = clean.loc[train_mask, target_col].values
            y_test = clean.loc[test_mask, target_col].values

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            fold_model = clone(self.model)
            fold_model.fit(X_train, y_train)

            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "train_size": int(train_mask.sum()),
                    "test_size": int(test_mask.sum()),
                    "test_dates": test_dates,
                    "model": fold_model,
                    "scaler": scaler,
                    **_split_metrics(fold_model, X_test, y_test),
                }
            )

        metric_names = ["accuracy", "precision", "recall", "f1_score"]
        mean = {m: float(np.mean([f[m] for f in fold_metrics])) for m in metric_names}
        std = {
            f"{m}_std": float(np.std([f[m] for f in fold_metrics]))
            for m in metric_names
        }

        return {
            "fold_metrics": fold_metrics,
            "mean": mean,
            "std": std,
            "feature_columns": feature_cols,
        }

    def fit_full(
        self, panel: pd.DataFrame, feature_cols: List[str], target_col: str = "target"
    ) -> None:
        """Fit on the full available (labeled) history.

        Produces a "production" model for scoring the most recent
        cross-section with :meth:`predict_scores` — not used for backtesting
        (which relies on the walk-forward fold models to avoid lookahead).

        Args:
            panel: Labeled panel.
            feature_cols: Feature columns to use.
            target_col: Binary target column.
        """
        self.feature_columns = feature_cols
        clean = panel.dropna(subset=feature_cols + [target_col])
        X = self.scaler.fit_transform(clean[feature_cols].values)
        y = clean[target_col].values
        self.model.fit(X, y)

    def predict_scores(
        self,
        panel_slice: pd.DataFrame,
        date_col: str = "date",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """Score a slice of the panel with ``predict_proba``.

        Args:
            panel_slice: Rows containing the trained feature columns (and
                ideally ``date_col``/``symbol_col`` for downstream ranking).
            date_col: Column identifying each cross-section, carried through
                if present.
            symbol_col: Symbol column, carried through if present.

        Returns:
            DataFrame indexed like ``panel_slice`` with ``date``/``symbol``
            (if present) and ``probability_up`` — usable directly as the
            cross-sectional ranking signal.

        Raises:
            ValueError: If the model hasn't been trained yet.
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained. Call train()/fit_full() first.")

        X = panel_slice[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        proba_up = self.model.predict_proba(X_scaled)[:, 1]

        cols: Dict[str, Any] = {}
        if date_col in panel_slice.columns:
            cols[date_col] = panel_slice[date_col].values
        if symbol_col in panel_slice.columns:
            cols[symbol_col] = panel_slice[symbol_col].values
        cols["probability_up"] = proba_up

        return pd.DataFrame(cols, index=panel_slice.index)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return native Random Forest feature importances, most important first.

        Args:
            top_n: Number of top features to return.

        Returns:
            DataFrame with ``feature`` and ``importance`` columns.

        Raises:
            ValueError: If the model hasn't been trained yet.
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained. Call train() first.")

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def save(self, filepath: str) -> None:
        """Persist the model, scaler, and feature columns.

        Args:
            filepath: Path prefix (without extension).
        """
        with open(f"{filepath}_model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        with open(f"{filepath}_metadata.pkl", "wb") as f:
            pickle.dump(
                {"scaler": self.scaler, "feature_columns": self.feature_columns}, f
            )

    def load(self, filepath: str) -> None:
        """Load a model, scaler, and feature columns saved by :meth:`save`.

        Args:
            filepath: Path prefix (without extension).
        """
        with open(f"{filepath}_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(f"{filepath}_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        self.scaler = metadata["scaler"]
        self.feature_columns = metadata["feature_columns"]
