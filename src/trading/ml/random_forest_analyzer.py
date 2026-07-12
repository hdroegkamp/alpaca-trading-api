"""Small, leakage-safe Random Forest classifier for price direction.

A deliberately simple baseline: one regularised Random Forest on a handful of
stationary features, evaluated honestly with a chronological train/val/test
split and walk-forward validation. No ensembling, calibration, or SHAP — the
emphasis is on a signal that is easy to trust, test, and explain.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Literal, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import pickle


class RandomForestAnalyzer:
    """Regularised Random Forest classifier for up/down prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 50,
        min_samples_leaf: int = 30,
        max_features: Union[Literal["sqrt", "log2"], float] = "sqrt",
        random_state: int = 42,
    ):
        """Initialize the Random Forest.

        Defaults are intentionally conservative (shallow trees, large leaves)
        to limit overfitting on noisy financial data.

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
        self._forward_periods = 1

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        forward_periods: int = 1,
    ) -> Dict[str, Any]:
        """Build leakage-safe chronological train/val/test splits.

        Each row's target references a price ``forward_periods`` bars ahead, so
        a gap of that size is inserted at the train/val and val/test boundaries.
        The scaler is fit on the training slice only.

        Args:
            df: DataFrame with features and target.
            target_col: Target column name.
            feature_cols: Feature columns (defaults to all numeric except target).
            test_size: Proportion for the test set.
            val_size: Proportion for the validation set.
            forward_periods: Bars the target looks ahead (gap size).

        Returns:
            Dict with X_/y_ train, val, test arrays plus column metadata.
        """
        self._forward_periods = forward_periods

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c != target_col]
        self.feature_columns = feature_cols

        df_clean = df[feature_cols + [target_col]].dropna()
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values

        # Chronological split with a gap. Never shuffle time-series data.
        n = len(X)
        gap = forward_periods
        test_split = int(n * (1 - test_size))
        val_ratio = val_size / (1 - test_size)
        val_split = int(test_split * (1 - val_ratio))

        train_end = val_split - gap
        val_end = test_split - gap

        if train_end < 50:
            raise ValueError(
                f"Training set too small after gap ({train_end} rows). "
                "Reduce forward_periods, test_size, or val_size."
            )

        X_train = self.scaler.fit_transform(X[:train_end])
        X_val = self.scaler.transform(X[val_split:val_end])
        X_test = self.scaler.transform(X[test_split:])

        return {
            "X_train": X_train,
            "y_train": y[:train_end],
            "X_val": X_val,
            "y_val": y[val_split:val_end],
            "X_test": X_test,
            "y_test": y[test_split:],
            "feature_columns": feature_cols,
            "target_column": target_col,
        }

    def train(self, data_dict: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """Fit the model and report metrics on all splits.

        Args:
            data_dict: Output of :meth:`prepare_data`.
            verbose: Print the test-set summary.

        Returns:
            Dict of accuracy/precision/recall/f1 for train, val, and test.
        """
        self.model.fit(data_dict["X_train"], data_dict["y_train"])

        metrics: Dict[str, Any] = {}
        for split in ["train", "val", "test"]:
            X, y = data_dict[f"X_{split}"], data_dict[f"y_{split}"]
            y_pred = self.model.predict(X)
            metrics[split] = {
                "accuracy": float(accuracy_score(y, y_pred)),
                "precision": float(precision_score(y, y_pred, zero_division=0)),
                "recall": float(recall_score(y, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            }

        if verbose:
            t = metrics["test"]
            print("\nTest Set Performance:")
            print(f"Accuracy:  {t['accuracy']:.4f}")
            print(f"Precision: {t['precision']:.4f}")
            print(f"Recall:    {t['recall']:.4f}")
            print(f"F1 Score:  {t['f1_score']:.4f}")
            print("\nConfusion Matrix:")
            print(
                confusion_matrix(
                    data_dict["y_test"], self.model.predict(data_dict["X_test"])
                )
            )

        return metrics

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        n_splits: int = 5,
        forward_periods: int = 1,
    ) -> Dict[str, Any]:
        """Expanding-window walk-forward validation.

        Trains on progressively larger windows and tests on the next fold, with
        a gap of ``forward_periods`` to prevent target leakage. This is the most
        honest estimate of out-of-sample performance for a time series.

        Args:
            df: Full DataFrame with features and target.
            feature_cols: Feature column names.
            target_col: Target column name.
            n_splits: Number of walk-forward folds.
            forward_periods: Gap between train and test in each fold.

        Returns:
            Dict with per-fold metrics plus aggregate mean and std.
        """
        df_clean = df[feature_cols + [target_col]].dropna()
        X_all = np.asarray(df_clean[feature_cols])
        y_all = np.asarray(df_clean[target_col])

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=forward_periods)
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_all[train_idx])
            X_test = scaler.transform(X_all[test_idx])
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            fold_model = clone(self.model)
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)

            fold_metrics.append(
                {
                    "fold": fold,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(
                        precision_score(y_test, y_pred, zero_division=0)
                    ),
                    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                }
            )

        metric_names = ["accuracy", "precision", "recall", "f1_score"]
        mean = {m: float(np.mean([f[m] for f in fold_metrics])) for m in metric_names}
        std = {
            f"{m}_std": float(np.std([f[m] for f in fold_metrics]))
            for m in metric_names
        }

        return {"fold_metrics": fold_metrics, "mean": mean, "std": std}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for a scaled feature array."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a scaled feature array."""
        return self.model.predict_proba(X)

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict from a DataFrame containing the model's feature columns.

        Args:
            df: DataFrame with the trained feature columns.

        Returns:
            DataFrame indexed like ``df`` with columns: prediction,
            probability_down, probability_up.
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained. Call train() first.")

        X = df[self.feature_columns].fillna(0).values
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return pd.DataFrame(
            {
                "prediction": predictions,
                "probability_down": probabilities[:, 0],
                "probability_up": probabilities[:, 1],
            },
            index=df.index,
        )

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return native Random Forest feature importances, most important first.

        Args:
            top_n: Number of top features to return.

        Returns:
            DataFrame with ``feature`` and ``importance`` columns.
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

    def evaluate_detailed(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Detailed evaluation on a scaled feature array.

        Args:
            X: Scaled test features.
            y: True labels.

        Returns:
            Dict with headline metrics, confusion-matrix cells, and a report.
        """
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "classification_report": classification_report(y, y_pred),
        }

    def save(self, filepath: str):
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

    def load(self, filepath: str):
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
