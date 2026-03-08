"""Random Forest analyzer with SHAP-based feature importance.

Uses Random Forest classification for price direction prediction with
interpretability through SHAP values.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import pickle


class RandomForestAnalyzer:
    """Tree-based classifier with feature importance analysis.

    Supports both Random Forest and Gradient Boosting backends.
    GradientBoosting uses early stopping on the validation set to
    prevent overfitting — typically outperforms RF on noisy financial data.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 50,
        min_samples_leaf: int = 30,
        max_leaf_nodes: Optional[int] = 60,
        random_state: int = 42,
        model_type: str = "rf",
        learning_rate: float = 0.05,
    ):
        """Initialize tree-based analyzer.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth (lower = less overfitting)
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf node
            max_leaf_nodes: Maximum leaf nodes per tree (caps complexity)
            random_state: Random seed for reproducibility
            model_type: 'rf' for Random Forest, 'gbm' for Gradient Boosting
            learning_rate: Learning rate for GBM (lower = more regularized)
        """
        self._model_type = model_type

        if model_type == "gbm":
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=min(max_depth or 3, 4),  # GBM needs shallow trees
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes=max_leaf_nodes,
                max_features="sqrt",
                learning_rate=learning_rate,
                subsample=0.8,
                random_state=random_state,
                n_iter_no_change=20,
                validation_fraction=0.15,
                tol=1e-4,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes=max_leaf_nodes,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            )

        self.scaler = StandardScaler()
        self.feature_columns = None
        self.shap_values = None
        self.shap_explainer = None
        self._n_estimators = n_estimators
        self._forward_periods = 1  # set by prepare_data for gap logic
        self._rf_model = self.model  # keep reference before possible calibration wrap

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        forward_periods: int = 1,
    ) -> Dict[str, Any]:
        """Prepare data for training with leakage-safe gap.

        Each row's target references a price forward_periods bars ahead.
        A gap of that size is inserted between train/val and val/test
        boundaries so that no training label references prices in the
        validation or test windows.

        Args:
            df: DataFrame with features and target
            target_col: Target column name
            feature_cols: List of feature columns (if None, use all numeric)
            test_size: Proportion for test set
            val_size: Proportion for validation set
            forward_periods: Number of periods the target looks ahead

        Returns:
            Dictionary with train/val/test splits
        """
        self._forward_periods = forward_periods

        # Handle missing target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        # Select features
        if feature_cols is None:
            # Use all numeric columns except target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)

        self.feature_columns = feature_cols

        # Ensure target is never used as a feature
        if target_col in feature_cols:
            feature_cols = [c for c in feature_cols if c != target_col]
            self.feature_columns = feature_cols

        # Remove rows with NaN
        df_clean = df[feature_cols + [target_col]].dropna()

        # Extract features and target
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values

        # Chronological split with gap to prevent target leakage.
        # Never shuffle time-series data.
        n = len(X)
        gap = forward_periods
        test_split = int(n * (1 - test_size))
        val_ratio = val_size / (1 - test_size)
        val_split = int(test_split * (1 - val_ratio))

        # Train ends gap rows before val starts; val ends gap rows before test
        train_end = val_split - gap
        val_end = test_split - gap

        if train_end < 50:
            raise ValueError(
                f"Training set too small after gap ({train_end} rows). "
                "Reduce forward_periods, test_size, or val_size."
            )

        X_train_raw = X[:train_end]
        X_val_raw = X[val_split:val_end]
        X_test_raw = X[test_split:]
        y_train = y[:train_end]
        y_val = y[val_split:val_end]
        y_test = y[test_split:]

        # Fit scaler on training data only; transform val and test with the
        # same parameters so test-set statistics never influence the model.
        X_train = self.scaler.fit_transform(X_train_raw)
        X_val = self.scaler.transform(X_val_raw)
        X_test = self.scaler.transform(X_test_raw)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_columns": feature_cols,
            "target_column": target_col,
        }

    def train(self, data_dict: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """Train tree-based model.

        For GradientBoosting, performs manual early stopping using the
        chronological validation set (instead of a random internal split)
        to maintain time ordering.

        Args:
            data_dict: Dictionary from prepare_data()
            verbose: Whether to print training info

        Returns:
            Training metrics dictionary
        """
        if self._model_type == "gbm":
            # Manual early stopping with proper chronological validation.
            # Disable GBM's internal validation_fraction to avoid random split.
            self.model.set_params(n_iter_no_change=None, validation_fraction=0.1)

            from sklearn.base import clone

            best_model = self.model
            best_val_score = -1
            no_improve = 0
            patience = 20

            # Train iteratively: fit with n trees, check val, stop early
            for n_est in range(50, self._n_estimators + 1, 10):
                candidate = clone(self.model)
                candidate.set_params(n_estimators=n_est, warm_start=False)
                candidate.fit(data_dict["X_train"], data_dict["y_train"])

                val_pred = candidate.predict(data_dict["X_val"])
                val_score = float(
                    f1_score(data_dict["y_val"], val_pred, zero_division=0)
                )

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model = candidate
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience // 10:
                    if verbose:
                        print(
                            f"Early stopping at {n_est} trees (best val F1={best_val_score:.4f})"
                        )
                    break

            self.model = best_model
            self._rf_model = self.model
        else:
            self.model.fit(data_dict["X_train"], data_dict["y_train"])

        # Evaluate on all sets
        metrics = {}

        for split_name in ["train", "val", "test"]:
            X = data_dict[f"X_{split_name}"]
            y = data_dict[f"y_{split_name}"]

            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)[:, 1]

            metrics[split_name] = {
                "accuracy": float(accuracy_score(y, y_pred)),
                "precision": float(precision_score(y, y_pred, zero_division=0)),
                "recall": float(recall_score(y, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            }

            if verbose and split_name == "test":
                print(f"\nTest Set Performance:")
                print(f"Accuracy:  {metrics[split_name]['accuracy']:.4f}")
                print(f"Precision: {metrics[split_name]['precision']:.4f}")
                print(f"Recall:    {metrics[split_name]['recall']:.4f}")
                print(f"F1 Score:  {metrics[split_name]['f1_score']:.4f}")

                print(f"\nConfusion Matrix:")
                print(confusion_matrix(y, y_pred))

        return metrics

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        n_splits: int = 5,
        forward_periods: int = 1,
    ) -> Dict[str, Any]:
        """Walk-forward (expanding window) validation for time series.

        Trains on progressively larger windows and tests on the next fold,
        with a gap of forward_periods to prevent target leakage.

        Args:
            df: Full DataFrame with features and target.
            feature_cols: Feature column names.
            target_col: Target column name.
            n_splits: Number of walk-forward folds.
            forward_periods: Gap between train and test in each fold.

        Returns:
            Dict with per-fold and aggregate metrics.
        """
        from sklearn.base import clone

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

            fold_model = clone(self._rf_model)
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

        # Aggregate
        avg = {
            metric: float(np.mean([f[metric] for f in fold_metrics]))
            for metric in ["accuracy", "precision", "recall", "f1_score"]
        }
        std = {
            f"{metric}_std": float(np.std([f[metric] for f in fold_metrics]))
            for metric in ["accuracy", "precision", "recall", "f1_score"]
        }

        return {
            "fold_metrics": fold_metrics,
            "mean": avg,
            "std": std,
        }

    def calibrate(self, data_dict: Dict[str, Any]) -> None:
        """Calibrate predicted probabilities using validation data.

        Wraps the trained model with isotonic regression so that
        predict_proba() outputs well-calibrated probabilities.
        Call after train().
        """
        self.model = CalibratedClassifierCV(self.model, cv="prefit", method="isotonic")
        self.model.fit(data_dict["X_val"], data_dict["y_val"])

    def hyperparameter_tuning(
        self,
        data_dict: Dict[str, Any],
        param_grid: Optional[Dict[str, List]] = None,
        n_splits: int = 5,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning with TimeSeriesSplit.

        Uses time-series-aware cross-validation to prevent future data
        from leaking into training folds.

        Args:
            data_dict: Dictionary from prepare_data()
            param_grid: Parameter grid for search
            n_splits: Number of time-series CV folds
            verbose: Verbosity level

        Returns:
            Best parameters and scores
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 5, 7],
                "min_samples_split": [30, 50, 80],
                "min_samples_leaf": [20, 30, 50],
                "max_leaf_nodes": [40, 60, 80, None],
            }

        # Use only training data for grid search with TimeSeriesSplit
        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]

        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            gap=self._forward_periods,
        )

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=tscv,
            scoring="f1",
            n_jobs=-1,
            verbose=verbose,
        )

        grid_search.fit(X_train, y_train)

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        return {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "cv_results": grid_search.cv_results_,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature array

        Returns:
            Predicted labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature array

        Returns:
            Predicted probabilities
        """
        return self.model.predict_proba(X)

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict from DataFrame.

        Args:
            df: DataFrame with feature columns

        Returns:
            DataFrame with predictions and probabilities
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained. Call train() first.")

        # Extract and scale features
        X = df[self.feature_columns].fillna(0).values
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.predict(X_scaled)
        probabilities = self.predict_proba(X_scaled)

        # Create result DataFrame
        result = pd.DataFrame(
            {
                "prediction": predictions,
                "probability_down": probabilities[:, 0],
                "probability_up": probabilities[:, 1],
            },
            index=df.index,
        )

        return result

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from Random Forest.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained. Call train() first.")

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": self._rf_model.feature_importances_,  # type: ignore[union-attr]
            }
        ).sort_values("importance", ascending=False)

        return importance_df.head(top_n)

    def calculate_shap_values(
        self, X: np.ndarray, max_samples: int = 1000
    ) -> np.ndarray:
        """Calculate SHAP values for interpretability.

        Args:
            X: Feature array
            max_samples: Maximum samples to use (for performance)

        Returns:
            SHAP values array
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is required for feature importance analysis. "
                "Install with: pip install shap"
            )

        # Subsample if needed
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Create explainer (use underlying RF, not calibration wrapper)
        self.shap_explainer = shap.TreeExplainer(self._rf_model)

        # Calculate SHAP values
        self.shap_values = self.shap_explainer.shap_values(X_sample)

        # For binary classification, use positive class.
        # Older SHAP (<0.46) returns a list of [neg_class, pos_class] arrays.
        # Newer SHAP (>=0.46) returns a single 3D array (n_samples, n_features, n_classes).
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        elif isinstance(self.shap_values, np.ndarray) and self.shap_values.ndim == 3:
            self.shap_values = self.shap_values[:, :, 1]

        return self.shap_values

    def get_shap_summary(self, X: np.ndarray, top_n: int = 20) -> pd.DataFrame:
        """Get SHAP value summary.

        Args:
            X: Feature array
            top_n: Number of top features

        Returns:
            DataFrame with SHAP importance
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)

        # Calculate mean absolute SHAP values
        shap_importance = np.abs(self.shap_values).mean(axis=0)  # type: ignore

        importance_df = pd.DataFrame(
            {"feature": self.feature_columns, "shap_importance": shap_importance}
        ).sort_values("shap_importance", ascending=False)

        return importance_df.head(top_n)

    def plot_shap_summary(self, X: np.ndarray, plot_type: str = "bar"):
        """Plot SHAP summary (requires matplotlib).

        Args:
            X: Feature array
            plot_type: 'bar' or 'beeswarm'
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib required for plotting")

        if self.shap_values is None:
            self.calculate_shap_values(X)

        # Create feature names
        feature_names = self.feature_columns

        if plot_type == "bar":
            shap.summary_plot(
                self.shap_values,
                X,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
            )
        else:
            shap.summary_plot(
                self.shap_values, X, feature_names=feature_names, show=False
            )

        return plt.gcf()

    def evaluate_detailed(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Get detailed evaluation metrics.

        Args:
            X: Test features
            y: Test labels

        Returns:
            Dictionary with detailed metrics
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]

        # Confusion matrix
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
        """Save model and scaler.

        Args:
            filepath: Path to save model (without extension)
        """
        # Save model
        with open(f"{filepath}_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # Save scaler and metadata
        metadata = {
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
        }

        with open(f"{filepath}_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load(self, filepath: str):
        """Load model and scaler.

        Args:
            filepath: Path to saved model (without extension)
        """
        # Load model
        with open(f"{filepath}_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load scaler and metadata
        with open(f"{filepath}_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.scaler = metadata["scaler"]
        self.feature_columns = metadata["feature_columns"]
