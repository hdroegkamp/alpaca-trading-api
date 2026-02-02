"""Random Forest analyzer with SHAP-based feature importance.

Uses Random Forest classification for price direction prediction with
interpretability through SHAP values.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import pickle


class RandomForestAnalyzer:
    """Random Forest classifier with feature importance analysis."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        random_state: int = 42,
    ):
        """Initialize Random Forest analyzer.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf node
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

        self.scaler = StandardScaler()
        self.feature_columns = None
        self.shap_values = None
        self.shap_explainer = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Dict[str, Any]:
        """Prepare data for training.

        Args:
            df: DataFrame with features and target
            target_col: Target column name
            feature_cols: List of feature columns (if None, use all numeric)
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Dictionary with train/val/test splits
        """
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

        # Remove rows with NaN
        df_clean = df[feature_cols + [target_col]].dropna()

        # Extract features and target
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split into train/val/test
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y  # type: ignore
        )

        # Second split: separate train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp  # type: ignore
        )

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
        """Train Random Forest model.

        Args:
            data_dict: Dictionary from prepare_data()
            verbose: Whether to print training info

        Returns:
            Training metrics dictionary
        """
        # Train model
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

    def hyperparameter_tuning(
        self,
        data_dict: Dict[str, Any],
        param_grid: Optional[Dict[str, List]] = None,
        cv: int = 5,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning with GridSearchCV.

        Args:
            data_dict: Dictionary from prepare_data()
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            verbose: Verbosity level

        Returns:
            Best parameters and scores
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [5, 10, 15],
                "min_samples_leaf": [2, 4, 6],
            }

        # Combine train and val for grid search
        X_train_val = np.vstack([data_dict["X_train"], data_dict["X_val"]])
        y_train_val = np.concatenate([data_dict["y_train"], data_dict["y_val"]])

        # Grid search
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=verbose,
        )

        grid_search.fit(X_train_val, y_train_val)

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
                "importance": self.model.feature_importances_,
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

        # Create explainer
        self.shap_explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values
        self.shap_values = self.shap_explainer.shap_values(X_sample)

        # For binary classification, use positive class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]

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
