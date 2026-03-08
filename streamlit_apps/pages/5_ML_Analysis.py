"""ML Analysis page for model training and evaluation.

Train LSTM and Random Forest models, analyze feature importance,
and compare ML strategies against traditional approaches.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.data.storage import DataStore
from src.trading.ml.feature_engineering import FeatureEngineering
from src.trading.ml.lstm_predictor import LSTMPredictor
from src.trading.ml.random_forest_analyzer import RandomForestAnalyzer
from src.trading.ml.ml_strategy import (
    MLPredictionStrategy,
    LSTMStrategy,
    RandomForestStrategy,
)
from src.trading.backtest.vectorized import VectorizedBacktest
from src.trading.strategy.examples.moving_average import MovingAverageCrossover

st.title("Machine Learning Analysis")
st.markdown("Train ML models, analyze predictions, and compare strategy performance")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # Data directory
    data_dir = st.text_input(
        "Data Directory", value="Z:\\market_data", help="Path to market data storage"
    )

    # Model save directory
    model_dir = st.text_input(
        "Model Directory", value="models", help="Directory to save/load trained models"
    )

    Path(model_dir).mkdir(exist_ok=True)

# Initialize data store
try:
    data_store = DataStore(data_dir=data_dir)
    inventory = data_store.get_inventory()
    if len(inventory) > 0:
        available_symbols = inventory["symbol"].unique().tolist()
    else:
        available_symbols = []
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Feature Engineering",
        "Model Training",
        "Predictions & Analysis",
        "Strategy Comparison",
    ]
)

# Tab 1: Feature Engineering
with tab1:
    st.header("Feature Engineering")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Data Selection")

        # Multi-symbol selector
        selected_symbols = st.multiselect(
            "Symbols",
            options=available_symbols,
            default=[],
            help=f"{len(available_symbols)} symbols available. "
            "Select one or more to load and generate features.",
            key="fe_symbol_select",
        )

        timeframe = st.selectbox(
            "Timeframe", options=["1Day", "1Hour", "15Min"], index=0
        )

        if st.button("Load & Generate Features", type="primary"):
            if not selected_symbols:
                st.error("Please select at least one symbol.")
            else:
                with st.spinner("Loading data and generating features..."):
                    fe = FeatureEngineering()
                    loaded: dict = {}
                    progress = st.progress(0)

                    for idx, sym in enumerate(selected_symbols):
                        sym_df = data_store.load(sym, timeframe)
                        if sym_df is None or len(sym_df) == 0:
                            st.warning(f"No data for {sym}, skipping.")
                            continue
                        loaded[sym] = fe.generate_features(sym_df)
                        progress.progress(
                            (idx + 1) / len(selected_symbols),
                            text=f"Loaded {sym}",
                        )

                    if not loaded:
                        st.error("No data loaded for any selected symbol.")
                    else:
                        # Store all loaded symbols
                        st.session_state["feature_data_all"] = loaded
                        st.session_state["feature_engineer"] = fe
                        st.session_state["fe_symbols"] = list(loaded.keys())
                        st.session_state["timeframe"] = timeframe

                        # Set active symbol to first loaded (used by Model Training)
                        first_sym = list(loaded.keys())[0]
                        st.session_state["feature_data"] = loaded[first_sym]
                        st.session_state["symbol"] = first_sym

                        n_feat = len(fe.get_feature_names(loaded[first_sym]))
                        st.success(
                            f"Loaded {len(loaded)} symbol(s) with {n_feat} features each"
                        )

    with col2:
        if "feature_data_all" in st.session_state:
            fe_syms = st.session_state.get("fe_symbols", [])

            # Let user pick which symbol to visualize
            viz_sym = st.selectbox(
                "Symbol to visualize",
                options=fe_syms,
                key="fe_viz_sym",
            )

            df_features = st.session_state["feature_data_all"][viz_sym]
            fe = st.session_state["feature_engineer"]
            feature_names = fe.get_feature_names(df_features)

            st.subheader(f"Feature Statistics — {viz_sym}")

            # Feature selector
            selected_features = st.multiselect(
                "Select features to visualize",
                options=feature_names,
                default=feature_names[:5] if len(feature_names) >= 5 else feature_names,
            )

            if selected_features:
                # Create time series plot
                fig = go.Figure()

                for feature in selected_features:
                    fig.add_trace(
                        go.Scatter(
                            x=df_features.index,
                            y=df_features[feature],
                            name=feature,
                            mode="lines",
                        )
                    )

                fig.update_layout(
                    title=f"Feature Time Series — {viz_sym}",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=400,
                    hovermode="x unified",
                )

                st.plotly_chart(fig, width="stretch")

                # Feature statistics table
                st.subheader("Feature Statistics")
                stats_df = df_features[selected_features].describe()
                st.dataframe(stats_df, use_container_width=True)

# Tab 2: Model Training
with tab2:
    st.header("Model Training")

    if "feature_data_all" not in st.session_state:
        st.info(
            "Please load data and generate features in the Feature Engineering tab first."
        )
    else:
        # Symbol selector for training
        _train_syms = st.session_state.get("fe_symbols", [])
        train_symbol = st.selectbox(
            "Symbol to train on",
            options=_train_syms,
            key="train_symbol_select",
            help="Choose which loaded symbol to use for model training.",
        )

        # Update active feature data when selection changes
        if train_symbol:
            st.session_state["feature_data"] = st.session_state["feature_data_all"][
                train_symbol
            ]
            st.session_state["symbol"] = train_symbol

        model_type = st.radio(
            "Select Model Type", options=["Random Forest", "LSTM"], horizontal=True
        )

        col1, col2 = st.columns([1, 1])

        if model_type == "Random Forest":
            with col1:
                st.subheader("Tree Model Configuration")

                tree_model_type = st.radio(
                    "Model Backend",
                    options=["Random Forest", "Gradient Boosting"],
                    horizontal=True,
                    help="Gradient Boosting uses early stopping and often "
                    "generalizes better on noisy financial data.",
                )

                n_estimators = st.slider("Number of Trees", 50, 500, 300, 50)
                max_depth = st.slider("Max Depth", 3, 20, 5, 1)
                test_size = st.slider("Test Size", 0.1, 0.3, 0.2, 0.05)
                forward_periods = st.slider("Prediction Horizon (periods)", 1, 10, 1)
                return_threshold = (
                    st.slider(
                        "Return Threshold (%)",
                        0.0,
                        5.0,
                        1.0,
                        0.25,
                        help="Drop rows where the forward return is within ±threshold "
                        "of zero. Removes ambiguous / noisy labels.",
                    )
                    / 100.0
                )

                # Feature selection
                st.subheader("Feature Selection")
                fe = st.session_state["feature_engineer"]
                df_features = st.session_state["feature_data"]
                all_rf_features = fe.get_feature_names(df_features)

                # Rank features by absolute correlation with target
                # and default to the top 3
                try:
                    df_ml_preview = fe.prepare_for_ml(
                        df_features,
                        forward_periods=forward_periods,
                        return_threshold=return_threshold,
                    )
                    _target_corr = (
                        df_ml_preview[all_rf_features]
                        .corrwith(df_ml_preview["target"])
                        .abs()
                        .sort_values(ascending=False)
                    )
                    default_rf_features = _target_corr.head(3).index.tolist()
                except Exception:
                    default_rf_features = all_rf_features[:3]

                rf_selected_features = st.multiselect(
                    "Features to use",
                    options=all_rf_features,
                    default=default_rf_features,
                    help=f"{len(all_rf_features)} total features available. "
                    "Defaults are the top 3 by absolute correlation with the target.",
                    key="rf_feature_select",
                )

                if st.button("Train Model", type="primary"):
                    if not rf_selected_features:
                        st.error("Please select at least one feature.")
                    else:
                        with st.spinner("Training model..."):
                            try:
                                df_features = st.session_state["feature_data"]
                                fe = st.session_state["feature_engineer"]

                                # Prepare data for classification
                                df_ml = fe.prepare_for_ml(
                                    df_features,
                                    forward_periods=forward_periods,
                                    return_threshold=return_threshold,
                                )

                                feature_cols = [
                                    f
                                    for f in rf_selected_features
                                    if f in df_ml.columns
                                ]

                                st.info(f"Using {len(feature_cols)} selected features")

                                # Initialize and train model
                                rf_model = RandomForestAnalyzer(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    model_type=(
                                        "gbm"
                                        if tree_model_type == "Gradient Boosting"
                                        else "rf"
                                    ),
                                )

                                data_dict = rf_model.prepare_data(
                                    df_ml,
                                    target_col="target",
                                    feature_cols=feature_cols,
                                    test_size=test_size,
                                    forward_periods=forward_periods,
                                )

                                metrics = rf_model.train(data_dict, verbose=True)

                                # Calibrate probabilities on validation set
                                rf_model.calibrate(data_dict)

                                # Walk-forward validation for robust estimate
                                wf_results = rf_model.walk_forward_validate(
                                    df_ml,
                                    feature_cols=feature_cols,
                                    target_col="target",
                                    n_splits=5,
                                    forward_periods=forward_periods,
                                )

                                # Store in session state
                                st.session_state["rf_model"] = rf_model
                                st.session_state["rf_data_dict"] = data_dict
                                st.session_state["rf_metrics"] = metrics
                                st.session_state["rf_wf_results"] = wf_results

                                st.success("Random Forest training complete!")

                            except Exception as e:
                                st.error(f"Training failed: {e}")
                                import traceback

                                st.code(traceback.format_exc())
                        # end with spinner (indentation close for the else branch)

            with col2:
                if "rf_metrics" in st.session_state:
                    st.subheader("Training Results")

                    metrics = st.session_state["rf_metrics"]

                    # Metrics comparison
                    metrics_df = pd.DataFrame(metrics).T
                    st.dataframe(
                        metrics_df.style.format("{:.4f}"), use_container_width=True
                    )

                    # Walk-forward results
                    if "rf_wf_results" in st.session_state:
                        wf = st.session_state["rf_wf_results"]
                        st.subheader("Walk-Forward Validation")
                        wf_df = pd.DataFrame(wf["fold_metrics"])
                        st.dataframe(
                            wf_df[
                                ["fold", "accuracy", "precision", "recall", "f1_score"]
                            ].style.format(
                                {
                                    "accuracy": "{:.4f}",
                                    "precision": "{:.4f}",
                                    "recall": "{:.4f}",
                                    "f1_score": "{:.4f}",
                                }
                            ),
                            use_container_width=True,
                        )
                        avg = wf["mean"]
                        std = wf["std"]
                        st.markdown(
                            f"**Mean Accuracy: {avg['accuracy']:.4f} "
                            f"± {std['accuracy_std']:.4f}**"
                        )

                    # Visualize metrics
                    fig = go.Figure()

                    for metric in ["accuracy", "precision", "recall", "f1_score"]:
                        fig.add_trace(
                            go.Bar(
                                x=["Train", "Val", "Test"],
                                y=[
                                    metrics["train"][metric],
                                    metrics["val"][metric],
                                    metrics["test"][metric],
                                ],
                                name=metric.replace("_", " ").title(),
                            )
                        )

                    fig.update_layout(
                        title="Performance Metrics by Split",
                        xaxis_title="Dataset",
                        yaxis_title="Score",
                        barmode="group",
                        height=300,
                    )

                    st.plotly_chart(fig, width="stretch")

                    # Save model button
                    if st.button("Save Model"):
                        model_path = (
                            Path(model_dir) / f"rf_{st.session_state['symbol']}"
                        )
                        st.session_state["rf_model"].save(str(model_path))
                        st.success(f"Model saved to {model_path}")

        else:  # LSTM
            with col1:
                st.subheader("LSTM Configuration")

                model_mode = st.radio(
                    "Model Mode",
                    options=["Classifier", "Regressor"],
                    index=0,
                    horizontal=True,
                    help="Classifier (recommended): sigmoid output + binary cross-entropy "
                    "directly optimises direction prediction. "
                    "Regressor: linear output + Huber loss predicts return magnitude.",
                )

                # Defaults tuned via grid search on MSFT 1Day data.
                # Best regression result: seq=40, units=[128,64], lr=1e-4, drop=0.4, bs=32 → 55.97%
                # Classifier adds BatchNorm + BCE on top of those params.
                sequence_length = st.slider("Sequence Length", 20, 120, 40, 10)
                lstm_units_1 = st.slider("LSTM Units (Layer 1)", 32, 256, 128, 32)
                lstm_units_2 = st.slider("LSTM Units (Layer 2)", 16, 128, 64, 16)
                epochs = st.slider("Epochs", 10, 200, 100, 10)
                batch_size = st.slider("Batch Size", 16, 128, 32, 16)
                dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.4, 0.05)
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                    value=1e-4,
                    format_func=lambda x: f"{x:g}",
                )

                # Feature selection
                st.subheader("Feature Selection")
                _fe = st.session_state["feature_engineer"]
                _df_feat = st.session_state["feature_data"]
                _all_lstm_features = _fe.get_feature_names(_df_feat)

                # Build recommended defaults using priority ordering
                _priority_keys = [
                    "returns_lag",
                    "log_returns",
                    "rsi",
                    "bb_percent",
                    "macd_histogram",
                    "macd_bullish",
                    "atr_percent",
                    "bb_bandwidth",
                    "volume_ratio",
                    "volume_change",
                    "close_open_range",
                    "high_low_range",
                    "bullish_candle",
                    "candle_body",
                    "_ratio",
                    "_distance",
                    "macd",
                ]
                _seen: set = set()
                _default_lstm_features = []
                for _key in _priority_keys:
                    for _col in _all_lstm_features:
                        if _key in _col and _col != "returns" and _col not in _seen:
                            _default_lstm_features.append(_col)
                            _seen.add(_col)

                lstm_selected_features = st.multiselect(
                    "Features to use",
                    options=_all_lstm_features,
                    default=_default_lstm_features[:3],
                    help=f"{len(_all_lstm_features)} total features available. "
                    "Defaults are the top 3 momentum / scale-free features.",
                    key="lstm_feature_select",
                )

                if st.button("Train LSTM", type="primary"):
                    if not lstm_selected_features:
                        st.error("Please select at least one feature.")
                    else:
                        with st.spinner(
                            "Training LSTM model (this may take a while)..."
                        ):
                            try:
                                df_features = st.session_state["feature_data"]

                                # Initialize LSTM
                                lstm_model = LSTMPredictor(
                                    sequence_length=sequence_length,
                                    lstm_units=[lstm_units_1, lstm_units_2],
                                    dropout_rate=dropout_rate,
                                    learning_rate=learning_rate,
                                    model_type=model_mode.lower(),
                                )

                                important_features = lstm_selected_features

                                data_dict = lstm_model.prepare_data(
                                    df_features,
                                    target_col="returns",
                                    target_type="return",
                                    feature_cols=important_features,
                                    test_size=0.2,
                                    val_size=0.15,
                                )

                                # Train model
                                history = lstm_model.train(
                                    data_dict,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    verbose=0,
                                )

                                # Store in session state
                                st.session_state["lstm_model"] = lstm_model
                                st.session_state["lstm_data_dict"] = data_dict
                                st.session_state["lstm_history"] = history
                                st.session_state["lstm_model_mode"] = model_mode

                                st.success("LSTM training complete!")

                            except Exception as e:
                                st.error(f"Training failed: {e}")
                                import traceback

                                st.code(traceback.format_exc())
                    # end with spinner (indentation close for the else branch)

            with col2:
                if "lstm_history" in st.session_state:
                    st.subheader("Training History")

                    history = st.session_state["lstm_history"]
                    _mode = st.session_state.get("lstm_model_mode", "Regressor")
                    _is_clf = _mode == "Classifier"

                    # Right-subplot shows accuracy for classifier, MAE for regressor.
                    if _is_clf:
                        metric_key = "accuracy"
                        val_metric_key = "val_accuracy"
                        metric_label = "Accuracy"
                    else:
                        metric_key = (
                            "mae" if "mae" in history else "mean_absolute_error"
                        )
                        val_metric_key = (
                            "val_mae"
                            if "val_mae" in history
                            else "val_mean_absolute_error"
                        )
                        metric_label = "MAE"

                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=["Loss", metric_label],
                    )

                    fig.add_trace(
                        go.Scatter(y=history["loss"], name="Train Loss", mode="lines"),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=history["val_loss"], name="Val Loss", mode="lines"
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=history[metric_key],
                            name=f"Train {metric_label}",
                            mode="lines",
                        ),
                        row=1,
                        col=2,
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=history[val_metric_key],
                            name=f"Val {metric_label}",
                            mode="lines",
                        ),
                        row=1,
                        col=2,
                    )

                    fig.update_xaxes(title_text="Epoch", row=1, col=1)
                    fig.update_xaxes(title_text="Epoch", row=1, col=2)
                    fig.update_layout(height=300, showlegend=True)

                    st.plotly_chart(fig, width="stretch")

                    # Evaluation metrics
                    lstm_model = st.session_state["lstm_model"]
                    data_dict = st.session_state["lstm_data_dict"]

                    eval_metrics = lstm_model.evaluate(
                        data_dict["X_test"], data_dict["y_test"]
                    )

                    st.subheader("Test Set Metrics")
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(
                        4
                    )

                    if _is_clf:
                        metrics_col1.metric(
                            "Brier RMSE",
                            f"{eval_metrics['rmse']:.4f}",
                            help="√MSE between predicted probability and binary label. "
                            "Lower is better; 0.5 is random.",
                        )
                        metrics_col2.metric(
                            "Brier MAE",
                            f"{eval_metrics['mae']:.4f}",
                            help="Mean absolute error between predicted probability and label.",
                        )
                    else:
                        metrics_col1.metric(
                            "RMSE (return space)",
                            f"{eval_metrics['rmse']:.6f}",
                            help="Root mean squared error in daily-return units (e.g. 0.01 = 1%).",
                        )
                        metrics_col2.metric(
                            "MAE (return space)",
                            f"{eval_metrics['mae']:.6f}",
                            help="Mean absolute error in daily-return units.",
                        )

                    metrics_col3.metric(
                        "Directional Accuracy",
                        f"{eval_metrics['directional_accuracy']:.2%}",
                        help="Fraction of bars where predicted direction matches actual. "
                        "50% = random; 55-60% is good; >60% is excellent.",
                    )

                    if _is_clf:
                        _cov = eval_metrics.get("confident_coverage", 1.0)
                        metrics_col4.metric(
                            "Confident DA",
                            f"{eval_metrics.get('confident_directional_accuracy', eval_metrics['directional_accuracy']):.2%}",
                            help=f"Directional accuracy restricted to predictions where "
                            f"P(up) \u2265 0.55 or \u2264 0.45 ({_cov:.0%} of test bars). "
                            "This is the relevant trading metric — we only act when the "
                            "model has real conviction.",
                        )
                    # Save model button
                    if st.button("Save Model"):
                        model_path = (
                            Path(model_dir) / f"lstm_{st.session_state['symbol']}"
                        )
                        lstm_model.save(str(model_path))
                        st.success(f"Model saved to {model_path}")

# Tab 3: Predictions & Analysis
with tab3:
    st.header("Predictions & Feature Importance")

    if "rf_model" not in st.session_state and "lstm_model" not in st.session_state:
        st.info("Please train a model in the Model Training tab first.")
    else:
        analysis_type = st.radio(
            "Analysis Type",
            options=["Predictions", "Feature Importance"],
            horizontal=True,
        )

        if analysis_type == "Predictions":
            st.subheader("Model Predictions")

            col1, col2 = st.columns([1, 1])

            with col1:
                if "lstm_model" in st.session_state:
                    st.write("**LSTM Predictions**")

                    lstm_model = st.session_state["lstm_model"]
                    data_dict = st.session_state["lstm_data_dict"]
                    df_features = st.session_state["feature_data"]

                    # Get predictions
                    pred_df = lstm_model.get_predictions_df(df_features, data_dict)

                    # Plot
                    fig = go.Figure()

                    for split in ["train", "val", "test"]:
                        split_data = pred_df[pred_df["split"] == split]
                        fig.add_trace(
                            go.Scatter(
                                x=split_data.index,
                                y=split_data["actual"],
                                name=f"Actual ({split})",
                                mode="lines",
                                line=dict(width=1),
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=split_data.index,
                                y=split_data["predicted"],
                                name=f"Predicted ({split})",
                                mode="lines",
                                line=dict(dash="dash", width=1),
                            )
                        )

                    _pred_mode = st.session_state.get("lstm_model_mode", "Regressor")
                    _chart_title = (
                        "LSTM Direction Probability"
                        if _pred_mode == "Classifier"
                        else "LSTM Return Predictions"
                    )
                    _y_label = (
                        "P(up) — probability"
                        if _pred_mode == "Classifier"
                        else "Return"
                    )

                    fig.update_layout(
                        title=_chart_title,
                        xaxis_title="Date",
                        yaxis_title=_y_label,
                        height=400,
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, width="stretch")

            with col2:
                if "rf_model" in st.session_state:
                    st.write("**Random Forest Predictions**")

                    rf_model = st.session_state["rf_model"]
                    data_dict = st.session_state["rf_data_dict"]

                    # Get predictions on test set
                    y_test = data_dict["y_test"]
                    y_pred = rf_model.predict(data_dict["X_test"])

                    # Accuracy over time
                    results_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})

                    results_df["correct"] = (
                        results_df["actual"] == results_df["predicted"]
                    ).astype(int)
                    results_df["rolling_accuracy"] = (
                        results_df["correct"].rolling(window=50).mean()
                    )

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            y=results_df["rolling_accuracy"],
                            name="Rolling Accuracy (50 periods)",
                            mode="lines",
                            fill="tozeroy",
                        )
                    )

                    fig.update_layout(
                        title="Random Forest Rolling Accuracy",
                        xaxis_title="Sample",
                        yaxis_title="Accuracy",
                        height=400,
                    )

                    st.plotly_chart(fig, width="stretch")

        else:  # Feature Importance
            st.subheader("Feature Importance Analysis")

            if "rf_model" in st.session_state:
                rf_model = st.session_state["rf_model"]
                data_dict = st.session_state["rf_data_dict"]

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.write("**Random Forest Feature Importance**")

                    importance_df = rf_model.get_feature_importance(top_n=20)

                    fig = px.bar(
                        importance_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Top 20 Features by Importance",
                    )
                    fig.update_layout(
                        height=500, yaxis={"categoryorder": "total ascending"}
                    )

                    st.plotly_chart(fig, width="stretch")

                with col2:
                    st.write("**SHAP Values**")

                    with st.spinner("Calculating SHAP values..."):
                        try:
                            shap_summary = rf_model.get_shap_summary(
                                data_dict["X_test"][:1000],  # Use subset for speed
                                top_n=20,
                            )

                            fig = px.bar(
                                shap_summary,
                                x="shap_importance",
                                y="feature",
                                orientation="h",
                                title="Top 20 Features by SHAP Importance",
                            )
                            fig.update_layout(
                                height=500, yaxis={"categoryorder": "total ascending"}
                            )

                            st.plotly_chart(fig, width="stretch")

                        except ImportError:
                            st.warning(
                                "SHAP library not installed. Install with: pip install shap"
                            )
                        except Exception as e:
                            st.error(f"SHAP calculation failed: {e}")

# Tab 4: Strategy Comparison
with tab4:
    st.header("Strategy Comparison")

    if "rf_model" not in st.session_state and "lstm_model" not in st.session_state:
        st.info("Please train at least one model in the Model Training tab first.")
    else:
        st.subheader("Backtest Configuration")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Symbol selection
            current_symbol = st.session_state.get("symbol", "")
            bt_symbols = st.multiselect(
                "Symbols to backtest",
                options=available_symbols,
                default=[current_symbol] if current_symbol in available_symbols else [],
                help="Select one or more symbols. Each strategy will be backtested "
                "on every symbol independently.",
                key="bt_symbol_select",
            )

            bt_timeframe = st.session_state.get("timeframe", "1Day")

            initial_capital = st.number_input(
                "Initial Capital", value=100000, step=10000
            )
            commission = st.number_input("Commission (%)", value=0.1, step=0.05) / 100

            if st.button("Run Backtest Comparison", type="primary"):
                if not bt_symbols:
                    st.error("Please select at least one symbol.")
                else:
                    with st.spinner("Running backtests..."):
                        try:
                            fe = st.session_state["feature_engineer"]

                            # Collect per-symbol results
                            # all_results: {symbol: {strategy_name: {equity, returns, metrics}}}
                            all_results: dict = {}
                            progress = st.progress(0)

                            for sym_idx, sym in enumerate(bt_symbols):
                                st.write(f"**{sym}** — loading data...")
                                sym_df = data_store.load(sym, bt_timeframe)
                                if sym_df is None or len(sym_df) == 0:
                                    st.warning(f"No data for {sym}, skipping.")
                                    continue

                                sym_features = fe.generate_features(sym_df)
                                sym_results: dict = {}

                                # Buy & Hold
                                bh_ret = sym_features["close"].pct_change().fillna(0)
                                sym_results["Buy & Hold"] = {
                                    "returns": bh_ret,
                                    "equity": (1 + bh_ret).cumprod() * initial_capital,
                                }

                                # MA Crossover
                                ma_strategy = MovingAverageCrossover(
                                    fast_window=50, slow_window=200
                                )
                                backtest_ma = VectorizedBacktest(
                                    strategy=ma_strategy,
                                    data=sym_features,
                                    initial_capital=initial_capital,
                                    commission=commission,
                                )
                                ma_result = backtest_ma.run()
                                sym_results["MA Crossover"] = {
                                    "returns": ma_result["strategy_return"],
                                    "equity": ma_result["equity"],
                                    "metrics": backtest_ma.get_summary(),
                                }

                                # Random Forest
                                if "rf_model" in st.session_state:
                                    rf_strategy = RandomForestStrategy(
                                        rf_model=st.session_state["rf_model"],
                                        feature_engineer=fe,
                                        confidence_threshold=0.6,
                                    )
                                    backtest_rf = VectorizedBacktest(
                                        strategy=rf_strategy,
                                        data=sym_features,
                                        initial_capital=initial_capital,
                                        commission=commission,
                                    )
                                    rf_result = backtest_rf.run()
                                    sym_results["Random Forest"] = {
                                        "returns": rf_result["strategy_return"],
                                        "equity": rf_result["equity"],
                                        "metrics": backtest_rf.get_summary(),
                                    }

                                # LSTM
                                if "lstm_model" in st.session_state:
                                    lstm_strategy = LSTMStrategy(
                                        lstm_model=st.session_state["lstm_model"],
                                        feature_engineer=fe,
                                        confidence_threshold=0.02,
                                    )
                                    backtest_lstm = VectorizedBacktest(
                                        strategy=lstm_strategy,
                                        data=sym_features,
                                        initial_capital=initial_capital,
                                        commission=commission,
                                    )
                                    lstm_result = backtest_lstm.run()
                                    sym_results["LSTM"] = {
                                        "returns": lstm_result["strategy_return"],
                                        "equity": lstm_result["equity"],
                                        "metrics": backtest_lstm.get_summary(),
                                    }

                                all_results[sym] = sym_results
                                progress.progress(
                                    (sym_idx + 1) / len(bt_symbols),
                                    text=f"Completed {sym}",
                                )

                            st.session_state["backtest_results"] = all_results
                            st.session_state["backtest_symbols"] = list(
                                all_results.keys()
                            )
                            st.success(
                                f"Backtests complete for {len(all_results)} symbol(s)!"
                            )

                        except Exception as e:
                            st.error(f"Backtest failed: {e}")
                            import traceback

                            st.code(traceback.format_exc())

        with col2:
            if "backtest_results" in st.session_state:
                all_results = st.session_state["backtest_results"]
                bt_syms = st.session_state.get("backtest_symbols", [])

                if len(bt_syms) == 0:
                    st.info("No results to display.")
                elif len(bt_syms) == 1:
                    # ---- Single-symbol view (same as before) ----
                    sym = bt_syms[0]
                    results = all_results[sym]

                    fig = go.Figure()
                    for strategy_name, result in results.items():
                        fig.add_trace(
                            go.Scatter(
                                x=result["equity"].index,
                                y=result["equity"],
                                name=strategy_name,
                                mode="lines",
                            )
                        )
                    fig.update_layout(
                        title=f"Equity Curves — {sym}",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=400,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, width="stretch")

                    st.subheader("Performance Metrics")
                    metrics_comparison = []
                    for strategy_name, result in results.items():
                        if "metrics" in result:
                            m = result["metrics"]
                            metrics_comparison.append(
                                {
                                    "Strategy": strategy_name,
                                    "Total Return": f"{m.get('total_return', 0):.2%}",
                                    "Sharpe Ratio": f"{m.get('sharpe_ratio', 0):.3f}",
                                    "Max Drawdown": f"{m.get('max_drawdown', 0):.2%}",
                                    "Win Rate": f"{m.get('win_rate', 0):.2%}",
                                    "Trades": m.get("n_trades", 0),
                                }
                            )
                    if metrics_comparison:
                        st.dataframe(
                            pd.DataFrame(metrics_comparison),
                            use_container_width=True,
                            hide_index=True,
                        )

                else:
                    # ---- Multi-symbol view ----
                    # Per-symbol detail selector
                    detail_sym = st.selectbox(
                        "View equity curves for",
                        options=bt_syms,
                        key="bt_detail_sym",
                    )
                    results = all_results[detail_sym]

                    fig = go.Figure()
                    for strategy_name, result in results.items():
                        fig.add_trace(
                            go.Scatter(
                                x=result["equity"].index,
                                y=result["equity"],
                                name=strategy_name,
                                mode="lines",
                            )
                        )
                    fig.update_layout(
                        title=f"Equity Curves — {detail_sym}",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=400,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, width="stretch")

                    # Per-symbol metrics table
                    st.subheader("Per-Symbol Metrics")
                    _metric_keys = [
                        "total_return",
                        "sharpe_ratio",
                        "max_drawdown",
                        "win_rate",
                        "n_trades",
                    ]
                    per_sym_rows = []
                    for sym in bt_syms:
                        for strat_name, res in all_results[sym].items():
                            if "metrics" not in res:
                                continue
                            m = res["metrics"]
                            per_sym_rows.append(
                                {
                                    "Symbol": sym,
                                    "Strategy": strat_name,
                                    "Total Return": m.get("total_return", 0),
                                    "Sharpe Ratio": m.get("sharpe_ratio", 0),
                                    "Max Drawdown": m.get("max_drawdown", 0),
                                    "Win Rate": m.get("win_rate", 0),
                                    "Trades": m.get("n_trades", 0),
                                }
                            )

                    if per_sym_rows:
                        per_sym_df = pd.DataFrame(per_sym_rows)
                        st.dataframe(
                            per_sym_df.style.format(
                                {
                                    "Total Return": "{:.2%}",
                                    "Sharpe Ratio": "{:.3f}",
                                    "Max Drawdown": "{:.2%}",
                                    "Win Rate": "{:.2%}",
                                    "Trades": "{:.0f}",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                        # Aggregated averages across symbols
                        st.subheader("Average Across Symbols")
                        avg_df = (
                            per_sym_df.groupby("Strategy")[
                                [
                                    "Total Return",
                                    "Sharpe Ratio",
                                    "Max Drawdown",
                                    "Win Rate",
                                    "Trades",
                                ]
                            ]
                            .mean()
                            .reset_index()
                        )
                        st.dataframe(
                            avg_df.style.format(
                                {
                                    "Total Return": "{:.2%}",
                                    "Sharpe Ratio": "{:.3f}",
                                    "Max Drawdown": "{:.2%}",
                                    "Win Rate": "{:.2%}",
                                    "Trades": "{:.0f}",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

st.markdown("---")
st.caption("ML Analysis Tool - Algorithmic Trading Platform")
