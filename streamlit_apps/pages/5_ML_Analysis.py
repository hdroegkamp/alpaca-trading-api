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

        # Symbol search input
        symbol_input = st.text_input(
            "Symbol",
            value="",
            placeholder="Type symbol (e.g., AAPL, MSFT, TSLA)",
            help=f"{len(available_symbols)} symbols available",
        ).upper()

        # Validate symbol exists
        if symbol_input and symbol_input not in available_symbols:
            st.warning(
                f"Symbol '{symbol_input}' not found in data. Available symbols: {len(available_symbols)}"
            )
            symbol = None
        else:
            symbol = symbol_input if symbol_input else None

        timeframe = st.selectbox(
            "Timeframe", options=["1Day", "1Hour", "15Min"], index=0
        )

        if st.button("Load & Generate Features", type="primary"):
            with st.spinner("Loading data and generating features..."):
                # Load data
                if symbol is None:
                    st.error("Please select a symbol")
                else:
                    df = data_store.load(symbol, timeframe)

                if df is None or len(df) == 0:
                    st.error(f"No data found for {symbol}")
                else:
                    # Generate features
                    fe = FeatureEngineering()
                    df_features = fe.generate_features(df)

                    # Store in session state
                    st.session_state["feature_data"] = df_features
                    st.session_state["feature_engineer"] = fe
                    st.session_state["symbol"] = symbol
                    st.session_state["timeframe"] = timeframe

                    st.success(
                        f"Loaded {len(df_features)} rows with {len(fe.get_feature_names(df_features))} features"
                    )

    with col2:
        if "feature_data" in st.session_state:
            st.subheader("Feature Statistics")

            df_features = st.session_state["feature_data"]
            fe = st.session_state["feature_engineer"]
            feature_names = fe.get_feature_names(df_features)

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
                    title="Feature Time Series",
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

    if "feature_data" not in st.session_state:
        st.info(
            "Please load data and generate features in the Feature Engineering tab first."
        )
    else:
        model_type = st.radio(
            "Select Model Type", options=["Random Forest", "LSTM"], horizontal=True
        )

        col1, col2 = st.columns([1, 1])

        if model_type == "Random Forest":
            with col1:
                st.subheader("Random Forest Configuration")

                n_estimators = st.slider("Number of Trees", 50, 500, 200, 50)
                max_depth = st.slider("Max Depth", 5, 30, 15, 5)
                test_size = st.slider("Test Size", 0.1, 0.3, 0.2, 0.05)
                forward_periods = st.slider("Prediction Horizon (periods)", 1, 10, 1)

                if st.button("Train Random Forest", type="primary"):
                    with st.spinner("Training Random Forest model..."):
                        try:
                            df_features = st.session_state["feature_data"]
                            fe = st.session_state["feature_engineer"]

                            # Prepare data for classification
                            df_ml = fe.prepare_for_ml(
                                df_features, forward_periods=forward_periods
                            )

                            # Initialize and train model
                            rf_model = RandomForestAnalyzer(
                                n_estimators=n_estimators, max_depth=max_depth
                            )

                            data_dict = rf_model.prepare_data(
                                df_ml, target_col="target", test_size=test_size
                            )

                            metrics = rf_model.train(data_dict, verbose=True)

                            # Store in session state
                            st.session_state["rf_model"] = rf_model
                            st.session_state["rf_data_dict"] = data_dict
                            st.session_state["rf_metrics"] = metrics

                            st.success("Random Forest training complete!")

                        except Exception as e:
                            st.error(f"Training failed: {e}")

            with col2:
                if "rf_metrics" in st.session_state:
                    st.subheader("Training Results")

                    metrics = st.session_state["rf_metrics"]

                    # Metrics comparison
                    metrics_df = pd.DataFrame(metrics).T
                    st.dataframe(
                        metrics_df.style.format("{:.4f}"), use_container_width=True
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

                if st.button("Train LSTM", type="primary"):
                    with st.spinner("Training LSTM model (this may take a while)..."):
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

                            # Prepare data.
                            # Use momentum / ratio / return-based features ONLY.
                            # Raw price-level columns (close, sma_N, ema_N, …) are
                            # excluded on purpose: they cause the model to learn
                            # "tomorrow ≈ today" (MSE-optimal on a random walk) which
                            # collapses directional accuracy to ~50 %.
                            feature_cols = st.session_state[
                                "feature_engineer"
                            ].get_feature_names(df_features)

                            # Keep columns that are normalised relative to price
                            # (ratios, distances) or are already scale-free indicators.
                            # IMPORTANT: no [:30] cap — lagged returns (returns_lag_1
                            # … lag_10) and lagged RSI were the features being cut off
                            # by that limit.  They encode momentum / mean-reversion
                            # and are the strongest directional predictors we have.
                            # Features are ordered so the highest-signal ones (lagged
                            # returns, RSI lags) come first.
                            _priority_keys = [
                                "returns_lag",  # explicit lagged return signal
                                "log_returns",  # log return of current bar
                                "rsi",  # 0-100 oscillator + lags
                                "bb_percent",  # 0-1 band position
                                "macd_histogram",  # MACD divergence (normalised by sign)
                                "macd_bullish",  # binary crossover flag
                                "atr_percent",  # ATR / close
                                "bb_bandwidth",  # volatility ratio
                                "volume_ratio",  # vol / vol_sma
                                "volume_change",
                                "close_open_range",
                                "high_low_range",
                                "bullish_candle",
                                "candle_body",
                                "_ratio",  # sma_N_ratio, ema_N_ratio
                                "_distance",  # sma_N_distance
                                "macd",  # raw MACD & signal (scaled by MinMaxScaler)
                            ]

                            _seen: set = set()
                            important_features = []
                            for _key in _priority_keys:
                                for _col in feature_cols:
                                    if (
                                        _key in _col
                                        and _col != "returns"
                                        and _col not in _seen
                                    ):
                                        important_features.append(_col)
                                        _seen.add(_col)

                            # Target: 1-period return column.  The classifier
                            # binarises it internally; the regressor scales it.
                            # val_size=0.15 gives ~1.7x more validation samples
                            # than the default 0.10, which dramatically reduces
                            # noise in val_loss and prevents early stopping from
                            # firing on epoch 1 due to a lucky first-epoch val score.
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
            initial_capital = st.number_input(
                "Initial Capital", value=100000, step=10000
            )
            commission = st.number_input("Commission (%)", value=0.1, step=0.05) / 100

            if st.button("Run Backtest Comparison", type="primary"):
                with st.spinner("Running backtests..."):
                    try:
                        df = st.session_state["feature_data"]
                        fe = st.session_state["feature_engineer"]

                        results = {}

                        # Benchmark: Buy and Hold
                        st.write("Running Buy & Hold benchmark...")
                        benchmark_returns = df["close"].pct_change().fillna(0)
                        results["Buy & Hold"] = {
                            "returns": benchmark_returns,
                            "equity": (1 + benchmark_returns).cumprod()
                            * initial_capital,
                        }

                        # Traditional strategy: Moving Average Crossover
                        st.write("Running Moving Average Crossover...")
                        ma_strategy = MovingAverageCrossover(
                            fast_window=50, slow_window=200
                        )
                        ma_signals = ma_strategy.generate_signals(df)

                        backtest_ma = VectorizedBacktest(
                            strategy=ma_strategy,
                            data=df,
                            initial_capital=initial_capital,
                            commission=commission,
                        )
                        ma_result = backtest_ma.run()
                        results["MA Crossover"] = {
                            "returns": ma_result["strategy_return"],
                            "equity": ma_result["equity"],
                            "metrics": ma_result,
                        }

                        # ML strategies
                        if "rf_model" in st.session_state:
                            st.write("Running Random Forest strategy...")
                            rf_strategy = RandomForestStrategy(
                                rf_model=st.session_state["rf_model"],
                                feature_engineer=fe,
                                confidence_threshold=0.6,
                            )

                            backtest_rf = VectorizedBacktest(
                                strategy=rf_strategy,
                                data=df,
                                initial_capital=initial_capital,
                                commission=commission,
                            )
                            rf_result = backtest_rf.run()
                            results["Random Forest"] = {
                                "returns": rf_result["strategy_return"],
                                "equity": rf_result["equity"],
                                "metrics": rf_result,
                            }

                        if "lstm_model" in st.session_state:
                            st.write("Running LSTM strategy...")
                            lstm_strategy = LSTMStrategy(
                                lstm_model=st.session_state["lstm_model"],
                                feature_engineer=fe,
                                confidence_threshold=0.02,
                            )

                            backtest_lstm = VectorizedBacktest(
                                strategy=lstm_strategy,
                                data=df,
                                initial_capital=initial_capital,
                                commission=commission,
                            )
                            lstm_result = backtest_lstm.run()
                            results["LSTM"] = {
                                "returns": lstm_result["strategy_return"],
                                "equity": lstm_result["equity"],
                                "metrics": lstm_result,
                            }

                        st.session_state["backtest_results"] = results
                        st.success("Backtests complete!")

                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
                        import traceback

                        st.code(traceback.format_exc())

        with col2:
            if "backtest_results" in st.session_state:
                results = st.session_state["backtest_results"]

                # Plot equity curves
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
                    title="Strategy Equity Curves",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400,
                    hovermode="x unified",
                )

                st.plotly_chart(fig, width="stretch")

                # Performance metrics comparison
                st.subheader("Performance Metrics")

                metrics_comparison = []
                for strategy_name, result in results.items():
                    if "metrics" in result:
                        metrics = result["metrics"]
                        metrics_comparison.append(
                            {
                                "Strategy": strategy_name,
                                "Total Return": f"{metrics.get('total_return', 0):.2%}",
                                "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.3f}",
                                "Max Drawdown": f"{metrics.get('max_drawdown', 0):.2%}",
                                "Win Rate": f"{metrics.get('win_rate', 0):.2%}",
                                "Trades": metrics.get("n_trades", 0),
                            }
                        )

                if metrics_comparison:
                    comparison_df = pd.DataFrame(metrics_comparison)
                    st.dataframe(
                        comparison_df, use_container_width=True, hide_index=True
                    )

st.markdown("---")
st.caption("ML Analysis Tool - Algorithmic Trading Platform")
