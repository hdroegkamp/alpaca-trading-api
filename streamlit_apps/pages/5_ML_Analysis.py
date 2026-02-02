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

st.set_page_config(page_title="ML Analysis", page_icon="🤖", layout="wide")

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

        symbol = st.selectbox(
            "Symbol",
            options=available_symbols[:100],
            index=0 if len(available_symbols) > 0 else None,
        )

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

                st.plotly_chart(fig, use_container_width=True)

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

                    st.plotly_chart(fig, use_container_width=True)

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

                sequence_length = st.slider("Sequence Length", 20, 120, 60, 10)
                lstm_units_1 = st.slider("LSTM Units (Layer 1)", 32, 256, 128, 32)
                lstm_units_2 = st.slider("LSTM Units (Layer 2)", 16, 128, 64, 16)
                epochs = st.slider("Epochs", 10, 200, 50, 10)
                batch_size = st.slider("Batch Size", 16, 128, 32, 16)

                if st.button("Train LSTM", type="primary"):
                    with st.spinner("Training LSTM model (this may take a while)..."):
                        try:
                            df_features = st.session_state["feature_data"]

                            # Initialize LSTM
                            lstm_model = LSTMPredictor(
                                sequence_length=sequence_length,
                                lstm_units=[lstm_units_1, lstm_units_2],
                            )

                            # Prepare data
                            feature_cols = st.session_state[
                                "feature_engineer"
                            ].get_feature_names(df_features)
                            # Limit features for LSTM to avoid overfitting
                            important_features = [
                                col
                                for col in feature_cols
                                if any(
                                    key in col
                                    for key in [
                                        "sma",
                                        "ema",
                                        "rsi",
                                        "macd",
                                        "returns",
                                        "close",
                                    ]
                                )
                            ][:20]

                            data_dict = lstm_model.prepare_data(
                                df_features,
                                target_col="close",
                                feature_cols=important_features,
                                test_size=0.2,
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

                            st.success("LSTM training complete!")

                        except Exception as e:
                            st.error(f"Training failed: {e}")
                            import traceback

                            st.code(traceback.format_exc())

            with col2:
                if "lstm_history" in st.session_state:
                    st.subheader("Training History")

                    history = st.session_state["lstm_history"]

                    # Plot training history
                    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "MAE"])

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
                            y=history["mean_absolute_error"],
                            name="Train MAE",
                            mode="lines",
                        ),
                        row=1,
                        col=2,
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=history["val_mean_absolute_error"],
                            name="Val MAE",
                            mode="lines",
                        ),
                        row=1,
                        col=2,
                    )

                    fig.update_xaxes(title_text="Epoch", row=1, col=1)
                    fig.update_xaxes(title_text="Epoch", row=1, col=2)
                    fig.update_layout(height=300, showlegend=True)

                    st.plotly_chart(fig, use_container_width=True)

                    # Evaluation metrics
                    lstm_model = st.session_state["lstm_model"]
                    data_dict = st.session_state["lstm_data_dict"]

                    eval_metrics = lstm_model.evaluate(
                        data_dict["X_test"], data_dict["y_test"]
                    )

                    st.subheader("Test Set Metrics")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    metrics_col1.metric("RMSE", f"{eval_metrics['rmse']:.4f}")
                    metrics_col2.metric("MAE", f"{eval_metrics['mae']:.4f}")
                    metrics_col3.metric(
                        "Directional Accuracy",
                        f"{eval_metrics['directional_accuracy']:.2%}",
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

                    fig.update_layout(
                        title="LSTM Price Predictions",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=400,
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, use_container_width=True)

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

                    st.plotly_chart(fig, use_container_width=True)

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

                    st.plotly_chart(fig, use_container_width=True)

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

                            st.plotly_chart(fig, use_container_width=True)

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

                st.plotly_chart(fig, use_container_width=True)

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
