"""ML Analysis page — naive Random Forest baseline.

Train a small, regularised Random Forest on a curated set of stationary
features, evaluate it honestly (chronological train/val/test split plus
walk-forward validation), inspect feature importance, and backtest the
resulting long/short strategy against simple baselines.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.data.storage import DataStore
from src.trading.ml.feature_engineering import FeatureEngineering
from src.trading.ml.random_forest_analyzer import RandomForestAnalyzer
from src.trading.ml.ml_strategy import RandomForestStrategy
from src.trading.backtest.vectorized import VectorizedBacktest
from src.trading.strategy.examples.moving_average import MovingAverageCrossover

st.title("Machine Learning Analysis")
st.markdown(
    "A simple, testable baseline: one small Random Forest on ~10 stationary "
    "features, validated honestly with walk-forward analysis."
)

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    data_dir = st.text_input(
        "Data Directory", value="Z:\\market_data", help="Path to market data storage"
    )
    model_dir = st.text_input(
        "Model Directory", value="models", help="Directory to save/load trained models"
    )
    Path(model_dir).mkdir(exist_ok=True)

# Initialize data store
try:
    data_store = DataStore(data_dir=data_dir)
    inventory = data_store.get_inventory()
    available_symbols = (
        inventory["symbol"].unique().tolist() if len(inventory) > 0 else []
    )
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Feature Engineering",
        "Model Training",
        "Walk-Forward & Importance",
        "Strategy Comparison",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1: Feature Engineering
# ---------------------------------------------------------------------------
with tab1:
    st.header("Feature Engineering")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Data Selection")
        selected_symbols = st.multiselect(
            "Symbols",
            options=available_symbols,
            default=[],
            help=f"{len(available_symbols)} symbols available.",
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
                            (idx + 1) / len(selected_symbols), text=f"Loaded {sym}"
                        )

                    if not loaded:
                        st.error("No data loaded for any selected symbol.")
                    else:
                        st.session_state["feature_data_all"] = loaded
                        st.session_state["feature_engineer"] = fe
                        st.session_state["fe_symbols"] = list(loaded.keys())
                        st.session_state["timeframe"] = timeframe
                        first_sym = list(loaded.keys())[0]
                        st.session_state["feature_data"] = loaded[first_sym]
                        st.session_state["symbol"] = first_sym
                        n_feat = len(fe.get_feature_names(loaded[first_sym]))
                        st.success(
                            f"Loaded {len(loaded)} symbol(s) with {n_feat} features each"
                        )

        st.caption(
            "Baseline features: " + ", ".join(FeatureEngineering.FEATURE_COLUMNS)
        )

    with col2:
        if "feature_data_all" in st.session_state:
            fe_syms = st.session_state.get("fe_symbols", [])
            viz_sym = st.selectbox(
                "Symbol to visualize", options=fe_syms, key="fe_viz_sym"
            )
            df_features = st.session_state["feature_data_all"][viz_sym]
            fe = st.session_state["feature_engineer"]
            feature_names = fe.get_feature_names(df_features)

            st.subheader(f"Feature Time Series — {viz_sym}")
            selected_features = st.multiselect(
                "Select features to visualize",
                options=feature_names,
                default=feature_names[:3],
            )
            if selected_features:
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
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=400,
                    hovermode="x unified",
                )
                st.plotly_chart(fig, width="stretch")

                st.subheader("Feature Statistics")
                st.dataframe(
                    df_features[selected_features].describe(),
                    use_container_width=True,
                )

# ---------------------------------------------------------------------------
# Tab 2: Model Training
# ---------------------------------------------------------------------------
with tab2:
    st.header("Model Training")

    if "feature_data_all" not in st.session_state:
        st.info("Load data and generate features in the Feature Engineering tab first.")
    else:
        train_syms = st.session_state.get("fe_symbols", [])
        train_symbol = st.selectbox(
            "Symbol to train on", options=train_syms, key="train_symbol_select"
        )
        if train_symbol:
            st.session_state["feature_data"] = st.session_state["feature_data_all"][
                train_symbol
            ]
            st.session_state["symbol"] = train_symbol

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Random Forest Configuration")
            n_estimators = st.slider("Number of Trees", 50, 500, 200, 50)
            max_depth = st.slider("Max Depth", 2, 12, 5, 1)
            min_samples_leaf = st.slider("Min Samples per Leaf", 5, 100, 30, 5)
            test_size = st.slider("Test Size", 0.1, 0.3, 0.2, 0.05)
            val_size = st.slider("Validation Size", 0.05, 0.25, 0.1, 0.05)
            forward_periods = st.slider("Prediction Horizon (periods)", 1, 10, 1)
            return_threshold = (
                st.slider(
                    "Return Threshold (%)",
                    0.0,
                    5.0,
                    1.0,
                    0.25,
                    help="Drop rows where the forward return is within ±threshold "
                    "of zero (ambiguous / noisy labels).",
                )
                / 100.0
            )

            fe = st.session_state["feature_engineer"]
            df_features = st.session_state["feature_data"]
            all_features = fe.get_feature_names(df_features)
            selected_features = st.multiselect(
                "Features to use",
                options=all_features,
                default=all_features,
                help="Defaults to the full curated baseline set.",
                key="rf_feature_select",
            )

            if st.button("Train Model", type="primary"):
                if not selected_features:
                    st.error("Please select at least one feature.")
                else:
                    with st.spinner("Training model..."):
                        try:
                            df_ml = fe.prepare_for_ml(
                                df_features,
                                forward_periods=forward_periods,
                                return_threshold=return_threshold,
                            )
                            feature_cols = [
                                f for f in selected_features if f in df_ml.columns
                            ]
                            rf_model = RandomForestAnalyzer(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                            )
                            data_dict = rf_model.prepare_data(
                                df_ml,
                                target_col="target",
                                feature_cols=feature_cols,
                                test_size=test_size,
                                val_size=val_size,
                                forward_periods=forward_periods,
                            )
                            metrics = rf_model.train(data_dict, verbose=False)
                            wf_results = rf_model.walk_forward_validate(
                                df_ml,
                                feature_cols=feature_cols,
                                target_col="target",
                                n_splits=5,
                                forward_periods=forward_periods,
                            )
                            st.session_state["rf_model"] = rf_model
                            st.session_state["rf_data_dict"] = data_dict
                            st.session_state["rf_metrics"] = metrics
                            st.session_state["rf_wf_results"] = wf_results
                            st.success("Training complete!")
                        except Exception as e:
                            st.error(f"Training failed: {e}")
                            import traceback

                            st.code(traceback.format_exc())

        with col2:
            if "rf_metrics" in st.session_state:
                st.subheader("Metrics by Split")
                metrics = st.session_state["rf_metrics"]
                st.dataframe(
                    pd.DataFrame(metrics).T.style.format("{:.4f}"),
                    use_container_width=True,
                )

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
                    title="Performance by Split",
                    yaxis_title="Score",
                    barmode="group",
                    height=300,
                )
                st.plotly_chart(fig, width="stretch")

                test_acc = metrics["test"]["accuracy"]
                st.caption(
                    f"Test accuracy {test_acc:.1%}. A coin flip is 50%; for daily "
                    "direction, honest models usually land near 50-55%."
                )

                if st.button("Save Model"):
                    model_path = Path(model_dir) / f"rf_{st.session_state['symbol']}"
                    st.session_state["rf_model"].save(str(model_path))
                    st.success(f"Model saved to {model_path}")

# ---------------------------------------------------------------------------
# Tab 3: Walk-Forward & Feature Importance
# ---------------------------------------------------------------------------
with tab3:
    st.header("Walk-Forward Validation & Feature Importance")

    if "rf_model" not in st.session_state:
        st.info("Train a model in the Model Training tab first.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Walk-Forward Validation")
            st.caption(
                "Expanding-window folds with a gap to prevent leakage — the most "
                "honest out-of-sample estimate for a time series."
            )
            wf = st.session_state.get("rf_wf_results")
            if wf:
                wf_df = pd.DataFrame(wf["fold_metrics"])
                st.dataframe(
                    wf_df[
                        ["fold", "train_size", "test_size", "accuracy", "f1_score"]
                    ].style.format({"accuracy": "{:.4f}", "f1_score": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                avg, std = wf["mean"], wf["std"]
                m1, m2 = st.columns(2)
                m1.metric(
                    "Mean Accuracy",
                    f"{avg['accuracy']:.2%}",
                    help=f"± {std['accuracy_std']:.2%} across folds",
                )
                m2.metric("Mean F1", f"{avg['f1_score']:.3f}")

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=wf_df["fold"],
                        y=wf_df["accuracy"],
                        mode="lines+markers",
                        name="Accuracy",
                    )
                )
                fig.add_hline(
                    y=0.5, line_dash="dash", line_color="gray", annotation_text="Random"
                )
                fig.update_layout(
                    title="Accuracy per Fold",
                    xaxis_title="Fold",
                    yaxis_title="Accuracy",
                    height=300,
                )
                st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Feature Importance")
            importance_df = st.session_state["rf_model"].get_feature_importance(
                top_n=20
            )
            fig = px.bar(
                importance_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Feature Importance (Random Forest)",
            )
            fig.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Tab 4: Strategy Comparison
# ---------------------------------------------------------------------------
with tab4:
    st.header("Strategy Comparison")

    if "rf_model" not in st.session_state:
        st.info("Train a model in the Model Training tab first.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Backtest Configuration")
            current_symbol = st.session_state.get("symbol", "")
            bt_symbols = st.multiselect(
                "Symbols to backtest",
                options=available_symbols,
                default=[current_symbol] if current_symbol in available_symbols else [],
                key="bt_symbol_select",
            )
            bt_timeframe = st.session_state.get("timeframe", "1Day")
            initial_capital = st.number_input(
                "Initial Capital", value=100000, step=10000
            )
            commission = st.number_input("Commission (%)", value=0.1, step=0.05) / 100
            long_threshold = st.slider("Long Threshold P(up) ≥", 0.50, 0.75, 0.55, 0.01)
            short_threshold = st.slider(
                "Short Threshold P(up) ≤", 0.25, 0.50, 0.45, 0.01
            )

            if st.button("Run Backtest Comparison", type="primary"):
                if not bt_symbols:
                    st.error("Please select at least one symbol.")
                elif long_threshold <= short_threshold:
                    st.error("Long threshold must be greater than short threshold.")
                else:
                    with st.spinner("Running backtests..."):
                        try:
                            fe = st.session_state["feature_engineer"]
                            all_results: dict = {}
                            progress = st.progress(0)

                            for sym_idx, sym in enumerate(bt_symbols):
                                sym_df = data_store.load(sym, bt_timeframe)
                                if sym_df is None or len(sym_df) == 0:
                                    st.warning(f"No data for {sym}, skipping.")
                                    continue
                                sym_features = fe.generate_features(sym_df)
                                sym_results: dict = {}

                                bh_ret = sym_features["close"].pct_change().fillna(0)
                                sym_results["Buy & Hold"] = {
                                    "returns": bh_ret,
                                    "equity": (1 + bh_ret).cumprod() * initial_capital,
                                }

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

                                rf_strategy = RandomForestStrategy(
                                    rf_model=st.session_state["rf_model"],
                                    feature_engineer=fe,
                                    long_threshold=long_threshold,
                                    short_threshold=short_threshold,
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

                if not bt_syms:
                    st.info("No results to display.")
                else:
                    detail_sym = st.selectbox(
                        "View equity curves for", options=bt_syms, key="bt_detail_sym"
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

                    st.subheader("Performance Metrics")
                    rows = []
                    for sym in bt_syms:
                        for strat_name, res in all_results[sym].items():
                            if "metrics" not in res:
                                continue
                            m = res["metrics"]
                            rows.append(
                                {
                                    "Symbol": sym,
                                    "Strategy": strat_name,
                                    "Total Return": f"{m.get('total_return', 0):.2%}",
                                    "Sharpe Ratio": f"{m.get('sharpe_ratio', 0):.3f}",
                                    "Max Drawdown": f"{m.get('max_drawdown', 0):.2%}",
                                    "Win Rate": f"{m.get('win_rate', 0):.2%}",
                                    "Trades": m.get("n_trades", 0),
                                }
                            )
                    if rows:
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True,
                            hide_index=True,
                        )
