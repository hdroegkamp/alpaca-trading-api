"""Portfolio ML page — pooled, cross-sectional stock-picking.

Trains ONE Random Forest across a whole universe (not one model per symbol),
ranks the universe by predicted score each day, and backtests a long/short
book built from that ranking. This is the "which stocks to buy/sell today"
view — for a single-symbol diagnostic walkthrough, see the ML Analysis page.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.data.storage import DataStore
from src.trading.data.universe import (
    UNIVERSES,
    load_universe_file,
    symbols_from_inventory,
)
from src.trading.ml.feature_engineering import FeatureEngineering
from src.trading.ml.panel import build_panel
from src.trading.ml.cross_sectional_labels import add_cross_sectional_labels
from src.trading.ml.pooled_model import PooledForestModel
from src.trading.backtest.portfolio import PortfolioBacktest

st.title("Portfolio ML — Cross-Sectional Stock Picking")
st.markdown(
    "One pooled Random Forest trained across the whole universe, ranked "
    "cross-sectionally each day, backtested as a long/short book. Scores "
    "from a per-symbol model aren't comparable across symbols — this is "
    "what makes ranking a universe possible."
)

with st.sidebar:
    st.header("Configuration")
    data_dir = st.text_input("Data Directory", value="data", key="pml_data_dir")

try:
    data_store = DataStore(data_dir=data_dir)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(
    ["Universe & Panel", "Pooled Model Training", "Portfolio Backtest"]
)

# ---------------------------------------------------------------------------
# Tab 1: Universe & Panel
# ---------------------------------------------------------------------------
with tab1:
    st.header("Universe & Panel")
    st.info(
        "Survivorship bias: this universe reflects symbols with data "
        "available today. Delisted names are absent, which optimistically "
        "biases backtested performance versus a true point-in-time universe.",
        icon="⚠️",
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Data Selection")
        timeframe = st.selectbox(
            "Timeframe", options=["1Day", "1Hour", "15Min"], index=0, key="pml_timeframe"
        )

        st.subheader("Universe Source")
        source = st.radio(
            "Choose symbols from",
            options=[
                "Curated universe",
                "Universe file",
                "Full local inventory",
                "Manual list",
            ],
            index=0,
            key="pml_universe_source",
        )

        symbols = []
        if source == "Curated universe":
            names = list(UNIVERSES.keys())
            universe_name = st.selectbox(
                "Universe", options=names, index=names.index("starter")
            )
            symbols = UNIVERSES[universe_name]
        elif source == "Universe file":
            universe_dir = Path("universes")
            files = sorted(universe_dir.glob("*.txt")) if universe_dir.exists() else []
            file_choice = st.selectbox(
                "File", options=[str(p) for p in files] if files else []
            )
            if file_choice:
                symbols = load_universe_file(file_choice)
        elif source == "Full local inventory":
            all_symbols = symbols_from_inventory(data_store, timeframe=timeframe)
            st.warning(
                f"{len(all_symbols)} symbols available locally. Building a panel "
                "across the full universe and multi-year history can take a long "
                "time and a lot of memory — consider capping it below for "
                "interactive iteration."
            )
            cap = st.number_input(
                "Limit to first N symbols (0 = no limit)",
                min_value=0,
                value=200,
                step=50,
            )
            symbols = all_symbols[:cap] if cap > 0 else all_symbols
        else:
            manual = st.text_area("Symbols (comma or whitespace separated)")
            symbols = [
                s.strip().upper() for s in manual.replace(",", " ").split() if s.strip()
            ]

        st.caption(f"{len(symbols)} symbol(s) selected.")

        st.subheader("Panel Parameters")
        forward_periods = st.slider(
            "Forward Periods (label horizon)", 1, 10, 1, key="pml_forward_periods"
        )
        min_history = st.slider("Min History (rows) per symbol", 20, 300, 70, 10)
        cache_path = st.text_input(
            "Panel cache path (optional)",
            value="",
            help="If set, the assembled panel is cached to this parquet file and "
            "reused on the next run with the same path.",
        )

        if st.button("Build Panel", type="primary"):
            if not symbols:
                st.error("Please select at least one symbol.")
            else:
                progress = st.progress(0, text="Building panel...")

                def on_progress(done, total, symbol):
                    progress.progress(done / total, text=f"[{done}/{total}] {symbol}")

                try:
                    panel = build_panel(
                        symbols,
                        data_store,
                        timeframe=timeframe,
                        forward_periods=forward_periods,
                        min_history=min_history,
                        cache_path=cache_path or None,
                        progress_callback=on_progress,
                    )
                    st.session_state["panel"] = panel
                    st.session_state["panel_forward_periods"] = forward_periods
                    st.success(
                        f"Panel built: {panel.shape[0]:,} rows, "
                        f"{panel['symbol'].nunique()} symbols."
                    )
                except Exception as e:
                    st.error(f"Panel build failed: {e}")
                    import traceback

                    st.code(traceback.format_exc())

    with col2:
        if "panel" in st.session_state:
            panel = st.session_state["panel"]
            st.subheader("Panel Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", f"{len(panel):,}")
            m2.metric("Symbols", panel["symbol"].nunique())
            m3.metric(
                "Date Range",
                f"{panel['date'].min().date()} → {panel['date'].max().date()}",
            )

            coverage = panel.groupby("date")["symbol"].nunique()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=coverage.index,
                    y=coverage.values,
                    mode="lines",
                    name="Symbols per date",
                )
            )
            fig.update_layout(
                title="Per-Date Symbol Coverage",
                xaxis_title="Date",
                yaxis_title="Symbol Count",
                height=300,
            )
            st.plotly_chart(fig, width="stretch")

            st.caption("Sample rows")
            st.dataframe(panel.head(20), use_container_width=True)
        else:
            st.info("Build a panel to see its summary here.")

# ---------------------------------------------------------------------------
# Tab 2: Pooled Model Training
# ---------------------------------------------------------------------------
with tab2:
    st.header("Pooled Model Training")

    if "panel" not in st.session_state:
        st.info("Build a panel in the Universe & Panel tab first.")
    else:
        panel = st.session_state["panel"]
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Cross-Sectional Labeling")
            top_quantile = st.slider("Top Quantile (labeled 1)", 0.05, 0.5, 0.2, 0.05)
            bottom_quantile = st.slider(
                "Bottom Quantile (labeled 0)", 0.05, 0.5, 0.2, 0.05
            )
            drop_middle = st.checkbox("Drop middle band", value=True)
            min_symbols_per_date = st.slider("Min symbols per date", 2, 50, 5)

            st.subheader("Random Forest Configuration")
            n_estimators = st.slider(
                "Number of Trees", 50, 500, 200, 50, key="pml_n_estimators"
            )
            max_depth = st.slider("Max Depth", 2, 12, 5, 1, key="pml_max_depth")
            min_samples_leaf = st.slider(
                "Min Samples per Leaf", 5, 100, 30, 5, key="pml_min_samples_leaf"
            )

            st.subheader("Walk-Forward Validation")
            n_splits = st.slider("Number of Folds", 2, 10, 5)
            embargo_periods = st.slider(
                "Embargo Periods",
                1,
                10,
                st.session_state.get("panel_forward_periods", 1),
                help="Should match the label's forward_periods to prevent leakage.",
            )

            if st.button("Train Pooled Model", type="primary"):
                if top_quantile + bottom_quantile > 1.0:
                    st.error("Top quantile + bottom quantile must not exceed 1.0.")
                else:
                    with st.spinner("Labeling and training..."):
                        try:
                            labeled = add_cross_sectional_labels(
                                panel,
                                top_quantile=top_quantile,
                                bottom_quantile=bottom_quantile,
                                drop_middle=drop_middle,
                                min_symbols_per_date=min_symbols_per_date,
                            )
                            feature_cols = FeatureEngineering().get_feature_names(
                                labeled
                            )
                            model = PooledForestModel(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                            )
                            wf = model.walk_forward_validate_panel(
                                labeled,
                                feature_cols=feature_cols,
                                target_col="target",
                                n_splits=n_splits,
                                embargo_periods=embargo_periods,
                            )
                            st.session_state["pooled_model"] = model
                            st.session_state["labeled_panel"] = labeled
                            st.session_state["pooled_wf"] = wf
                            st.session_state["pooled_feature_cols"] = feature_cols
                            st.success("Training complete!")
                        except Exception as e:
                            st.error(f"Training failed: {e}")
                            import traceback

                            st.code(traceback.format_exc())

        with col2:
            if "pooled_wf" in st.session_state:
                wf = st.session_state["pooled_wf"]
                st.subheader("Walk-Forward Fold Metrics")
                fold_df = pd.DataFrame(
                    [
                        {
                            k: v
                            for k, v in f.items()
                            if k not in ("model", "scaler", "test_dates")
                        }
                        for f in wf["fold_metrics"]
                    ]
                )
                st.dataframe(
                    fold_df.style.format(
                        {
                            "accuracy": "{:.4f}",
                            "precision": "{:.4f}",
                            "recall": "{:.4f}",
                            "f1_score": "{:.4f}",
                        }
                    ),
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
                        x=fold_df["fold"],
                        y=fold_df["accuracy"],
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

                st.subheader("Feature Importance")
                st.caption(
                    "From the most recent fold's model (largest training window)."
                )
                last_fold_model = wf["fold_metrics"][-1]["model"]
                importance_df = pd.DataFrame(
                    {
                        "feature": st.session_state["pooled_feature_cols"],
                        "importance": last_fold_model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                fig2 = px.bar(importance_df, x="importance", y="feature", orientation="h")
                fig2.update_layout(height=350, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig2, width="stretch")
            else:
                st.info("Train a model to see walk-forward results here.")

# ---------------------------------------------------------------------------
# Tab 3: Portfolio Backtest
# ---------------------------------------------------------------------------
with tab3:
    st.header("Portfolio Backtest")
    st.info(
        "Costs are a flat commission fraction on turnover only — no slippage "
        "or market-impact model. Daily long/short rebalancing across many "
        "names generates substantial turnover, so results are sensitive to "
        "the book size and rebalance cadence below.",
        icon="⚠️",
    )

    if "pooled_wf" not in st.session_state:
        st.info("Train a pooled model in the Pooled Model Training tab first.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Backtest Configuration")
            top_pct = st.slider("Long Leg %", 0.05, 0.5, 0.2, 0.05)
            bottom_pct = st.slider("Short Leg %", 0.05, 0.5, 0.2, 0.05)
            commission = (
                st.number_input("Commission (%)", value=0.1, step=0.05) / 100
            )
            rebalance_every_n_days = st.slider("Rebalance Every N Days", 1, 20, 1)
            initial_capital = st.number_input(
                "Initial Capital", value=100000, step=10000
            )

            if st.button("Run Portfolio Backtest", type="primary"):
                with st.spinner("Running portfolio backtest..."):
                    try:
                        wf = st.session_state["pooled_wf"]
                        panel = st.session_state["panel"]
                        feature_cols = st.session_state["pooled_feature_cols"]

                        bt = PortfolioBacktest(
                            panel=panel,
                            fold_models=wf["fold_metrics"],
                            feature_columns=feature_cols,
                            top_pct=top_pct,
                            bottom_pct=bottom_pct,
                            commission=commission,
                            rebalance_every_n_days=rebalance_every_n_days,
                            initial_capital=initial_capital,
                        )
                        results = bt.run()
                        st.session_state["portfolio_results"] = results
                        st.session_state["portfolio_book"] = bt.book
                        st.session_state["portfolio_summary"] = bt.get_summary()
                        st.success("Backtest complete!")
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
                        import traceback

                        st.code(traceback.format_exc())

        with col2:
            if "portfolio_results" in st.session_state:
                results = st.session_state["portfolio_results"]
                summary = st.session_state["portfolio_summary"]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=results["date"],
                        y=results["equity"],
                        name="Portfolio Equity",
                        mode="lines",
                    )
                )
                fig.update_layout(
                    title="Portfolio Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Equity ($)",
                    height=350,
                )
                st.plotly_chart(fig, width="stretch")

                fig_dd = go.Figure()
                fig_dd.add_trace(
                    go.Scatter(
                        x=results["date"],
                        y=results["drawdown"] * 100,
                        fill="tozeroy",
                        name="Drawdown",
                    )
                )
                fig_dd.update_layout(
                    title="Drawdown",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    height=250,
                )
                st.plotly_chart(fig_dd, width="stretch")

                st.subheader("Summary")
                st.dataframe(
                    pd.DataFrame([summary])
                    .T.rename(columns={0: "value"})
                    .style.format("{:.4f}"),
                    use_container_width=True,
                )

                st.subheader("Book on a Date")
                book = st.session_state["portfolio_book"]
                book_dates = sorted(book["date"].unique())
                selected_date = st.select_slider(
                    "Date", options=book_dates, value=book_dates[-1]
                )
                st.dataframe(
                    book[book["date"] == selected_date].sort_values(
                        "weight", ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Run a backtest to see the equity curve here.")
