"""Strategy Analyzer - Test and optimize trading strategies."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.data import DataStore
from src.trading.strategy.examples import MovingAverageCrossover, MeanReversion
from src.trading.backtest import VectorizedBacktest

st.title("Strategy Analyzer")
st.markdown("Test and optimize trading strategies with backtesting")
st.markdown("---")

# Get data directory from session state
data_dir = st.session_state.get("data_dir", "data")


# Load data store
@st.cache_resource
def get_data_store(data_dir):
    return DataStore(data_dir=data_dir, organize_by_timeframe=True)


@st.cache_data
def get_inventory(_store):
    return _store.get_inventory()


@st.cache_data
def load_symbol_data(_store, symbol, timeframe):
    return _store.load(symbol, timeframe)


try:
    store = get_data_store(data_dir)
    inventory = get_inventory(store)

    if inventory.empty:
        st.error(f"No data found in {data_dir}")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar - Configuration
with st.sidebar:
    st.header("Backtest Configuration")

    # Data selection
    available_timeframes = sorted(inventory["timeframe"].unique())
    selected_timeframe = st.selectbox(
        "Timeframe",
        available_timeframes,
        index=0 if "1Day" in available_timeframes else 0,
    )

    timeframe_inventory = inventory[inventory["timeframe"] == selected_timeframe]
    available_symbols = sorted(timeframe_inventory["symbol"].unique())

    selected_symbol = st.selectbox("Symbol", available_symbols)

    st.markdown("---")

    # Strategy selection
    st.subheader("Strategy")
    strategy_type = st.selectbox(
        "Strategy Type", ["Moving Average Crossover", "Mean Reversion"]
    )

    st.markdown("---")

    # Strategy parameters
    st.subheader("Parameters")

    if strategy_type == "Moving Average Crossover":
        fast_window = st.slider("Fast MA Window", 5, 50, 20)
        slow_window = st.slider("Slow MA Window", 20, 200, 50)
        strategy_params = {"fast_window": fast_window, "slow_window": slow_window}

    else:  # Mean Reversion
        window = st.slider("Window", 10, 100, 20)
        num_std = st.slider("Number of Std Devs", 1.0, 3.0, 2.0, 0.1)
        strategy_params = {"window": window, "num_std": num_std}

    st.markdown("---")

    # Backtest parameters
    st.subheader("Backtest Settings")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000,
    )

    commission = (
        st.number_input(
            "Commission Rate (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01
        )
        / 100
    )

    run_backtest = st.button("Run Backtest", type="primary", use_container_width=True)

# Main content
if selected_symbol:
    data = load_symbol_data(store, selected_symbol, selected_timeframe)

    if data is None or len(data) == 0:
        st.error(f"Could not load data for {selected_symbol}")
        st.stop()

    # Show data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Symbol", selected_symbol)
    with col2:
        st.metric("Bars", f"{len(data):,}")
    with col3:
        st.metric("Start", data.index[0].strftime("%Y-%m-%d"))
    with col4:
        st.metric("End", data.index[-1].strftime("%Y-%m-%d"))

    st.markdown("---")

    if run_backtest:
        with st.spinner("Running backtest..."):
            try:
                # Initialize strategy
                if strategy_type == "Moving Average Crossover":
                    strategy = MovingAverageCrossover(**strategy_params)
                else:
                    strategy = MeanReversion(**strategy_params)

                # Run backtest
                backtest = VectorizedBacktest(
                    strategy=strategy,
                    data=data,
                    initial_capital=initial_capital,
                    commission=commission,
                )

                results = backtest.run()
                summary = backtest.get_summary()

                # Store results in session state
                st.session_state["backtest_results"] = {
                    "strategy": strategy_type,
                    "params": strategy_params,
                    "summary": summary,
                    "results": results,
                    "symbol": selected_symbol,
                    "timeframe": selected_timeframe,
                }

                st.success("Backtest completed!")

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

    # Display results if available
    if "backtest_results" in st.session_state:
        results_data = st.session_state["backtest_results"]
        summary = results_data["summary"]
        results = results_data["results"]

        # Performance metrics
        st.subheader("Performance Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Return", f"{summary['total_return']*100:.2f}%", delta=None)
            st.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")

        with col2:
            st.metric("Max Drawdown", f"{summary['max_drawdown']*100:.2f}%")
            st.metric("CAGR", f"{summary.get('cagr', 0)*100:.2f}%")

        with col3:
            st.metric("Win Rate", f"{summary['win_rate']*100:.1f}%")
            st.metric("Volatility", f"{summary['volatility']*100:.2f}%")

        with col4:
            st.metric(
                "Trades", f"{summary.get('n_trades', summary.get('num_trades', 0))}"
            )
            st.metric("Days", f"{summary.get('n_days', len(results))}")

        st.markdown("---")

        # Equity curve
        st.subheader("Equity Curve")

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=("Portfolio Value", "Drawdown"),
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results["equity"],
                name="Equity",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Buy & Hold comparison
        buy_hold_equity = initial_capital * (data["close"] / data["close"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=buy_hold_equity,
                name="Buy & Hold",
                line=dict(color="gray", width=1, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Drawdown
        cummax = results["equity"].cummax()
        drawdown = (results["equity"] - cummax) / cummax * 100

        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=drawdown,
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="red", width=1),
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        fig.update_layout(height=700, hovermode="x unified", showlegend=True)

        st.plotly_chart(fig, width="stretch")

        # Strategy signals visualization
        st.markdown("---")
        st.subheader("Trading Signals")

        # Show last N signals
        signal_changes = results["position"].diff()
        buy_signals = results[signal_changes > 0]
        sell_signals = results[signal_changes < 0]

        fig = go.Figure()

        # Price
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["close"],
                name="Close Price",
                line=dict(color="blue", width=1),
            )
        )

        # Buy signals
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=data.loc[buy_signals.index, "close"],
                    name="Buy",
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="green",
                        line=dict(width=1, color="darkgreen"),
                    ),
                )
            )

        # Sell signals
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=data.loc[sell_signals.index, "close"],
                    name="Sell",
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="red",
                        line=dict(width=1, color="darkred"),
                    ),
                )
            )

        fig.update_layout(
            title=f"{selected_symbol} - Trading Signals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(fig, width="stretch")

        # Trade log
        st.markdown("---")
        with st.expander("View Trade Log"):
            trade_log = pd.DataFrame(
                {
                    "Date": results.index,
                    "Position": results["position"],
                    "Equity": results["equity"],
                    "Returns": results["strategy_return"],
                }
            )

            # Only show rows where position changed
            position_changes = trade_log[trade_log["Position"].diff() != 0]

            st.dataframe(
                position_changes.sort_index(ascending=False).head(50),
                use_container_width=True,
            )

        # All metrics table
        st.markdown("---")
        with st.expander("Detailed Metrics"):
            metrics_df = pd.DataFrame([summary]).T
            metrics_df.columns = ["Value"]
            st.dataframe(metrics_df, use_container_width=True)

    else:
        st.info("Configure parameters in the sidebar and click 'Run Backtest' to start")

        # Show strategy description
        st.markdown("---")
        st.subheader("Strategy Description")

        if strategy_type == "Moving Average Crossover":
            st.markdown(
                """
            **Moving Average Crossover Strategy**
            
            This strategy generates signals based on the crossover of two moving averages:
            - **Long Signal**: When fast MA crosses above slow MA
            - **Short Signal**: When fast MA crosses below slow MA
            
            **Parameters:**
            - Fast MA Window: Shorter period moving average
            - Slow MA Window: Longer period moving average
            
            **Characteristics:**
            - Trend-following strategy
            - Works best in trending markets
            - May generate false signals in sideways markets
            """
            )
        else:
            st.markdown(
                """
            **Mean Reversion Strategy (Bollinger Bands)**
            
            This strategy assumes prices revert to their mean:
            - **Long Signal**: When price crosses below lower Bollinger Band
            - **Short Signal**: When price crosses above upper Bollinger Band
            
            **Parameters:**
            - Window: Lookback period for calculating mean and standard deviation
            - Number of Std Devs: Width of the bands (typically 2.0)
            
            **Characteristics:**
            - Mean-reversion strategy
            - Works best in range-bound markets
            - May underperform in strong trends
            """
            )

else:
    st.info("Select a symbol in the sidebar to begin")
