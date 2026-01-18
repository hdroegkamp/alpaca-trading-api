"""Data Explorer Page - Visualize and analyze historical market data."""

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

st.set_page_config(page_title="Data Explorer", layout="wide")

st.title("Data Explorer")
st.markdown("Visualize and analyze historical market data")
st.markdown("---")

# Get data directory from session state
data_dir = st.session_state.get("data_dir", "Z:\\market_data")


# Load data store
@st.cache_resource
def get_data_store(data_dir):
    """Load data store (cached)."""
    return DataStore(data_dir=data_dir, organize_by_timeframe=True)


@st.cache_data
def get_inventory(_store):
    """Get inventory of available data (cached)."""
    return _store.get_inventory()


@st.cache_data
def load_symbol_data(_store, symbol, timeframe):
    """Load data for a symbol (cached)."""
    return _store.load(symbol, timeframe)


try:
    store = get_data_store(data_dir)
    inventory = get_inventory(store)

    if inventory.empty:
        st.error(f"No data found in {data_dir}. Please download data first.")
        st.info(
            'Run: `python scripts/batch_download.py --universe starter --start 2020-01-01 --data-dir "Z:\\market_data"`'
        )
        st.stop()

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar - Symbol selection
with st.sidebar:
    st.header("Data Selection")

    # Timeframe filter
    available_timeframes = sorted(inventory["timeframe"].unique())
    selected_timeframe = st.selectbox(
        "Timeframe",
        available_timeframes,
        index=0 if "1Day" in available_timeframes else 0,
    )

    # Filter inventory by timeframe
    timeframe_inventory = inventory[inventory["timeframe"] == selected_timeframe]

    # Symbol selection
    available_symbols = sorted(timeframe_inventory["symbol"].unique())

    # Add symbol search
    search_term = st.text_input("Search symbols", "").upper()
    if search_term:
        filtered_symbols = [s for s in available_symbols if search_term in s]
    else:
        filtered_symbols = available_symbols

    selected_symbol = st.selectbox(
        "Symbol", filtered_symbols, index=0 if filtered_symbols else None
    )

    # Multi-symbol comparison option
    st.markdown("---")
    compare_mode = st.checkbox("Compare Multiple Symbols")

    if compare_mode:
        comparison_symbols = st.multiselect(
            "Select symbols to compare",
            available_symbols,
            default=[selected_symbol] if selected_symbol else [],
        )

    st.markdown("---")

    # Display options
    st.subheader("Display Options")
    show_volume = st.checkbox("Show Volume", value=True)
    show_ma = st.checkbox("Show Moving Averages", value=False)

    if show_ma:
        ma_windows = st.multiselect(
            "MA Periods", [10, 20, 50, 100, 200], default=[20, 50]
        )

# Main content
if not compare_mode:
    # Single symbol analysis
    if selected_symbol:
        # Load data
        data = load_symbol_data(store, selected_symbol, selected_timeframe)

        if data is not None and len(data) > 0:
            # Data info
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Symbol", selected_symbol)
            with col2:
                st.metric("Bars", f"{len(data):,}")
            with col3:
                st.metric("Start Date", data.index[0].strftime("%Y-%m-%d"))
            with col4:
                st.metric("End Date", data.index[-1].strftime("%Y-%m-%d"))

            # Date range selector
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=data.index[0].date(),
                    min_value=data.index[0].date(),
                    max_value=data.index[-1].date(),
                )

            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=data.index[-1].date(),
                    min_value=data.index[0].date(),
                    max_value=data.index[-1].date(),
                )

            # Filter data by date range
            mask = (data.index.date >= start_date) & (data.index.date <= end_date)
            filtered_data = data.loc[mask]

            if len(filtered_data) == 0:
                st.warning("No data available for selected date range")
                st.stop()

            # Create price chart
            if show_volume:
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"{selected_symbol} Price", "Volume"),
                )
            else:
                fig = go.Figure()

            # Candlestick chart
            candlestick = go.Candlestick(
                x=filtered_data.index,
                open=filtered_data["open"],
                high=filtered_data["high"],
                low=filtered_data["low"],
                close=filtered_data["close"],
                name="OHLC",
            )

            if show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)

            # Add moving averages if selected
            if show_ma and ma_windows:
                for window in ma_windows:
                    ma = filtered_data["close"].rolling(window=window).mean()
                    ma_trace = go.Scatter(
                        x=filtered_data.index,
                        y=ma,
                        name=f"MA{window}",
                        line=dict(width=1),
                    )
                    if show_volume:
                        fig.add_trace(ma_trace, row=1, col=1)
                    else:
                        fig.add_trace(ma_trace)

            # Add volume bars
            if show_volume:
                colors = [
                    (
                        "red"
                        if filtered_data["close"].iloc[i]
                        < filtered_data["open"].iloc[i]
                        else "green"
                    )
                    for i in range(len(filtered_data))
                ]

                volume_trace = go.Bar(
                    x=filtered_data.index,
                    y=filtered_data["volume"],
                    name="Volume",
                    marker_color=colors,
                    showlegend=False,
                )
                fig.add_trace(volume_trace, row=2, col=1)

            # Update layout
            fig.update_layout(
                title=f"{selected_symbol} - {selected_timeframe}",
                xaxis_rangeslider_visible=False,
                height=700 if show_volume else 600,
                hovermode="x unified",
            )

            if show_volume:
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
            else:
                fig.update_yaxes(title_text="Price")

            fig.update_xaxes(title_text="Date")

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            st.markdown("---")
            st.subheader("Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Price Statistics**")
                st.metric("Current Price", f"${filtered_data['close'].iloc[-1]:.2f}")
                st.metric("Period High", f"${filtered_data['high'].max():.2f}")
                st.metric("Period Low", f"${filtered_data['low'].min():.2f}")
                st.metric("Average Close", f"${filtered_data['close'].mean():.2f}")

            with col2:
                st.markdown("**Returns**")
                returns = filtered_data["close"].pct_change()
                total_return = (
                    filtered_data["close"].iloc[-1] / filtered_data["close"].iloc[0]
                ) - 1
                st.metric("Total Return", f"{total_return*100:.2f}%")
                st.metric("Avg Daily Return", f"{returns.mean()*100:.3f}%")
                st.metric("Daily Volatility", f"{returns.std()*100:.3f}%")
                st.metric(
                    "Sharpe Ratio (Annual)",
                    f"{(returns.mean()/returns.std())*float((252**0.5)):.2f}",
                )

            with col3:
                st.markdown("**Volume**")
                st.metric("Avg Volume", f"{filtered_data['volume'].mean():,.0f}")
                st.metric("Max Volume", f"{filtered_data['volume'].max():,.0f}")
                st.metric("Min Volume", f"{filtered_data['volume'].min():,.0f}")
                st.metric("Total Volume", f"{filtered_data['volume'].sum():,.0f}")

            # Data table
            st.markdown("---")
            with st.expander("View Raw Data"):
                st.dataframe(
                    filtered_data.sort_index(ascending=False).head(100),
                    use_container_width=True,
                )

                # Download button
                csv = filtered_data.to_csv()
                st.download_button(
                    label="Download Full Dataset as CSV",
                    data=csv,
                    file_name=f"{selected_symbol}_{selected_timeframe}.csv",
                    mime="text/csv",
                )

        else:
            st.error(f"Could not load data for {selected_symbol}")

else:
    # Multi-symbol comparison
    if comparison_symbols and len(comparison_symbols) > 0:
        st.subheader("Multi-Symbol Comparison")

        # Load all selected symbols
        comparison_data = {}
        for symbol in comparison_symbols:
            data = load_symbol_data(store, symbol, selected_timeframe)
            if data is not None:
                comparison_data[symbol] = data

        if not comparison_data:
            st.warning("No data available for selected symbols")
            st.stop()

        # Normalize prices for comparison
        st.markdown("### Normalized Price Comparison (Base 100)")

        fig = go.Figure()

        for symbol, data in comparison_data.items():
            normalized = (data["close"] / data["close"].iloc[0]) * 100
            fig.add_trace(
                go.Scatter(x=data.index, y=normalized, name=symbol, mode="lines")
            )

        fig.update_layout(
            title="Normalized Price Comparison",
            xaxis_title="Date",
            yaxis_title="Normalized Price (Base 100)",
            height=600,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Performance comparison table
        st.markdown("---")
        st.subheader("Performance Comparison")

        comparison_stats = []
        for symbol, data in comparison_data.items():
            returns = data["close"].pct_change()
            total_return = (data["close"].iloc[-1] / data["close"].iloc[0]) - 1

            comparison_stats.append(
                {
                    "Symbol": symbol,
                    "Start Price": f"${data['close'].iloc[0]:.2f}",
                    "End Price": f"${data['close'].iloc[-1]:.2f}",
                    "Total Return": f"{total_return*100:.2f}%",
                    "Volatility": f"{returns.std()*100:.2f}%",
                    "Sharpe": f"{(returns.mean()/returns.std())*float((252**0.5)):.2f}",
                    "Bars": len(data),
                }
            )

        comparison_df = pd.DataFrame(comparison_stats)
        st.dataframe(comparison_df, use_container_width=True)

    else:
        st.info("Select symbols in the sidebar to compare")
