"""Technical Analysis - Advanced charting and indicators."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.data import DataStore

st.set_page_config(page_title="Technical Analysis", layout="wide")

st.title("Technical Analysis")
st.markdown("Advanced charting with technical indicators")
st.markdown("---")

# Get data directory
data_dir = st.session_state.get("data_dir", "Z:\\market_data")


@st.cache_resource
def get_data_store(data_dir):
    return DataStore(data_dir=data_dir, organize_by_timeframe=True)


@st.cache_data
def load_symbol_data(_store, symbol, timeframe):
    return _store.load(symbol, timeframe)


try:
    store = get_data_store(data_dir)
    inventory = store.get_inventory()

    if inventory.empty:
        st.error(f"No data found in {data_dir}")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar - Configuration
with st.sidebar:
    st.header("Chart Configuration")

    # Symbol selection
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

    # Indicators
    st.subheader("Technical Indicators")

    show_sma = st.checkbox("Simple Moving Average (SMA)")
    if show_sma:
        sma_periods = st.multiselect(
            "SMA Periods", [10, 20, 50, 100, 200], default=[20, 50]
        )

    show_ema = st.checkbox("Exponential Moving Average (EMA)")
    if show_ema:
        ema_periods = st.multiselect(
            "EMA Periods", [9, 12, 20, 26, 50, 100, 200], default=[12, 26]
        )

    show_bollinger = st.checkbox("Bollinger Bands")
    if show_bollinger:
        bb_period = st.slider("BB Period", 10, 50, 20)
        bb_std = st.slider("BB Std Dev", 1.0, 3.0, 2.0, 0.1)

    show_rsi = st.checkbox("Relative Strength Index (RSI)")
    if show_rsi:
        rsi_period = st.slider("RSI Period", 5, 30, 14)

    show_macd = st.checkbox("MACD")
    if show_macd:
        macd_fast = st.slider("MACD Fast", 5, 20, 12)
        macd_slow = st.slider("MACD Slow", 20, 50, 26)
        macd_signal = st.slider("MACD Signal", 5, 15, 9)

    show_volume = st.checkbox("Volume", value=True)


# Helper functions for indicators
def calculate_rsi(data, period=14):
    """Calculate RSI indicator."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# Main content
if selected_symbol:
    data = load_symbol_data(store, selected_symbol, selected_timeframe)

    if data is None or len(data) == 0:
        st.error(f"Could not load data for {selected_symbol}")
        st.stop()

    # Data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Symbol", selected_symbol)
    with col2:
        st.metric("Current Price", f"${data['close'].iloc[-1]:.2f}")
    with col3:
        daily_change = ((data["close"].iloc[-1] / data["close"].iloc[-2]) - 1) * 100
        st.metric("Daily Change", f"{daily_change:.2f}%", delta=f"{daily_change:.2f}%")
    with col4:
        st.metric("Bars", f"{len(data):,}")

    st.markdown("---")

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=data.index[-252].date() if len(data) > 252 else data.index[0].date(),
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

    # Filter data
    mask = (data.index.date >= start_date) & (data.index.date <= end_date)
    filtered_data = data.loc[mask].copy()

    if len(filtered_data) == 0:
        st.warning("No data available for selected date range")
        st.stop()

    # Determine number of subplot rows
    num_rows = 1
    if show_volume:
        num_rows += 1
    if show_rsi:
        num_rows += 1
    if show_macd:
        num_rows += 1

    # Create subplot layout
    row_heights = [0.5] + [0.15] * (num_rows - 1)
    subplot_titles = ["Price"]

    if show_volume:
        subplot_titles.append("Volume")
    if show_rsi:
        subplot_titles.append("RSI")
    if show_macd:
        subplot_titles.append("MACD")

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=filtered_data.index,
            open=filtered_data["open"],
            high=filtered_data["high"],
            low=filtered_data["low"],
            close=filtered_data["close"],
            name="OHLC",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Add SMAs
    if show_sma and sma_periods:
        for period in sma_periods:
            sma = filtered_data["close"].rolling(window=period).mean()
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=sma,
                    name=f"SMA{period}",
                    line=dict(width=1.5),
                ),
                row=1,
                col=1,
            )

    # Add EMAs
    if show_ema and ema_periods:
        for period in ema_periods:
            ema = filtered_data["close"].ewm(span=period, adjust=False).mean()
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=ema,
                    name=f"EMA{period}",
                    line=dict(width=1.5, dash="dash"),
                ),
                row=1,
                col=1,
            )

    # Add Bollinger Bands
    if show_bollinger:
        sma = filtered_data["close"].rolling(window=bb_period).mean()
        std = filtered_data["close"].rolling(window=bb_period).std()
        upper_band = sma + (bb_std * std)
        lower_band = sma - (bb_std * std)

        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=upper_band,
                name="BB Upper",
                line=dict(width=1, color="gray", dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=lower_band,
                name="BB Lower",
                line=dict(width=1, color="gray", dash="dot"),
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.1)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    current_row = 2

    # Add Volume
    if show_volume:
        colors = [
            (
                "red"
                if filtered_data["close"].iloc[i] < filtered_data["open"].iloc[i]
                else "green"
            )
            for i in range(len(filtered_data))
        ]

        fig.add_trace(
            go.Bar(
                x=filtered_data.index,
                y=filtered_data["volume"],
                name="Volume",
                marker_color=colors,
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )
        current_row += 1

    # Add RSI
    if show_rsi:
        rsi = calculate_rsi(filtered_data["close"], rsi_period)

        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=rsi,
                name="RSI",
                line=dict(color="purple", width=1.5),
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )

        # Add RSI levels
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row=current_row,  # type: ignore
            col=1,  # type: ignore
        )
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            row=current_row,  # type: ignore
            col=1,  # type: ignore
        )
        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="gray",
            opacity=0.3,
            row=current_row,  # type: ignore
            col=1,  # type: ignore
        )

        fig.update_yaxes(range=[0, 100], row=current_row, col=1)  # type: ignore
        current_row += 1

    # Add MACD
    if show_macd:
        macd_line, signal_line, histogram = calculate_macd(
            filtered_data["close"], macd_fast, macd_slow, macd_signal
        )

        # MACD histogram
        colors = ["green" if val >= 0 else "red" for val in histogram]
        fig.add_trace(
            go.Bar(
                x=filtered_data.index,
                y=histogram,
                name="MACD Histogram",
                marker_color=colors,
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )

        # MACD line
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=macd_line,
                name="MACD",
                line=dict(color="blue", width=1.5),
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )

        # Signal line
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=signal_line,
                name="Signal",
                line=dict(color="orange", width=1.5),
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title=f"{selected_symbol} - Technical Analysis",
        xaxis_rangeslider_visible=False,
        height=200 * num_rows + 100,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(title_text="Date", row=num_rows, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Technical summary
    st.markdown("---")
    st.subheader("Technical Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Trend Indicators**")

        # SMA 50 vs 200 (Golden/Death Cross)
        if len(filtered_data) >= 200:
            sma50 = filtered_data["close"].rolling(50).mean().iloc[-1]
            sma200 = filtered_data["close"].rolling(200).mean().iloc[-1]
            current_price = filtered_data["close"].iloc[-1]

            if sma50 > sma200:
                st.success("Golden Cross (Bullish)")
            else:
                st.error("Death Cross (Bearish)")

            st.metric("Price vs SMA50", f"{((current_price/sma50)-1)*100:.2f}%")
            st.metric("Price vs SMA200", f"{((current_price/sma200)-1)*100:.2f}%")

    with col2:
        st.markdown("**Momentum Indicators**")

        if show_rsi:
            current_rsi = rsi.iloc[-1]
            if current_rsi > 70:
                st.warning(f"RSI: {current_rsi:.1f} (Overbought)")
            elif current_rsi < 30:
                st.info(f"RSI: {current_rsi:.1f} (Oversold)")
            else:
                st.metric("RSI", f"{current_rsi:.1f}")

        if show_macd:
            current_histogram = histogram.iloc[-1]
            if current_histogram > 0:
                st.success(f"MACD: Bullish ({current_histogram:.2f})")
            else:
                st.error(f"MACD: Bearish ({current_histogram:.2f})")

    with col3:
        st.markdown("**Volatility Indicators**")

        if show_bollinger:
            current_price = filtered_data["close"].iloc[-1]
            upper = upper_band.iloc[-1]
            lower = lower_band.iloc[-1]
            bb_position = (current_price - lower) / (upper - lower) * 100

            st.metric("BB Position", f"{bb_position:.1f}%")

            if bb_position > 80:
                st.warning("Near Upper Band")
            elif bb_position < 20:
                st.info("Near Lower Band")

        # ATR (Average True Range)
        high_low = filtered_data["high"] - filtered_data["low"]
        high_close = abs(filtered_data["high"] - filtered_data["close"].shift())
        low_close = abs(filtered_data["low"] - filtered_data["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]

        st.metric("ATR (14)", f"${atr:.2f}")

else:
    st.info("Select a symbol in the sidebar to begin")
