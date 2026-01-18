"""Data Quality Checker - Verify data completeness and identify issues."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.data import DataStore

st.set_page_config(page_title="Data Quality", page_icon="📈", layout="wide")

st.title("Data Quality Checker")
st.markdown("Verify data completeness and identify potential issues")
st.markdown("---")

# Get data directory
data_dir = st.session_state.get("data_dir", "Z:\\market_data")


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

# Overall statistics
st.header("Data Inventory Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Symbols", inventory["symbol"].nunique())

with col2:
    st.metric("Total Datasets", len(inventory))

with col3:
    st.metric("Total Bars", f"{inventory['rows'].sum():,}")

with col4:
    st.metric("Total Storage", f"{inventory['file_size_mb'].sum():.1f} MB")

st.markdown("---")

# Inventory by timeframe
st.subheader("Data by Timeframe")

timeframe_summary = (
    inventory.groupby("timeframe")
    .agg(
        {
            "symbol": "count",
            "rows": ["sum", "mean"],
            "file_size_mb": "sum",
            "start_date": "min",
            "end_date": "max",
        }
    )
    .round(2)
)

timeframe_summary.columns = [
    "Symbols",
    "Total Bars",
    "Avg Bars/Symbol",
    "Storage (MB)",
    "Earliest Date",
    "Latest Date",
]

st.dataframe(timeframe_summary, use_container_width=True)

st.markdown("---")

# Date range analysis
st.subheader("Date Range Analysis")

# Select timeframe
selected_timeframe = st.selectbox(
    "Select Timeframe", sorted(inventory["timeframe"].unique())
)

timeframe_data = inventory[inventory["timeframe"] == selected_timeframe].copy()

# Date coverage visualization
fig = go.Figure()

for idx, row in timeframe_data.iterrows():
    fig.add_trace(
        go.Scatter(
            x=[row["start_date"], row["end_date"]],
            y=[row["symbol"], row["symbol"]],
            mode="lines+markers",
            name=row["symbol"],
            line=dict(width=10),
            showlegend=False,
            hovertemplate=f"<b>{row['symbol']}</b><br>"
            + f"Start: {row['start_date']}<br>"
            + f"End: {row['end_date']}<br>"
            + f"Bars: {row['rows']:,}<extra></extra>",
        )
    )

fig.update_layout(
    title=f"Date Coverage by Symbol ({selected_timeframe})",
    xaxis_title="Date",
    yaxis_title="Symbol",
    height=max(400, len(timeframe_data) * 20),
    hovermode="closest",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Data gaps detection
st.subheader("Data Gap Detection")

gap_symbol = st.selectbox(
    "Select Symbol to Check for Gaps", sorted(timeframe_data["symbol"].unique())
)

if gap_symbol:
    data = load_symbol_data(store, gap_symbol, selected_timeframe)

    if data is not None and len(data) > 0:
        # Calculate expected business days (for daily data)
        if selected_timeframe in ["1Day", "1D"]:
            # Get date range
            date_range = pd.date_range(
                start=data.index[0], end=data.index[-1], freq="B"  # Business days
            )

            # Find missing dates
            existing_dates = set(data.index.date)
            expected_dates = set(date_range.date)
            missing_dates = sorted(expected_dates - existing_dates)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Expected Business Days", len(expected_dates))

            with col2:
                st.metric("Actual Days", len(existing_dates))

            with col3:
                coverage = len(existing_dates) / len(expected_dates) * 100
                st.metric("Coverage", f"{coverage:.1f}%")

            if missing_dates:
                st.warning(f"Found {len(missing_dates)} missing business days")

                with st.expander(f"View Missing Dates ({len(missing_dates)})"):
                    # Group consecutive missing dates
                    gaps = []
                    if missing_dates:
                        gap_start = missing_dates[0]
                        gap_end = missing_dates[0]

                        for i in range(1, len(missing_dates)):
                            if (missing_dates[i] - gap_end).days <= 3:  # Allow weekends
                                gap_end = missing_dates[i]
                            else:
                                gaps.append(
                                    (gap_start, gap_end, (gap_end - gap_start).days + 1)
                                )
                                gap_start = missing_dates[i]
                                gap_end = missing_dates[i]

                        gaps.append(
                            (gap_start, gap_end, (gap_end - gap_start).days + 1)
                        )

                    gaps_df = pd.DataFrame(gaps, columns=["Start", "End", "Days"])
                    st.dataframe(gaps_df, use_container_width=True)
            else:
                st.success("No missing business days detected!")

        else:
            st.info("Gap detection currently only available for daily data")

        st.markdown("---")

        # Data quality checks
        st.subheader("Data Quality Checks")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Price Data Quality**")

            # Check for zero/negative prices
            zero_prices = (data["close"] <= 0).sum()
            if zero_prices > 0:
                st.error(f"Found {zero_prices} bars with zero/negative close prices")
            else:
                st.success("No zero/negative prices detected")

            # Check for extreme price changes
            returns = data["close"].pct_change()
            extreme_moves = (abs(returns) > 0.20).sum()  # >20% daily moves
            if extreme_moves > 0:
                st.warning(f"Found {extreme_moves} bars with >20% daily moves")

                with st.expander("View Extreme Moves"):
                    extreme_data = data[abs(returns) > 0.20].copy()
                    extreme_data["return"] = returns[abs(returns) > 0.20] * 100
                    st.dataframe(
                        extreme_data[
                            ["open", "high", "low", "close", "return"]
                        ].sort_index(ascending=False),
                        use_container_width=True,
                    )
            else:
                st.success("No extreme price moves detected")

            # Check for duplicate dates
            duplicates = data.index.duplicated().sum()
            if duplicates > 0:
                st.error(f"Found {duplicates} duplicate timestamps")
            else:
                st.success("No duplicate timestamps detected")

        with col2:
            st.markdown("**Volume Data Quality**")

            # Check for zero volume
            zero_volume = (data["volume"] == 0).sum()
            if zero_volume > 0:
                st.warning(f"Found {zero_volume} bars with zero volume")
            else:
                st.success("No zero volume bars detected")

            # Check for unusually high volume
            avg_volume = data["volume"].mean()
            std_volume = data["volume"].std()
            high_volume = (data["volume"] > avg_volume + 5 * std_volume).sum()

            if high_volume > 0:
                st.info(
                    f"Found {high_volume} bars with unusually high volume (>5 std dev)"
                )
            else:
                st.success("No unusual volume spikes detected")

            # Check OHLC consistency
            invalid_ohlc = 0
            invalid_ohlc += (data["high"] < data["low"]).sum()
            invalid_ohlc += (data["high"] < data["open"]).sum()
            invalid_ohlc += (data["high"] < data["close"]).sum()
            invalid_ohlc += (data["low"] > data["open"]).sum()
            invalid_ohlc += (data["low"] > data["close"]).sum()

            if invalid_ohlc > 0:
                st.error(f"Found {invalid_ohlc} bars with invalid OHLC relationships")
            else:
                st.success("OHLC data is consistent")

        # Statistical summary
        st.markdown("---")
        st.subheader("Statistical Summary")

        summary_stats = data[["open", "high", "low", "close", "volume"]].describe()
        st.dataframe(summary_stats, use_container_width=True)

st.markdown("---")

# Comparison of symbols
st.subheader("Symbol Comparison")

symbols_to_compare = st.multiselect(
    "Select symbols to compare data quality",
    sorted(timeframe_data["symbol"].unique()),
    default=sorted(timeframe_data["symbol"].unique())[:5],
)

if symbols_to_compare:
    comparison_data = []

    for symbol in symbols_to_compare:
        data = load_symbol_data(store, symbol, selected_timeframe)

        if data is not None:
            returns = data["close"].pct_change()

            comparison_data.append(
                {
                    "Symbol": symbol,
                    "Bars": len(data),
                    "Start": data.index[0].strftime("%Y-%m-%d"),
                    "End": data.index[-1].strftime("%Y-%m-%d"),
                    "Avg Price": f"${data['close'].mean():.2f}",
                    "Avg Volume": f"{data['volume'].mean():,.0f}",
                    "Volatility": f"{returns.std()*100:.2f}%",
                    "Zero Volume": (data["volume"] == 0).sum(),
                    "Duplicates": data.index.duplicated().sum(),
                }
            )

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # Download comparison report
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        label="Download Comparison Report",
        data=csv,
        file_name=f"data_quality_report_{selected_timeframe}.csv",
        mime="text/csv",
    )
