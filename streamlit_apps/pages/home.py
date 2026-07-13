"""Home page - Overview and getting started."""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-weight: 600;
    }

    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stMetric label {
        color: #b0b0b0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.subheader("Configuration")
    data_dir = st.text_input(
        "Data Directory",
        value="data",
        help="Path to your market data storage",
    )
    st.session_state["data_dir"] = data_dir
    st.markdown("---")
    st.caption("Built with Streamlit for algorithmic trading analysis")

# Title
st.title("Trading Analysis Suite")
st.markdown("---")

# Page cards
st.header("Welcome to the Trading Analysis Suite")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Data Explorer")
    st.markdown(
        """
    Visualize historical price data, volume patterns, and basic statistics.

    - Interactive price charts
    - Volume analysis
    - Statistical summaries
    - Multi-symbol comparison
    """
    )

with col2:
    st.subheader("Strategy Analyzer")
    st.markdown(
        """
    Test trading strategies and analyze performance metrics.

    - Run backtests
    - Parameter optimization
    - Performance metrics
    - Equity curves
    """
    )

with col3:
    st.subheader("Technical Analysis")
    st.markdown(
        """
    Advanced charting and technical indicators.

    - Moving averages
    - Bollinger Bands
    - RSI, MACD
    - Custom indicators
    """
    )

col4, col5 = st.columns(2)

with col4:
    st.subheader("Data Quality")
    st.markdown(
        """
    Verify data completeness and identify potential issues.

    - Gap detection
    - Outlier analysis
    - Coverage summaries
    - Symbol comparisons
    """
    )

with col5:
    st.subheader("ML Analysis")
    st.markdown(
        """
    Train machine learning models and compare strategies.

    - LSTM price prediction
    - Random Forest signals
    - Feature importance
    - Strategy backtesting
    """
    )

st.markdown("---")

# Getting started section
st.header("Getting Started")

st.markdown(
    """
### Quick Start Guide

1. **Configure Data Directory**: Set your data directory in the sidebar (default: `data`)

2. **Explore Your Data**: Navigate to **Data Explorer** to visualize and analyze your downloaded data

3. **Analyze Strategies**: Use **Strategy Analyzer** to test different trading strategies

4. **Check Data Quality**: Verify your data completeness and identify any gaps

5. **Train ML Models**: Use **ML Analysis** to train LSTM or Random Forest models and compare against traditional strategies

### Tips for Efficient Use

- **Cache Results**: Streamlit caches data and computations automatically
- **Use Filters**: Apply date ranges and symbol filters to speed up analysis
- **Save Parameters**: Bookmark URLs with parameters for quick access
- **Download Results**: Export charts and tables for reports
"""
)

st.markdown("---")

# System information
with st.expander("System Information"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Data Directory", data_dir)

    with col2:
        from src.trading.data import DataStore

        try:
            store = DataStore(data_dir=data_dir, organize_by_timeframe=True)
            inventory = store.get_inventory()
            symbol_count = len(inventory) if not inventory.empty else 0
            st.metric("Datasets Available", symbol_count)
        except Exception as e:
            st.metric("Datasets Available", "Error")
            st.error(f"Could not load data: {e}")

    with col3:
        try:
            if not inventory.empty:
                total_size = inventory["file_size_mb"].sum()
                st.metric("Total Storage", f"{total_size:.1f} MB")
            else:
                st.metric("Total Storage", "0 MB")
        except Exception:
            st.metric("Total Storage", "N/A")

st.markdown("---")
st.caption(
    "Trading Analysis Suite | Built for algorithmic trading research and development"
)
