# Streamlit Trading Analysis Suite

A comprehensive suite of Streamlit applications for exploring trading data and analyzing algorithmic trading strategies.

## Apps Overview

### 1. Main Dashboard (`main.py`)
Landing page with overview and navigation to all tools.

### 2. Data Explorer
- Interactive price charts with candlesticks
- Volume analysis
- Multi-symbol comparison (normalized returns)
- Statistical summaries
- Export data to CSV

### 3. Strategy Analyzer
- Backtest trading strategies
- Moving Average Crossover strategy
- Mean Reversion (Bollinger Bands) strategy
- Performance metrics (Sharpe, Sortino, drawdowns)
- Equity curve visualization
- Trading signals overlay
- Compare strategy vs Buy & Hold

### 4. Technical Analysis
- Advanced charting with multiple indicators
- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- Bollinger Bands
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Volume analysis
- Technical summary with trend/momentum indicators

### 5. Data Quality Checker
- Data inventory overview
- Date coverage visualization
- Gap detection for missing trading days
- Quality checks (zero prices, extreme moves, OHLC consistency)
- Volume analysis
- Symbol comparison reports

## Installation

Install required packages:

```powershell
.\.venv\Scripts\pip.exe install streamlit plotly
```

Or use the updated requirements.txt:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## Running the Apps

### Start the Main Dashboard

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_apps\main.py
```

This will:
1. Open your default web browser
2. Load the main dashboard at `http://localhost:8501`
3. Navigate between pages using the sidebar

### Direct Access to Specific Pages

You can also run individual pages directly:

```powershell
# Data Explorer
.\.venv\Scripts\streamlit.exe run streamlit_apps\pages\1_Data_Explorer.py

# Strategy Analyzer
.\.venv\Scripts\streamlit.exe run streamlit_apps\pages\2_Strategy_Analyzer.py

# Technical Analysis
.\.venv\Scripts\streamlit.exe run streamlit_apps\pages\3_Technical_Analysis.py

# Data Quality
.\.venv\Scripts\streamlit.exe run streamlit_apps\pages\4_Data_Quality.py
```

## Configuration

### Data Directory

By default, apps use `Z:\market_data` as the data directory. You can change this in the sidebar of the main app, or modify the default in each page.

### Caching

Streamlit caches data automatically for performance:
- **Data Store**: Cached per data directory
- **Inventory**: Cached per data store
- **Symbol Data**: Cached per symbol/timeframe combination

To clear cache: Click the menu (⋮) → "Clear cache" in the Streamlit app.

## Features & Performance Tips

### Efficient Data Loading
- Data is cached using `@st.cache_data` and `@st.cache_resource`
- Only loads data when needed
- Reuses loaded data across interactions

### Interactive Filtering
- Date range selectors for focused analysis
- Symbol search and filtering
- Multi-select for comparisons

### Visualization
- Plotly charts for interactive exploration
- Zoom, pan, and hover for details
- Download charts as images
- Export data to CSV

### Real-time Updates
- Sidebar controls update main content instantly
- Parameter changes immediately reflected in backtests
- No need to rerun entire app

## Example Workflows

### Exploring New Data

1. Navigate to **Data Quality** page
2. Check inventory and coverage
3. Identify any gaps or issues
4. Use **Data Explorer** to visualize specific symbols

### Testing a Strategy

1. Go to **Strategy Analyzer**
2. Select symbol and timeframe
3. Choose strategy and adjust parameters
4. Click "Run Backtest"
5. Analyze equity curve and metrics
6. Compare with Buy & Hold

### Technical Analysis

1. Open **Technical Analysis** page
2. Select symbol and date range
3. Enable desired indicators (MA, RSI, MACD, etc.)
4. Analyze trends and momentum
5. Look for trading signals

### Comparing Symbols

1. Use **Data Explorer** in "Compare Mode"
2. Select multiple symbols
3. View normalized price comparison
4. Compare performance metrics

## Keyboard Shortcuts

- **R**: Rerun the app
- **C**: Clear cache
- **S**: Toggle sidebar

## Customization

### Adding New Indicators

Edit `streamlit_apps/pages/3_Technical_Analysis.py` and add your indicator:

```python
# In sidebar
show_custom = st.checkbox("My Custom Indicator")
if show_custom:
    param = st.slider("Parameter", 1, 100, 20)

# In main section
if show_custom:
    indicator_values = calculate_my_indicator(data, param)
    fig.add_trace(go.Scatter(...))
```

### Adding New Strategies

1. Create strategy in `src/trading/strategy/examples/`
2. Import in `streamlit_apps/pages/2_Strategy_Analyzer.py`
3. Add to strategy selection dropdown
4. Add parameter controls

### Custom Pages

Create new pages in `streamlit_apps/pages/`:

```python
# streamlit_apps/pages/5_My_Custom_Page.py
import streamlit as st

st.set_page_config(page_title="My Page", page_icon="📊", layout="wide")
st.title("My Custom Analysis")
# Your code here
```

Pages are automatically discovered and added to navigation.

## Troubleshooting

### Port Already in Use

If port 8501 is busy:

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_apps\main.py --server.port 8502
```

### Data Not Loading

1. Check data directory path in sidebar
2. Verify data exists: `python scripts/view_inventory.py --data-dir "Z:\market_data"`
3. Clear Streamlit cache (⋮ → Clear cache)

### Slow Performance

1. Reduce date range for analysis
2. Limit number of symbols in comparisons
3. Clear cache periodically
4. Close unused browser tabs

### Memory Issues

For large datasets:
1. Use date range filters
2. Analyze fewer symbols at once
3. Close other applications
4. Restart Streamlit app

## Best Practices

### For Efficiency

1. **Use Caching**: Don't reload data unnecessarily
2. **Filter Early**: Apply date/symbol filters before processing
3. **Lazy Loading**: Only load what you need when you need it
4. **Batch Operations**: Process multiple symbols together when possible

### For Analysis

1. **Start Broad**: Use Data Explorer first to understand data
2. **Check Quality**: Always validate data before analysis
3. **Compare Baselines**: Compare strategies against Buy & Hold
4. **Multiple Timeframes**: Test on different timeframes
5. **Document Findings**: Download charts and export data

### For Development

1. **Use Session State**: Store results to avoid recomputation
2. **Separate Concerns**: Keep data loading and visualization separate
3. **Error Handling**: Always handle data loading errors
4. **Type Hints**: Use type hints for better IDE support

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## Next Steps

After familiarizing yourself with the apps:

1. Create custom indicators in Technical Analysis
2. Implement new strategies in Strategy Analyzer
3. Build custom comparison dashboards
4. Add parameter optimization tools
5. Integrate with live data feeds
