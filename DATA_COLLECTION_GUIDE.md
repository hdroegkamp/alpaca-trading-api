# Data Collection Guide for Backtesting

## Overview

This guide covers best practices for collecting and organizing historical market data for algorithmic trading backtesting.

## Organized Storage Structure

Your data is now organized by timeframe for better management:

```
Z:\market_data\
├── .metadata\              # Metadata JSON files
│   ├── AAPL_1D.json
│   └── MSFT_1D.json
├── 1Day\                   # Daily data
│   ├── AAPL.parquet
│   ├── MSFT.parquet
│   └── GOOGL.parquet
├── 1Hour\                  # Hourly data
│   ├── AAPL.parquet
│   └── SPY.parquet
└── 15Min\                  # 15-minute data
    └── AAPL.parquet
```

## Quick Start - Download to Z: Drive

### Basic Usage

```powershell
# Download daily data for multiple symbols
.\.venv\Scripts\python.exe scripts\download_data.py AAPL MSFT GOOGL `
    --start 2020-01-01 `
    --data-dir "Z:\market_data" `
    --show-inventory

# Download hourly data
.\.venv\Scripts\python.exe scripts\download_data.py SPY QQQ `
    --start 2023-01-01 `
    --timeframe 1Hour `
    --data-dir "Z:\market_data"
```

### Batch Download Script

Use the provided batch download script for downloading entire lists:

```powershell
.\.venv\Scripts\python.exe scripts\batch_download.py --data-dir "Z:\market_data"
```

## Best Practices for Data Collection

### 1. **Choose Appropriate Timeframes**

| Timeframe | Best For | Storage Size | Lookback Period |
|-----------|----------|--------------|-----------------|
| 1Day | Position trading, swing strategies | Small (~1MB/symbol/5yr) | 5-10 years |
| 1Hour | Day trading, intraday strategies | Medium (~20MB/symbol/5yr) | 2-3 years |
| 15Min | Scalping, high-frequency strategies | Large (~80MB/symbol/5yr) | 1 year |
| 5Min | Ultra-short-term strategies | Very Large (~240MB/symbol/5yr) | 6 months |

### 2. **Data Quality Considerations**

- **Start Date**: Choose based on strategy requirements
  - Long-term strategies: 10+ years (includes multiple market cycles)
  - Medium-term: 5 years (includes bull/bear markets)
  - Short-term: 2-3 years (recent market regime)

- **Market Coverage**: 
  - Include market crash periods (2008, 2020) for robustness testing
  - Include bull markets (2010-2019) for growth validation
  - Include sideways markets (2015-2016) for range-bound testing

### 3. **Symbol Selection Strategy**

#### Core Holdings (Must Have)
```python
# Major indices
INDICES = ["SPY", "QQQ", "IWM", "DIA"]

# Sector ETFs for diversification
SECTORS = [
    "XLK",  # Technology
    "XLF",  # Financials
    "XLE",  # Energy
    "XLV",  # Healthcare
    "XLI",  # Industrials
    "XLP",  # Consumer Staples
    "XLY",  # Consumer Discretionary
    "XLU",  # Utilities
    "XLB",  # Materials
]

# Popular stocks
POPULAR = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
```

#### Universe Expansion
- Start with 20-50 symbols for initial testing
- Expand to 100-200 for production
- Focus on liquid symbols (>$1M daily volume)

### 4. **Data Download Schedule**

#### Initial Download
```powershell
# Download 5 years of daily data for core universe
.\.venv\Scripts\python.exe scripts\batch_download.py `
    --data-dir "Z:\market_data" `
    --start 2020-01-01 `
    --timeframe 1Day
```

#### Regular Updates
- **Daily data**: Update weekly (Alpaca provides up-to-date data)
- **Intraday data**: Update monthly or before major backtests
- **Use incremental updates** to avoid re-downloading entire history

### 5. **Storage Management**

#### Estimate Storage Requirements

```python
# Rough estimates for Z: drive space planning
Daily (1D):
  - 100 symbols × 5 years = ~100 MB
  - 500 symbols × 10 years = ~500 MB

Hourly (1H):
  - 50 symbols × 3 years = ~3 GB
  - 200 symbols × 3 years = ~12 GB

15-Minute (15Min):
  - 20 symbols × 2 years = ~3 GB
  - 100 symbols × 2 years = ~15 GB
```

#### Compression Benefits
- Parquet with gzip compression: **70-80% size reduction**
- Efficient columnar storage for faster loading
- Metadata stored separately for quick inventory checks

### 6. **Data Validation**

After downloading, validate your data:

```powershell
# Check inventory
.\.venv\Scripts\python.exe scripts\download_data.py --data-dir "Z:\market_data" --show-inventory

# Or use Python directly
.\.venv\Scripts\python.exe -c "
from src.trading.data import DataStore
store = DataStore('Z:\\market_data')
print(store.get_inventory())
"
```

### 7. **Universe Files for Batch Downloads**

Create universe files for different strategies:

#### `universes/sp500.txt`
```
AAPL
MSFT
GOOGL
AMZN
...
```

#### `universes/tech_focus.txt`
```
AAPL
MSFT
NVDA
GOOGL
META
TSLA
AMD
```

## Update Workflow

### Weekly Update Script
```powershell
# Update daily data for all existing symbols
.\.venv\Scripts\python.exe scripts\update_data.py `
    --data-dir "Z:\market_data" `
    --timeframe 1Day
```

### Monthly Deep Refresh
```powershell
# Re-download last 6 months to catch any corrections
.\.venv\Scripts\python.exe scripts\batch_download.py `
    --data-dir "Z:\market_data" `
    --start (Get-Date).AddMonths(-6).ToString("yyyy-MM-dd") `
    --force-refresh
```

## Data Inventory Management

### Check Storage Usage
```powershell
# Python one-liner to see data inventory
.\.venv\Scripts\python.exe -c "
from src.trading.data import DataStore
import pandas as pd
store = DataStore('Z:\\market_data')
inventory = store.get_inventory()
print('\nDATA INVENTORY')
print('='*80)
print(inventory)
print(f'\nTotal files: {len(inventory)}')
print(f'Total storage: {inventory['file_size_mb'].sum():.2f} MB')
print(f'Date range: {inventory['start_date'].min()} to {inventory['end_date'].max()}')
"
```

## Recommended Starting Point

### Beginner Setup (Daily Strategies)
```powershell
# ~150 MB on Z: drive
.\.venv\Scripts\python.exe scripts\download_data.py `
    SPY QQQ AAPL MSFT GOOGL AMZN TSLA NVDA META `
    --start 2019-01-01 `
    --data-dir "Z:\market_data" `
    --show-inventory
```

### Intermediate Setup (Intraday + Daily)
```powershell
# ~2 GB on Z: drive
# Daily data for 50 symbols (5 years)
.\.venv\Scripts\python.exe scripts\batch_download.py `
    --universe universes/watchlist_50.txt `
    --start 2020-01-01 `
    --timeframe 1Day `
    --data-dir "Z:\market_data"

# Hourly data for top 10 symbols (2 years)
.\.venv\Scripts\python.exe scripts\download_data.py `
    AAPL MSFT GOOGL AMZN TSLA SPY QQQ `
    --start 2023-01-01 `
    --timeframe 1Hour `
    --data-dir "Z:\market_data"
```

### Advanced Setup (Full Universe)
```powershell
# ~10-20 GB on Z: drive
# Daily data for S&P 500 (10 years)
# Hourly data for watchlist (3 years)
# 15-min data for active trading (1 year)
```

## Troubleshooting

### Common Issues

1. **Rate Limiting**: Alpaca has API rate limits
   - Solution: Add delays between requests in batch downloads
   - Free tier: 200 requests/minute

2. **Missing Data**: Some symbols may have gaps
   - Solution: Use `--force-refresh` to re-download
   - Check symbol listing date (IPO date)

3. **Storage Full**: Z: drive running out of space
   - Solution: Archive old data, reduce timeframes
   - Keep only recent data for intraday

## Next Steps

After collecting data:

1. **Validate Data Quality**: Run data quality checks
2. **Backtest Development**: Start with daily strategies
3. **Walk-Forward Analysis**: Use recent data for out-of-sample testing
4. **Regular Updates**: Set up automated weekly downloads

## Related Scripts

- `scripts/download_data.py` - Single/multi-symbol download
- `scripts/batch_download.py` - Batch download from file
- `scripts/update_data.py` - Update existing data
- `scripts/validate_data.py` - Check data quality
