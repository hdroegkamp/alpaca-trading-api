# Z: Drive Setup Complete!

## What's New

Your trading system now supports **organized data storage** on your Z: drive with:

- [x] **Automatic directory organization** by timeframe (1Day/, 1Hour/, 15Min/)
- [x] **Metadata tracking** for all downloaded datasets
- [x] **Batch download** capabilities with predefined universes
- [x] **Inventory management** to see what data you have
- [x] **Comprehensive guides** for data collection best practices

## Quick Commands

### Download Your First Data
```powershell
# Start with the "starter" universe (10 popular symbols)
.\.venv\Scripts\python.exe scripts\batch_download.py `
    --universe starter `
    --start 2020-01-01 `
    --data-dir "Z:\market_data" `
    --show-inventory
```

### View Your Data Inventory
```powershell
.\.venv\Scripts\python.exe scripts\view_inventory.py --data-dir "Z:\market_data"
```

### Run a Backtest
```powershell
.\.venv\Scripts\python.exe scripts\run_backtest.py AAPL `
    --strategy MovingAverageCrossover `
    --fast-window 10 `
    --slow-window 30 `
    --data-dir "Z:\market_data" `
    --plot
```

## Your Directory Structure on Z:

```
Z:\market_data\
├── .metadata\              # JSON metadata files
│   ├── AAPL_1Day.json     # Tracks: rows, dates, file size, last updated
│   └── MSFT_1Day.json
├── 1Day\                   # Daily OHLCV data
│   ├── AAPL.parquet       # Compressed parquet format
│   ├── MSFT.parquet
│   └── ...
├── 1Hour\                  # Hourly data (when you add it)
└── 15Min\                  # 15-minute data (when you add it)
```

## New Scripts Available

| Script | Purpose |
|--------|---------|
| `batch_download.py` | Download multiple symbols at once from predefined universes or files |
| `view_inventory.py` | View what data you have stored with detailed statistics |
| `download_data.py` | *(Updated)* Now supports `--show-inventory` and organized storage |
| `run_backtest.py` | *(Updated)* Now works with organized Z: drive structure |

## Predefined Universes

Use with `--universe` flag in batch_download.py:

- **starter**: 10 popular symbols (SPY, QQQ, AAPL, MSFT, etc.)
- **indices**: 6 major market indices
- **sectors**: 11 sector ETFs (all SPDR sectors)
- **mega_cap**: 8 largest US companies
- **tech**: 10 technology stocks
- **finance**: 10 financial stocks
- **energy**: 10 energy stocks
- **healthcare**: 10 healthcare stocks

## Universe Files

Pre-created files in `universes/` directory:

- `top30.txt`: Top 30 most liquid US stocks
- `etf_diversified.txt`: ETFs for broad market coverage
- `tech_focus.txt`: Technology-focused watchlist

You can create your own! Just make a text file with one symbol per line.

## Documentation

**QUICKSTART_Z_DRIVE.md** - Quick reference for Z: drive usage
**DATA_COLLECTION_GUIDE.md** - Comprehensive best practices guide

Key topics covered:
- Storage space planning
- Best timeframes for different strategies
- Symbol selection strategies
- Data quality considerations
- Update schedules and workflows
- Troubleshooting common issues

## Storage Estimates

Plan your Z: drive space:

| Data Type | Storage per Symbol | Recommended For |
|-----------|-------------------|-----------------|
| 5 years daily | ~1 MB | Position/swing trading |
| 3 years hourly | ~20 MB | Day trading |
| 2 years 15-min | ~80 MB | Scalping strategies |

**Example**: 
- 100 symbols × 5 years daily = ~100 MB
- 50 symbols × 3 years hourly = ~1 GB
- 20 symbols × 2 years 15-min = ~1.6 GB

## Best Practices Summary

### For Backtesting Daily Strategies:
1. Download 5-10 years of daily data
2. Include major crashes (2008, 2020) for robustness testing
3. Start with 20-50 liquid symbols
4. Storage: ~50-100 MB

### For Intraday Strategies:
1. Download 2-3 years of hourly or 15-min data
2. Focus on high-volume symbols (SPY, QQQ, mega caps)
3. Start with 10-20 symbols
4. Storage: ~500 MB - 2 GB

### Data Collection Workflow:
1. **Week 1**: Download core indices and mega caps (starter universe)
2. **Week 2**: Add sector ETFs for diversification
3. **Week 3**: Expand to broader stock universe
4. **Week 4**: Add intraday data for active trading strategies

### Maintenance:
- **Weekly**: Update daily data for recent trading days
- **Monthly**: Deep refresh of recent data (last 6 months)
- **Quarterly**: Expand universe with new symbols

## Technical Improvements

### DataStore Class Enhancements:
- `organize_by_timeframe` parameter for structured storage
- `get_metadata()` to check dataset details
- `get_inventory()` returns pandas DataFrame with all data info
- `list_symbols()` now accepts timeframe filter
- Automatic metadata saving (rows, dates, file size, last updated)

### Benefits:
- **Faster loading**: Organized by timeframe
- **Better tracking**: Know exactly what data you have
- **Space efficient**: Gzip-compressed parquet files
- **Easy management**: Clear directory structure

## Example Workflow

```powershell
# 1. Download initial data
.\.venv\Scripts\python.exe scripts\batch_download.py --universe starter --start 2020-01-01 --data-dir "Z:\market_data"

# 2. Check what you have
.\.venv\Scripts\python.exe scripts\view_inventory.py --data-dir "Z:\market_data"

# 3. Run backtests
.\.venv\Scripts\python.exe scripts\run_backtest.py AAPL --data-dir "Z:\market_data" --plot

# 4. Add more data as needed
.\.venv\Scripts\python.exe scripts\batch_download.py --universe sectors --data-dir "Z:\market_data" --skip-existing

# 5. Add intraday data for specific symbols
.\.venv\Scripts\python.exe scripts\download_data.py SPY QQQ --timeframe 1Hour --start 2023-01-01 --data-dir "Z:\market_data"
```

## Testing Verification

All tests still passing:
- [x] 12/12 unit tests passed
- [x] Type annotations all clean (no Pylance errors)
- [x] Backward compatible with existing code

## Need Help?

1. **Quick reference**: See QUICKSTART_Z_DRIVE.md
2. **Best practices**: See DATA_COLLECTION_GUIDE.md
3. **System overview**: See TRADING_README.md
4. **Project structure**: See PROJECT_STRUCTURE.md

## Ready to Start!

Your system is now ready for serious backtesting with organized, scalable data storage. Start by downloading the starter universe and running your first backtest!

```powershell
# One command to get started:
.\.venv\Scripts\python.exe scripts\batch_download.py --universe starter --start 2020-01-01 --data-dir "Z:\market_data" --show-inventory
```

Happy Trading!
