# Quick Start - Z: Drive Data Collection

## Initial Setup - Download Your First Data

### Option 1: Starter Universe (Recommended for Beginners)
```powershell
# Download 10 popular symbols with daily data since 2020
.\.venv\Scripts\python.exe scripts\batch_download.py `
    --universe starter `
    --start 2020-01-01 `
    --data-dir "Z:\market_data" `
    --show-inventory
```

**Result**: ~50 MB on Z: drive, ready for backtesting

### Option 2: Custom Symbol List
```powershell
# Download specific symbols
.\.venv\Scripts\python.exe scripts\download_data.py AAPL MSFT GOOGL `
    --start 2020-01-01 `
    --data-dir "Z:\market_data" `
    --show-inventory
```

### Option 3: From File
```powershell
# Use predefined universe file
.\.venv\Scripts\python.exe scripts\batch_download.py `
    --universe-file universes\top30.txt `
    --start 2020-01-01 `
    --data-dir "Z:\market_data"
```

## Your Z: Drive Structure

After downloading, your Z: drive will look like this:

```
Z:\market_data\
├── .metadata\           # Metadata (JSON files tracking each dataset)
├── 1Day\               # Daily bars
│   ├── AAPL.parquet
│   ├── MSFT.parquet
│   └── ...
├── 1Hour\              # Hourly bars (when you add them)
└── 15Min\              # 15-minute bars (when you add them)
```

## Check What You Have

```powershell
# Quick inventory check
.\.venv\Scripts\python.exe -c "from src.trading.data import DataStore; import pandas as pd; store = DataStore('Z:\\market_data'); inv = store.get_inventory(); print(inv) if not inv.empty else print('No data yet')"
```

## Run Your First Backtest

```powershell
# Backtest AAPL with moving average crossover
.\.venv\Scripts\python.exe scripts\run_backtest.py AAPL `
    --strategy MovingAverageCrossover `
    --fast-window 10 `
    --slow-window 30 `
    --data-dir "Z:\market_data" `
    --plot
```

## Predefined Universes Available

Use with `--universe` flag:

| Universe | Symbols | Description |
|----------|---------|-------------|
| `starter` | 10 | SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM |
| `indices` | 6 | Major market indices (SPY, QQQ, IWM, etc.) |
| `sectors` | 11 | All sector ETFs (XLK, XLF, XLE, etc.) |
| `mega_cap` | 8 | Largest US companies |
| `tech` | 10 | Technology stocks |
| `finance` | 10 | Financial sector stocks |
| `energy` | 10 | Energy sector stocks |
| `healthcare` | 10 | Healthcare sector stocks |

## Add Different Timeframes

### Add Hourly Data
```powershell
# Add hourly data for day trading strategies
.\.venv\Scripts\python.exe scripts\download_data.py AAPL MSFT `
    --start 2023-01-01 `
    --timeframe 1Hour `
    --data-dir "Z:\market_data"
```

### Add 15-Minute Data
```powershell
# Add 15-min data for scalping strategies
.\.venv\Scripts\python.exe scripts\download_data.py SPY QQQ `
    --start 2024-01-01 `
    --timeframe 15Min `
    --data-dir "Z:\market_data"
```

## Tips for Z: Drive Management

1. **Space Planning**:
   - Daily data: ~1 MB per symbol per 5 years
   - Hourly data: ~20 MB per symbol per 5 years
   - 15-min data: ~80 MB per symbol per 5 years

2. **Start Small**: Begin with daily data only, add intraday later

3. **Regular Updates**: Run weekly to get latest data
   ```powershell
   .\.venv\Scripts\python.exe scripts\batch_download.py `
       --universe starter `
       --start (Get-Date).AddMonths(-1).ToString("yyyy-MM-dd") `
       --data-dir "Z:\market_data"
   ```

4. **Skip Existing Data**: Use `--skip-existing` to avoid re-downloading
   ```powershell
   .\.venv\Scripts\python.exe scripts\batch_download.py `
       --universe tech `
       --start 2020-01-01 `
       --data-dir "Z:\market_data" `
       --skip-existing
   ```

## Recommended Collection Strategy

### Week 1: Foundation
```powershell
# Get broad market coverage (indices + top stocks)
.\.venv\Scripts\python.exe scripts\batch_download.py `
    --universe starter `
    --start 2020-01-01 `
    --data-dir "Z:\market_data"
```

### Week 2: Sector Diversification
```powershell
# Add sector ETFs for sector rotation strategies
.\.venv\Scripts\python.exe scripts\batch_download.py `
    --universe sectors `
    --start 2020-01-01 `
    --data-dir "Z:\market_data"
```

### Week 3: Expanded Universe
```powershell
# Add top 30 most liquid stocks
.\.venv\Scripts\python.exe scripts\batch_download.py `
    --universe-file universes\top30.txt `
    --start 2020-01-01 `
    --data-dir "Z:\market_data"
```

### Week 4: Intraday Data
```powershell
# Add hourly data for select symbols
.\.venv\Scripts\python.exe scripts\download_data.py SPY QQQ AAPL MSFT `
    --start 2023-01-01 `
    --timeframe 1Hour `
    --data-dir "Z:\market_data"
```

## Troubleshooting

**Problem**: "Data not found" error when running backtest
**Solution**: Make sure you're using the same `--data-dir` path:
```powershell
.\.venv\Scripts\python.exe scripts\run_backtest.py AAPL --data-dir "Z:\market_data"
```

**Problem**: Rate limit errors
**Solution**: Add delay between downloads:
```powershell
.\.venv\Scripts\python.exe scripts\batch_download.py --delay 1.0 ...
```

**Problem**: Want to see what's stored
**Solution**: Always use `--show-inventory` flag to see summary

## Next Steps

Once you have data on Z: drive:
1. Read [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md) for comprehensive best practices
2. Read [TRADING_README.md](TRADING_README.md) for backtesting guide
3. Experiment with different strategies and parameters
4. Start building your own custom strategies
