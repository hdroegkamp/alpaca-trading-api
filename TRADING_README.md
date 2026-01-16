# Algorithmic Trading System

A modular Python framework for backtesting and deploying quantitative trading strategies.

## 🏗️ Architecture

```
src/trading/
├── strategy/          # Trading strategy implementations
│   ├── base.py       # Strategy interface
│   └── examples/     # Example strategies (MA crossover, mean reversion)
├── backtest/         # Backtesting engine
│   ├── vectorized.py # Fast vectorized backtesting
│   └── metrics.py    # Performance metrics (Sharpe, Sortino, etc.)
├── data/             # Data ingestion and storage
│   ├── ingest.py     # Fetch data from Alpaca API
│   └── storage.py    # Parquet-based data store
├── portfolio/        # Position sizing and management
├── risk/             # Risk management rules
└── utils/            # Logging and utilities

scripts/
├── download_data.py  # Download historical data
└── run_backtest.py   # Run backtests

tests/
└── unit/            # Unit tests
```

## 🚀 Quick Start

### 1. Download Historical Data

```powershell
# Download data for AAPL from 2020
.\.venv\Scripts\python.exe scripts\download_data.py AAPL --start 2020-01-01

# Download multiple symbols
.\.venv\Scripts\python.exe scripts\download_data.py AAPL MSFT GOOGL TSLA --start 2020-01-01
```

Data is saved in `data/` directory as compressed parquet files.

### 2. Run a Backtest

```powershell
# Moving Average Crossover strategy
.\.venv\Scripts\python.exe scripts\run_backtest.py AAPL --strategy MovingAverageCrossover --fast-window 10 --slow-window 30 --plot

# Mean Reversion strategy
.\.venv\Scripts\python.exe scripts\run_backtest.py AAPL --strategy MeanReversion --window 20 --num-std 2.0 --plot
```

### 3. Run Tests

```powershell
# Run all tests
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Run specific test file
.\.venv\Scripts\python.exe -m pytest tests/unit/test_strategy.py -v
```

## 📊 Built-in Strategies

### Moving Average Crossover
Goes long when fast MA > slow MA, short when fast MA < slow MA.

**Parameters:**
- `fast_window` (default: 20): Fast moving average period
- `slow_window` (default: 50): Slow moving average period

**Usage:**
```powershell
.\.venv\Scripts\python.exe scripts\run_backtest.py AAPL --strategy MovingAverageCrossover --fast-window 10 --slow-window 30
```

### Mean Reversion
Bollinger Bands mean reversion strategy. Goes long below lower band, short above upper band.

**Parameters:**
- `window` (default: 20): Rolling window for mean/std
- `num_std` (default: 2.0): Number of std devs for bands

**Usage:**
```powershell
.\.venv\Scripts\python.exe scripts\run_backtest.py AAPL --strategy MeanReversion --window 20 --num-std 2.0
```

## 🔧 Creating Custom Strategies

Create a new strategy by subclassing `Strategy`:

```python
# src/trading/strategy/examples/my_strategy.py
from ..base import Strategy
import pandas as pd

class MyStrategy(Strategy):
    def __init__(self, param1=10, param2=20):
        super().__init__(param1=param1, param2=param2)
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals.
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            DataFrame with 'position' column (-1, 0, or 1)
        """
        signals = pd.DataFrame(index=data.index)
        
        # Your strategy logic here
        # Example: simple logic based on params
        signals['position'] = 0.0
        
        # ... compute indicators and signals ...
        
        return signals[['position']]
```

Then register it in `run_backtest.py` to use from command line.

## 📈 Performance Metrics

The backtest engine calculates:

- **Total Return**: Overall return over the period
- **CAGR**: Compound annual growth rate
- **Sharpe Ratio**: Risk-adjusted return (mean/std of returns)
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized standard deviation of returns
- **Win Rate**: Percentage of profitable trades
- **Number of Trades**: Total position changes

## 🧪 Backtesting Design

### Vectorized Backtesting
The current implementation uses **vectorized backtesting** for speed:
- Fast execution using pandas/numpy operations
- Assumes end-of-bar execution
- Simple fill model (market orders filled at bar close)
- Suitable for rapid strategy prototyping and research

**Limitations:**
- Doesn't model intraday execution
- Simplified fill model (no partial fills, slippage is commission-based)
- Lookahead bias possible if not careful

### Future: Event-Driven Backtesting
For more realistic testing, an event-driven engine will be added:
- Tick-by-tick replay of historical data
- Order book modeling
- Realistic fill simulation with slippage
- Latency modeling

## 🔐 Data Storage

Data is stored as compressed parquet files in `data/`:
- Format: `{SYMBOL}_{TIMEFRAME}.parquet`
- Example: `AAPL_1Day.parquet`
- Efficient columnar storage with compression
- Fast read/write for large datasets

List available data:
```python
from src.trading.data import DataStore
store = DataStore()
print(store.list_symbols())
```

## 🎯 Roadmap

### Current (Phase 1 - Research)
- ✅ Vectorized backtesting
- ✅ Basic strategies (MA crossover, mean reversion)
- ✅ Data download and storage
- ✅ Performance metrics
- ✅ Unit tests

### Phase 2 - Realism
- ⬜ Event-driven backtest engine
- ⬜ Slippage models
- ⬜ Multiple timeframes
- ⬜ Walk-forward analysis
- ⬜ Parameter optimization

### Phase 3 - Production
- ⬜ Live order execution adapter
- ⬜ Position tracking
- ⬜ Real-time monitoring
- ⬜ Circuit breakers and risk checks
- ⬜ Alerting and notifications

### Phase 4 - Advanced
- ⬜ Multi-asset portfolios
- ⬜ ML-based strategies
- ⬜ Feature engineering pipeline
- ⬜ Experiment tracking (MLflow)
- ⬜ CI/CD for strategy deployment

## 📚 API Reference

### Strategy Base Class

```python
class Strategy(ABC):
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from OHLCV data.
        
        Args:
            data: DataFrame with columns [open, high, low, close, volume]
                  and DatetimeIndex
        
        Returns:
            DataFrame with same index and 'position' column
            position values: -1 (short), 0 (flat), 1 (long)
        """
```

### Backtest Engine

```python
from src.trading.backtest import VectorizedBacktest

backtest = VectorizedBacktest(
    strategy=my_strategy,
    data=historical_data,
    initial_capital=100000.0,
    commission=0.001  # 0.1% per trade
)

results = backtest.run()
summary = backtest.get_summary()
backtest.plot()
```

### Data Management

```python
from src.trading.data import AlpacaDataFetcher, DataStore

# Fetch data
fetcher = AlpacaDataFetcher()
data = fetcher.fetch_bars('AAPL', start='2020-01-01', end='2023-12-31')

# Store data
store = DataStore(data_dir='data')
store.save('AAPL', data)

# Load data
data = store.load('AAPL')
```

## 🛡️ Risk Management

The `RiskManager` class enforces limits:
- Max position size (% of capital)
- Max daily loss
- Max drawdown
- Circuit breaker triggers

Example:
```python
from src.trading.risk import RiskManager

risk_mgr = RiskManager(
    max_position_size=0.2,    # 20% max per position
    max_daily_loss=0.02,      # 2% max daily loss
    max_drawdown=0.10         # 10% max drawdown
)

allowed, reason = risk_mgr.check_position_size(
    position_value=15000,
    portfolio_value=100000
)
```

## 🔍 Debugging and Logging

Logs are output to console by default. Configure logging:

```python
from src.trading.utils.logging import setup_logger

logger = setup_logger('my_strategy', log_file='logs/strategy.log')
logger.info("Strategy initialized")
```

## 💡 Tips & Best Practices

1. **Start Simple**: Begin with a simple strategy to understand the framework
2. **Test First**: Write unit tests for your strategy logic
3. **Version Control**: Commit your strategies and track experiments
4. **Parameter Sweeps**: Use scripts to test multiple parameter combinations
5. **Walk-Forward**: Validate strategies on out-of-sample data
6. **Slippage**: Always include realistic transaction costs
7. **Data Quality**: Inspect your data for gaps and anomalies
8. **Reproducibility**: Pin package versions and seed random number generators

## 📖 Resources

- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [Quantitative Trading](https://www.quantstart.com/) - QuantStart tutorials

## 🤝 Contributing

This is a personal project, but improvements are welcome:
1. Add new strategies in `src/trading/strategy/examples/`
2. Improve performance metrics in `src/trading/backtest/metrics.py`
3. Add tests for new features
4. Document your changes

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves risk of loss. Always test strategies thoroughly on paper accounts before live trading.

---

**Version**: 0.1.0  
**Last Updated**: January 2026
