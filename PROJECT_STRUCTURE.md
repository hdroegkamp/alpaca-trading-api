# Trading System - Project Structure

Generated: January 2026

## Directory Tree

```
alpaca-trading-api/
│
├── .env                         # API credentials (DO NOT COMMIT)
├── .gitignore                   # Git ignore rules
├── README.md                    # Original setup instructions
├── TRADING_README.md            # Trading system documentation ⭐
├── requirements.txt             # Python dependencies
│
├── .venv/                       # Python virtual environment (ignored)
│
├── data/                        # Historical market data (ignored)
│   ├── AAPL_1Day.parquet       # Example: AAPL daily bars
│   ├── MSFT_1Day.parquet       # Example: MSFT daily bars
│   └── ...
│
├── logs/                        # Log files (ignored)
│   └── *.log
│
├── scripts/                     # Executable scripts
│   ├── check_alpaca.py         # Verify Alpaca connection
│   ├── download_data.py        # Download historical data ⭐
│   ├── run_backtest.py         # Run backtests ⭐
│   └── example_workflow.py     # End-to-end example ⭐
│
├── src/                         # Source code
│   └── trading/                 # Main trading package
│       ├── __init__.py
│       │
│       ├── strategy/            # Trading strategies
│       │   ├── __init__.py
│       │   ├── base.py         # Strategy interface ⭐
│       │   └── examples/
│       │       ├── __init__.py
│       │       └── moving_average.py  # MA crossover & mean reversion ⭐
│       │
│       ├── backtest/            # Backtesting engine
│       │   ├── __init__.py
│       │   ├── vectorized.py   # Vectorized backtest ⭐
│       │   └── metrics.py      # Performance metrics ⭐
│       │
│       ├── data/                # Data management
│       │   ├── __init__.py
│       │   ├── ingest.py       # Alpaca data fetcher ⭐
│       │   └── storage.py      # Parquet storage ⭐
│       │
│       ├── portfolio/           # Position management
│       │   ├── __init__.py
│       │   └── manager.py      # Portfolio manager
│       │
│       ├── risk/                # Risk management
│       │   ├── __init__.py
│       │   └── rules.py        # Risk rules & circuit breakers
│       │
│       └── utils/               # Utilities
│           ├── __init__.py
│           └── logging.py      # Logging setup
│
└── tests/                       # Test suite
    ├── __init__.py
    └── unit/
        ├── __init__.py
        ├── test_strategy.py    # Strategy tests ⭐
        └── test_backtest.py    # Backtest tests ⭐

```

## Quick Reference

### ⭐ Key Files to Understand

1. **Strategy Development**
   - `src/trading/strategy/base.py` - Strategy interface
   - `src/trading/strategy/examples/moving_average.py` - Example strategies

2. **Backtesting**
   - `src/trading/backtest/vectorized.py` - Backtest engine
   - `scripts/run_backtest.py` - CLI runner

3. **Data**
   - `src/trading/data/ingest.py` - Download from Alpaca
   - `scripts/download_data.py` - CLI tool

4. **Documentation**
   - `TRADING_README.md` - Complete system documentation

### Common Workflows

#### Download Data
```powershell
.\.venv\Scripts\python.exe scripts\download_data.py AAPL --start 2020-01-01
```

#### Run Backtest
```powershell
.\.venv\Scripts\python.exe scripts\run_backtest.py AAPL --strategy MovingAverageCrossover
```

#### Run Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

#### Full Example
```powershell
.\.venv\Scripts\python.exe scripts\example_workflow.py
```

### Module Responsibilities

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `strategy` | Define trading logic | `Strategy`, `MovingAverageCrossover`, `MeanReversion` |
| `backtest` | Test strategies on historical data | `VectorizedBacktest`, `PerformanceMetrics` |
| `data` | Fetch and store market data | `AlpacaDataFetcher`, `DataStore` |
| `portfolio` | Manage positions and sizing | `PortfolioManager` |
| `risk` | Enforce risk limits | `RiskManager` |
| `utils` | Helper functions | `setup_logger` |

### Design Principles

1. **Modularity**: Each component is independent and testable
2. **Simplicity**: Start simple, add complexity as needed
3. **Testability**: Unit tests for core logic
4. **Extensibility**: Easy to add new strategies
5. **Performance**: Vectorized operations for speed
6. **Safety**: Risk management and circuit breakers built-in

### Phase 1 (Current): Research & Prototyping
- [x] Vectorized backtesting
- [x] Basic strategies (MA, mean reversion)
- [x] Data download and storage
- [x] Performance metrics
- [x] Unit tests

### Phase 2 (Future): Realistic Testing
- ⬜ Event-driven backtest
- ⬜ Slippage models
- ⬜ Walk-forward validation
- ⬜ Parameter optimization

### Phase 3 (Future): Production
- ⬜ Live execution
- ⬜ Real-time monitoring
- ⬜ Alerting
- ⬜ CI/CD pipeline

---

**Next Steps**: Read `TRADING_README.md` for detailed documentation.
