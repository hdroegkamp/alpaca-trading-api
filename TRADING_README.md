# Algorithmic Trading System

A modular Python framework for backtesting and deploying quantitative trading strategies.

## Architecture

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

## Quick Start

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

### 4. Live Paper Trading

See the dedicated **Live Paper Trading** section below.

## Built-in Strategies

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

## Performance Metrics

The backtest engine calculates:

- **Total Return**: Overall return over the period
- **CAGR**: Compound annual growth rate
- **Sharpe Ratio**: Risk-adjusted return (mean/std of returns)
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized standard deviation of returns
- **Win Rate**: Percentage of profitable trades
- **Number of Trades**: Total position changes

## Backtesting Design

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

## Data Storage

Data is stored as compressed parquet files in `data/`:
- Format: `{TIMEFRAME}/{SYMBOL}.parquet`, with metadata in `.metadata/{SYMBOL}_{TIMEFRAME}.json`
- Example: `1Day/AAPL.parquet`
- Prices are split/dividend-adjusted (see **Keep the archive current** below)
- Efficient columnar storage with compression
- Fast read/write for large datasets

List available data:
```python
from src.trading.data import DataStore
store = DataStore()
print(store.list_symbols())
```

## Roadmap

### Current (Phase 1 - Research)
- [x] Vectorized backtesting
- [x] Basic strategies (MA crossover, mean reversion)
- [x] Data download and storage
- [x] Performance metrics
- [x] Unit tests

### Phase 2 - Realism
- ⬜ Event-driven backtest engine
- ⬜ Slippage models
- ⬜ Multiple timeframes
- ⬜ Walk-forward analysis
- ⬜ Parameter optimization

### Phase 3 - Production
- [x] Live order execution adapter (`src/trading/execution/broker.py`)
- [x] Position tracking (`data/live/equity_log.csv`, `orders_log.csv`)
- [x] Real-time monitoring (Live Performance dashboard page)
- [x] Circuit breakers and risk checks (`RiskManager` reused per-order, market-open gate)
- ⬜ Alerting and notifications (a weekly Claude Code review agent summarizes
  performance — see below — but no threshold-based alerting yet)

### Phase 4 - Advanced
- ⬜ Multi-asset portfolios
- ⬜ ML-based strategies
- ⬜ Feature engineering pipeline
- ⬜ Experiment tracking (MLflow)
- ⬜ CI/CD for strategy deployment

## API Reference

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

## Risk Management

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

## Live Paper Trading

Turns the pooled cross-sectional model into a real (paper-account) long/short book via
`src/trading/execution/`. Nothing here is auto-scheduled — you decide when it runs.

### One-time setup

```powershell
copy .env.example .env
# then fill in APCA_API_KEY_ID / APCA_API_SECRET_KEY from
# https://app.alpaca.markets/paper/dashboard/overview
.\.venv\Scripts\python.exe scripts\check_alpaca.py   # confirms connectivity
```

### Keep the archive current (before every retrain)

```powershell
.\.venv\Scripts\python.exe scripts\update_archive.py
.\.venv\Scripts\python.exe scripts\check_data_quality.py
```

All bars are stored **split- and dividend-adjusted** (`adjustment="all"`, recorded in each
file's `data/.metadata/*.json`). Raw bars turn every split into a fake one-day crash — NVDA's
10:1 split reads as −90% — which corrupts returns, features and labels.

An adjusted history is rewritten whenever a split or dividend happens, so plain appending is
unsafe. `update_archive.py` re-fetches a week of overlap with each update: if the overlapping
closes still match it appends; if not (or the file isn't adjusted yet) it re-downloads that
symbol's full history. Re-running after an interruption resumes where it stopped.

`check_data_quality.py` exits non-zero if any symbol is unadjusted and lists one-day moves
above 50% (genuine events or bad prints — worth a look either way). `train_pooled_model.py`
refuses unadjusted data unless `--force`; `evaluate_walk_forward.py` warns.

### Train a model

```powershell
.\.venv\Scripts\python.exe scripts\train_pooled_model.py --universe starter
```

Trains against the full `data/` archive and saves to `data/live/pooled_model_*.pkl`. Re-run
this periodically (e.g. weekly) — the daily rebalance below only loads and scores, it never
retrains.

Before saving, two gates must pass (both backtest with `PortfolioBacktest`, net of
`--commission`, default 0.001):

1. **Method gate (walk-forward).** Trains `--wf-splits` (default 10) expanding-window fold
   models and stitches their out-of-sample books into one ~10-year curve. Its Sharpe must
   clear `--wf-min-sharpe` (default 0.5). This is the gate that certifies an edge — over
   ~2,400 OOS days the Sharpe's standard error is ~0.3.
2. **Artifact gate (holdout).** Holds out the most recent `--holdout-days` (default 60)
   dates, trains a candidate with the production hyperparameters on the rest, and requires
   holdout Sharpe > `--min-sharpe` (default 0.0). This only catches an obviously broken
   retrain — a 60-day Sharpe has a standard error of ~2, so it cannot tell skill from luck.

If either fails, the model is **not saved** and the command exits non-zero; override with
`--force` if you've reviewed why and want to save anyway. Both gates' metrics are stashed in
the saved `_metadata.pkl` alongside the scaler/feature columns.

Both gates train on the labeled panel but backtest on the **full** panel (every symbol, every
date). The labeled panel drops each date's middle band *by forward return*, so backtesting on
it picks the book from hindsight winners/losers — see `src/trading/ml/promotion.py`.

To compare method changes (universe, label horizon, rebalance cadence) without touching the
live model, use the read-only evaluation script — fix the configurations and pass bar before
running, rather than iterating until something clears the gate:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_walk_forward.py --universe starter
.\.venv\Scripts\python.exe scripts\evaluate_walk_forward.py --universe universes\top30.txt --forward-periods 5 --rebalance-every 5
```

### Dry run (default — computes and logs intended trades, submits nothing)

```powershell
.\.venv\Scripts\python.exe scripts\run_live_rebalance.py --universe starter
```

### Go live

```powershell
.\.venv\Scripts\python.exe scripts\run_live_rebalance.py --universe starter --live
```

Do this manually the first several times and watch the output before automating anything —
that burn-in period is the point. `--capital-fraction` (default 0.5) caps how much of account
equity gets deployed; `--max-position-pct` (default 0.2) feeds `RiskManager` as a per-position
cap; both are worth sanity-checking on early runs.

### Circuit breaker (daily loss / drawdown)

Every run checks today's equity against `--max-daily-loss-pct` (default 0.02, i.e. 2%) and
`--max-drawdown-pct` (default 0.10, i.e. 10% from peak equity, read from `data/live/equity_log.csv`).
If either is breached, the run halts **before submitting any orders**, logs the reason, appends
a `mode=HALTED` row to `equity_log.csv`, exits non-zero, and — critically — persists the trip to
`data/live/circuit_breaker.json` so **every subsequent run** (including scheduled ones) also halts
immediately, without doing any scoring or trading, until a human reviews what happened and runs:

```powershell
.\.venv\Scripts\python.exe scripts\run_live_rebalance.py --reset-circuit-breaker
```

After a reset, drawdown is measured from the peak equity logged *since that reset*, so the
pre-halt peak doesn't re-trip the breaker on the very next run.

There is no automatic reset. If `scripts\run_live_rebalance.bat` ever exits non-zero, it appends
an explicit `HALTED` marker to `logs\live_rebalance_task.log` so this isn't buried in scrollback.

### Track performance

```powershell
streamlit run streamlit_apps\main.py
```

Open the **Live Performance** page — reads `data/live/equity_log.csv` and `orders_log.csv`
(both git-ignored, same as the rest of `data/`), which every run appends to regardless of
dry-run or `--live`.

### Automate the daily run (only once you're comfortable with manual runs)

`scripts/run_live_rebalance.bat` wraps the `--live` command with the right working directory
and redirects output to `logs\live_rebalance_task.log`. Register it with Windows Task
Scheduler to run each weekday shortly after market open (adjust `/st` for your machine's local
time relative to 9:30 AM ET):

```powershell
schtasks /create /tn "AlpacaLiveRebalance" /tr "C:\Users\hdroe\OneDrive\Documents\Coursera\alpaca-trading-api\scripts\run_live_rebalance.bat" /sc WEEKLY /d MON,TUE,WED,THU,FRI /st 09:35
```

Remove it later with `schtasks /delete /tn "AlpacaLiveRebalance" /f`.

### Weekly performance review (optional)

`scripts/weekly_review.bat` runs a read-only, headless Claude Code prompt over the two log
files and appends a plain-English summary to `logs\weekly_review.log` — evaluation, not
execution, so it's kept on its own schedule separate from the daily rebalance. Unlike the
rest of this section, this path isn't exercised by the standalone Claude Code CLI in this
environment (only the VSCode extension is), so treat the exact flags as documented-but-not-
tested here and verify against `claude --help` on your machine:

```powershell
npm install -g @anthropic-ai/claude-code   # one-time
claude setup-token                          # one-time; makes unattended runs possible
```

Then register a second scheduled task, e.g. weekly on Monday morning:

```powershell
schtasks /create /tn "AlpacaWeeklyReview" /tr "C:\Users\hdroe\OneDrive\Documents\Coursera\alpaca-trading-api\scripts\weekly_review.bat" /sc WEEKLY /d MON /st 08:00
```

## Debugging and Logging

Logs are output to console by default. Configure logging:

```python
from src.trading.utils.logging import setup_logger

logger = setup_logger('my_strategy', log_file='logs/strategy.log')
logger.info("Strategy initialized")
```

## Tips & Best Practices

1. **Start Simple**: Begin with a simple strategy to understand the framework
2. **Test First**: Write unit tests for your strategy logic
3. **Version Control**: Commit your strategies and track experiments
4. **Parameter Sweeps**: Use scripts to test multiple parameter combinations
5. **Walk-Forward**: Validate strategies on out-of-sample data
6. **Slippage**: Always include realistic transaction costs
7. **Data Quality**: Inspect your data for gaps and anomalies
8. **Reproducibility**: Pin package versions and seed random number generators

## Resources

- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [Quantitative Trading](https://www.quantstart.com/) - QuantStart tutorials

## Contributing

This is a personal project, but improvements are welcome:
1. Add new strategies in `src/trading/strategy/examples/`
2. Improve performance metrics in `src/trading/backtest/metrics.py`
3. Add tests for new features
4. Document your changes

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves risk of loss. Always test strategies thoroughly on paper accounts before live trading.

---

**Version**: 0.1.0  
**Last Updated**: January 2026
