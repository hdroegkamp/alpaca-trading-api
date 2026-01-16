#!/usr/bin/env python3
"""
Quick example demonstrating the trading system workflow.

This script:
1. Downloads sample data for AAPL (or loads if already present)
2. Runs a simple Moving Average Crossover backtest
3. Displays performance metrics

Run: .\.venv\Scripts\python.exe scripts\example_workflow.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading.data import AlpacaDataFetcher, DataStore
from src.trading.strategy.examples import MovingAverageCrossover
from src.trading.backtest import VectorizedBacktest
from src.trading.utils.logging import setup_logger

logger = setup_logger("example")


def main():
    """Run example workflow."""
    symbol = "AAPL"
    start_date = "2020-01-01"

    print("\n" + "=" * 60)
    print("ALGORITHMIC TRADING SYSTEM - EXAMPLE WORKFLOW")
    print("=" * 60 + "\n")

    # Step 1: Load or download data
    print(f"Step 1: Loading data for {symbol}...")
    store = DataStore(data_dir="data")

    data = store.load(symbol)

    if data is None:
        print(f"  Data not found locally. Downloading from Alpaca...")
        try:
            fetcher = AlpacaDataFetcher()
            data = fetcher.fetch_bars(
                symbol=symbol, start=start_date, end=datetime.now().strftime("%Y-%m-%d")
            )
            store.save(symbol, data)
            print(f"  ✓ Downloaded {len(data)} bars")
        except Exception as e:
            print(f"  ✗ Error downloading data: {e}")
            print("\n  To download data manually, run:")
            print(
                f"    .venv\\Scripts\\python.exe scripts\\download_data.py {symbol} --start {start_date}"
            )
            return
    else:
        print(
            f"  ✓ Loaded {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}"
        )

    # Step 2: Initialize strategy
    print("\nStep 2: Initializing strategy...")
    strategy = MovingAverageCrossover(fast_window=20, slow_window=50)
    print(f"  ✓ {strategy}")

    # Step 3: Run backtest
    print("\nStep 3: Running backtest...")
    backtest = VectorizedBacktest(
        strategy=strategy,
        data=data,
        initial_capital=100000.0,
        commission=0.001,  # 0.1% commission
    )

    results = backtest.run()
    print(f"  ✓ Backtest complete ({len(results)} periods)")

    # Step 4: Display results
    print("\nStep 4: Performance Summary")
    summary = backtest.get_summary()

    print("\n" + "-" * 60)
    print(f"Strategy:          {summary['strategy']}")
    print(f"Symbol:            {symbol}")
    print(f"Period:            {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Trading Days:      {summary['n_days']}")
    print(f"Number of Trades:  {summary['n_trades']}")
    print("\n" + "-" * 60)
    print(f"Initial Capital:   ${summary['initial_capital']:,.2f}")
    print(f"Final Equity:      ${summary['final_equity']:,.2f}")
    print(f"Total Return:      {summary['total_return']:.2%}")
    print(f"CAGR:              {summary['cagr']:.2%}")
    print("\n" + "-" * 60)
    print(f"Sharpe Ratio:      {summary['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:      {summary['max_drawdown']:.2%}")
    print(f"Volatility (Ann.): {summary['volatility']:.2%}")
    print(f"Win Rate:          {summary['win_rate']:.2%}")
    print("=" * 60 + "\n")

    # Step 5: What's next?
    print("Next Steps:")
    print("  1. Try different parameters:")
    print(
        f"     .venv\\Scripts\\python.exe scripts\\run_backtest.py {symbol} --strategy MovingAverageCrossover --fast-window 10 --slow-window 30"
    )
    print("\n  2. Test a different strategy:")
    print(
        f"     .venv\\Scripts\\python.exe scripts\\run_backtest.py {symbol} --strategy MeanReversion --window 20 --num-std 2.0"
    )
    print("\n  3. Visualize results:")
    print(
        f"     .venv\\Scripts\\python.exe scripts\\run_backtest.py {symbol} --strategy MovingAverageCrossover --plot"
    )
    print("\n  4. Download more symbols:")
    print(
        f"     .venv\\Scripts\\python.exe scripts\\download_data.py MSFT GOOGL TSLA --start {start_date}"
    )
    print("\n  5. Create your own strategy:")
    print("     See TRADING_README.md for instructions\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
