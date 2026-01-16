#!/usr/bin/env python3
"""Run backtest on historical data.

Usage:
    python scripts/run_backtest.py AAPL --strategy MovingAverageCrossover --fast-window 10 --slow-window 30
    python scripts/run_backtest.py MSFT --strategy MeanReversion --window 20 --num-std 2.0
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading.data import DataStore
from src.trading.strategy.examples import MovingAverageCrossover, MeanReversion
from src.trading.backtest import VectorizedBacktest
from src.trading.utils.logging import setup_logger

logger = setup_logger("run_backtest")


def parse_strategy_params(args):
    """Parse strategy-specific parameters from command line."""
    params = {}

    # Moving Average Crossover params
    if hasattr(args, "fast_window") and args.fast_window:
        params["fast_window"] = args.fast_window
    if hasattr(args, "slow_window") and args.slow_window:
        params["slow_window"] = args.slow_window

    # Mean Reversion params
    if hasattr(args, "window") and args.window:
        params["window"] = args.window
    if hasattr(args, "num_std") and args.num_std:
        params["num_std"] = args.num_std

    return params


def main():
    parser = argparse.ArgumentParser(
        description="Run vectorized backtest on historical data"
    )
    parser.add_argument("symbol", help="Trading symbol to backtest")
    parser.add_argument(
        "--strategy",
        default="MovingAverageCrossover",
        choices=["MovingAverageCrossover", "MeanReversion"],
        help="Strategy to use",
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--timeframe", default="1Day", help="Data timeframe")
    parser.add_argument(
        "--initial-capital", type=float, default=100000.0, help="Initial capital"
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Commission rate (0.001 = 0.1%%)",
    )
    parser.add_argument("--plot", action="store_true", help="Show equity curve plot")

    # Strategy-specific parameters
    parser.add_argument(
        "--fast-window", type=int, help="Fast MA window (for MA crossover)"
    )
    parser.add_argument(
        "--slow-window", type=int, help="Slow MA window (for MA crossover)"
    )
    parser.add_argument("--window", type=int, help="Window for mean reversion")
    parser.add_argument(
        "--num-std", type=float, help="Number of std devs for Bollinger Bands"
    )

    args = parser.parse_args()

    logger.info(f"Running backtest for {args.symbol}")
    logger.info(f"Strategy: {args.strategy}")

    try:
        # Load data
        store = DataStore(data_dir=args.data_dir)
        data = store.load(args.symbol, timeframe=args.timeframe)

        if data is None:
            logger.error(
                f"No data found for {args.symbol}. Run download_data.py first."
            )
            sys.exit(1)

        logger.info(f"Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")

        # Parse strategy params
        strategy_params = parse_strategy_params(args)

        # Initialize strategy
        if args.strategy == "MovingAverageCrossover":
            strategy = MovingAverageCrossover(**strategy_params)
        elif args.strategy == "MeanReversion":
            strategy = MeanReversion(**strategy_params)
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

        logger.info(f"Strategy parameters: {strategy}")

        # Run backtest
        logger.info("Running backtest...")
        backtest = VectorizedBacktest(
            strategy=strategy,
            data=data,
            initial_capital=args.initial_capital,
            commission=args.commission,
        )

        results = backtest.run()

        # Print summary
        summary = backtest.get_summary()

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Strategy:          {summary['strategy']}")
        print(f"Symbol:            {args.symbol}")
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

        # Plot if requested
        if args.plot:
            logger.info("Generating plot...")
            backtest.plot()

        logger.info("Backtest complete!")

    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
