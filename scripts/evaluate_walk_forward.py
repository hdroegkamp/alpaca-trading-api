#!/usr/bin/env python3
"""Judge the pooled cross-sectional *method* across many out-of-sample windows.

A single 60-day holdout is far too short to tell skill from noise (a 60-day
Sharpe has a standard error of ~2). This script answers the prior question —
does the method hold up across many expanding-window walk-forward folds? —
and is the place to compare method changes (universe, label horizon,
rebalance cadence) before anything goes near the promotion gate.

Fold models train on the *labeled* panel (extremes only, as in production)
but are backtested on the *full* panel — every symbol on every date, as the
live rebalance sees it (see src/trading/ml/promotion.py).

Reports:
  - per-fold net/gross Sharpe, return, and max drawdown
  - stitched out-of-sample Sharpe, net of --commission and of --alt-commission
  - rank IC (Spearman of score vs the label-horizon forward return), with a
    t-stat over non-overlapping dates
  - the same IC for simple raw-feature baselines (does the model add anything?)
  - a random-score null for the stitched Sharpe (same cadence and costs)
  - how often a rolling --window-day Sharpe clears 0 (how noisy a short gate is)

Usage:
    python scripts/evaluate_walk_forward.py --universe starter
    python scripts/evaluate_walk_forward.py --universe universes/top30.txt
    python scripts/evaluate_walk_forward.py --universe starter --forward-periods 5 --rebalance-every 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.trading.backtest.portfolio import PortfolioBacktest
from src.trading.data.quality import audit_symbols, summarize_audit
from src.trading.data.storage import DataStore
from src.trading.data.universe import resolve_universe
from src.trading.ml.cross_sectional_labels import add_cross_sectional_labels
from src.trading.ml.feature_engineering import FeatureEngineering
from src.trading.ml.panel import build_panel
from src.trading.ml.promotion import walk_forward_backtest


class _RandomScorer:
    """Stand-in fold model that scores every row uniformly at random."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def predict_proba(self, X):
        p = self.rng.random(len(X))
        return np.column_stack([1 - p, p])


def _backtest(panel, folds, feature_cols, args, commission):
    bt = PortfolioBacktest(
        panel=panel,
        fold_models=folds,
        feature_columns=feature_cols,
        top_pct=args.top_pct,
        bottom_pct=args.bottom_pct,
        commission=commission,
        rebalance_every_n_days=args.rebalance_every,
    )
    results = bt.run()
    return results, bt.get_summary()


def _rank_ic(scored, score_col, step):
    """Per-date Spearman IC of ``score_col`` vs forward_return, on every ``step``-th date.

    With an h-day label, consecutive dates' forward returns overlap, so only
    every h-th date is used — otherwise the t-stat is inflated ~sqrt(h).
    """
    dates = np.sort(scored["date"].unique())[::step]
    valid = scored[scored["date"].isin(dates)].dropna(subset=[score_col, "forward_return"])
    ic = valid.groupby("date").apply(
        lambda g: g[score_col].rank().corr(g["forward_return"].rank()) if len(g) >= 3 else np.nan,
        include_groups=False,
    ).dropna()
    t_stat = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
    return ic, t_stat


def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation of the pooled method")
    parser.add_argument("--universe", default="starter")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="1Day")
    parser.add_argument("--min-history", type=int, default=70)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument(
        "--forward-periods",
        type=int,
        default=1,
        help="Label horizon in bars (default 1). The embargo is set to match.",
    )
    parser.add_argument(
        "--rebalance-every", type=int, default=1, help="Rebalance cadence in trading days"
    )
    parser.add_argument("--top-pct", type=float, default=0.2)
    parser.add_argument("--bottom-pct", type=float, default=0.2)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument(
        "--alt-commission",
        type=float,
        default=0.0002,
        help="Second cost level to report the stitched Sharpe at (default 0.0002)",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=30)
    parser.add_argument("--n-null", type=int, default=50, help="Random-score null trials")
    parser.add_argument("--window", type=int, default=60, help="Rolling Sharpe window (days)")
    args = parser.parse_args()

    store = DataStore(data_dir=args.data_dir)
    symbols = resolve_universe(args.universe, data_store=store, timeframe=args.timeframe)
    for line in summarize_audit(audit_symbols(store, symbols, args.timeframe)):
        print(f"WARNING: {line}")
    panel = build_panel(
        symbols,
        store,
        timeframe=args.timeframe,
        forward_periods=args.forward_periods,
        min_history=args.min_history,
    )
    labeled = add_cross_sectional_labels(
        panel, top_quantile=args.top_pct, bottom_quantile=args.bottom_pct
    )
    feature_cols = FeatureEngineering.FEATURE_COLUMNS
    print(
        f"Universe '{args.universe}': {panel['symbol'].nunique()} symbols, "
        f"{panel['date'].nunique()} dates, {panel['date'].min().date()} -> "
        f"{panel['date'].max().date()} | label horizon {args.forward_periods}d, "
        f"rebalance every {args.rebalance_every}d"
    )

    wf = walk_forward_backtest(
        labeled,
        panel,
        feature_cols=feature_cols,
        model_params={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
        },
        n_splits=args.n_splits,
        embargo_periods=args.forward_periods,
        top_pct=args.top_pct,
        bottom_pct=args.bottom_pct,
        commission=args.commission,
        rebalance_every_n_days=args.rebalance_every,
    )
    folds = wf["folds"]

    # --- Per-fold portfolio performance ---------------------------------
    print(f"\nPer-fold out-of-sample results (commission={args.commission}):")
    print(f"{'fold':>4}  {'test window':<25} {'days':>5} {'net SR':>7} {'gross SR':>8} "
          f"{'net ret':>8} {'max DD':>8}")
    n_net_pos = n_gross_pos = 0
    for f in folds:
        _, net = _backtest(panel, [f], feature_cols, args, args.commission)
        _, gross = _backtest(panel, [f], feature_cols, args, 0.0)
        n_net_pos += net["sharpe_ratio"] > 0
        n_gross_pos += gross["sharpe_ratio"] > 0
        td = f["test_dates"]
        window = f"{td.min().date()} -> {td.max().date()}"
        print(f"{f['fold']:>4}  {window:<25} {len(td):>5} {net['sharpe_ratio']:>7.2f} "
              f"{gross['sharpe_ratio']:>8.2f} {net['total_return']:>8.1%} {net['max_drawdown']:>8.1%}")
    print(f"Folds with net Sharpe > 0: {n_net_pos}/{len(folds)}; gross > 0: {n_gross_pos}/{len(folds)}")

    # --- Stitched OOS curve ---------------------------------------------
    net_results, net_summary = wf["results"], wf["summary"]
    _, alt_summary = _backtest(panel, folds, feature_cols, args, args.alt_commission)
    _, gross_summary = _backtest(panel, folds, feature_cols, args, 0.0)
    n_days = len(net_results)
    print(f"\nStitched OOS ({n_days} days): Sharpe net@{args.commission} "
          f"{net_summary['sharpe_ratio']:.2f}, net@{args.alt_commission} "
          f"{alt_summary['sharpe_ratio']:.2f}, gross {gross_summary['sharpe_ratio']:.2f} | "
          f"net@{args.commission} total return {net_summary['total_return']:.1%}, "
          f"max DD {net_summary['max_drawdown']:.1%}")
    print(f"  (Sharpe standard error at this length ~ {np.sqrt(252 / n_days):.2f})")

    # --- Rank IC: model vs raw-feature baselines --------------------------
    scored_parts = []
    for f in folds:
        rows = panel[panel["date"].isin(f["test_dates"])].dropna(subset=feature_cols).copy()
        X = f["scaler"].transform(rows[feature_cols].values)
        rows["score"] = f["model"].predict_proba(X)[:, 1]
        scored_parts.append(rows)
    scored = pd.concat(scored_parts, ignore_index=True)

    print(f"\nRank IC vs {args.forward_periods}-day forward return "
          f"(non-overlapping OOS dates):")
    ic, t = _rank_ic(scored, "score", args.forward_periods)
    print(f"  {'model':<18} mean IC {ic.mean():+.4f}  t-stat {t:+.2f}  "
          f"(% dates > 0: {(ic > 0).mean():.0%}, n={len(ic)})")
    for feat in ["momentum_5d", "momentum_20d", "rsi", "volatility_20d"]:
        fic, ft = _rank_ic(scored, feat, args.forward_periods)
        print(f"  {feat:<18} mean IC {fic.mean():+.4f}  t-stat {ft:+.2f}")

    # --- Random-score null ----------------------------------------------
    if args.n_null > 0:
        null_sharpes = []
        for seed in range(args.n_null):
            rand_folds = [{**f, "model": _RandomScorer(seed * 1000 + f["fold"])} for f in folds]
            _, s = _backtest(panel, rand_folds, feature_cols, args, args.commission)
            null_sharpes.append(s["sharpe_ratio"])
        null_sharpes = np.array(null_sharpes)
        pct = (null_sharpes < net_summary["sharpe_ratio"]).mean()
        print(f"\nRandom-score null ({args.n_null} trials, same cadence/costs): mean Sharpe "
              f"{null_sharpes.mean():.2f}, 5th-95th pct [{np.percentile(null_sharpes, 5):.2f}, "
              f"{np.percentile(null_sharpes, 95):.2f}] — model beats {pct:.0%} of random books "
              "(random books also churn more, so compare gross Sharpe too)")

    # --- How noisy is a short gate? --------------------------------------
    r = net_results["portfolio_return"]
    rolling = (r.rolling(args.window).mean() / r.rolling(args.window).std() * np.sqrt(252)).dropna()
    print(f"\nRolling {args.window}-day net Sharpe over the OOS curve: "
          f"> 0 in {(rolling > 0).mean():.0%} of windows, "
          f"range [{rolling.min():.2f}, {rolling.max():.2f}], latest {rolling.iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
