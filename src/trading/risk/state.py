"""Persisted circuit-breaker state for the live rebalance loop.

``RiskManager`` (see ``rules.py``) is pure, in-process logic with no file
I/O, so a tripped circuit breaker doesn't survive between separate
invocations of ``scripts/run_live_rebalance.py`` — e.g. across daily Task
Scheduler runs. This module persists the trip decision to a small JSON file
so a bad day actually halts *future* scheduled runs until a human
deliberately resets it.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from .rules import RiskManager

DEFAULT_STATE_PATH = Path("data/live/circuit_breaker.json")

PathLike = Union[str, Path]


def load_breaker_state(path: PathLike = DEFAULT_STATE_PATH) -> dict:
    """Load the persisted breaker state.

    Returns a default, non-triggered state if no state file exists yet
    (e.g. the very first run).
    """
    path = Path(path)
    default = {"triggered": False, "reason": None, "triggered_at": None, "reset_at": None}
    if not path.exists():
        return default
    with open(path) as f:
        return {**default, **json.load(f)}


def save_breaker_state(
    path: PathLike = DEFAULT_STATE_PATH,
    triggered: bool = True,
    reason: Optional[str] = None,
    triggered_at: Optional[str] = None,
    reset_at: Optional[str] = None,
) -> None:
    """Persist breaker state to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "triggered": triggered,
                "reason": reason,
                "triggered_at": triggered_at,
                "reset_at": reset_at,
            },
            f,
            indent=2,
        )


def reset_breaker_state(
    path: PathLike = DEFAULT_STATE_PATH, reset_at: Optional[str] = None
) -> None:
    """Clear a tripped breaker — the deliberate manual-reset step.

    Records ``reset_at`` (default: now) so drawdown is subsequently measured
    from the peak *since this reset*; otherwise the pre-halt all-time peak
    would re-trip the breaker on the very next run.
    """
    if reset_at is None:
        reset_at = datetime.now().isoformat(timespec="seconds")
    save_breaker_state(path, triggered=False, reason=None, triggered_at=None, reset_at=reset_at)


def evaluate_circuit_breaker(
    equity_log_path: PathLike,
    current_equity: float,
    risk_manager: RiskManager,
    peak_since: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Check today's equity against ``risk_manager``'s daily-loss/drawdown limits.

    Reads ``equity_log_path`` (one row per prior run, oldest first) to find
    the most recent prior equity reading — used as a same-run-cadence proxy
    for "yesterday's" equity, since this script runs at most once per
    trading day — and the all-time peak equity, both compared against
    ``current_equity``.

    Args:
        equity_log_path: Path to ``data/live/equity_log.csv``. If it
            doesn't exist yet or is empty, there's nothing to compare
            against, so this never trips.
        current_equity: This run's current account equity.
        risk_manager: Supplies the configured ``max_daily_loss`` /
            ``max_drawdown`` thresholds.
        peak_since: ISO timestamp of the last manual reset, if any. Peak
            equity for the drawdown check only considers log rows at or
            after it (requires the log's ``timestamp`` column).

    Returns:
        ``(tripped, reason)`` — ``reason`` is set only when ``tripped`` is
        True.
    """
    equity_log_path = Path(equity_log_path)
    if not equity_log_path.exists():
        return False, None

    history = pd.read_csv(equity_log_path)
    if history.empty:
        return False, None

    prior_equity = float(history["equity"].iloc[-1])
    peak_history = history
    if peak_since is not None:
        since = pd.Timestamp(peak_since)
        peak_history = history[pd.to_datetime(history["timestamp"]) >= since]
    peak_equity = current_equity
    if not peak_history.empty:
        peak_equity = max(float(peak_history["equity"].max()), current_equity)

    daily_pnl = current_equity - prior_equity
    allowed, reason = risk_manager.check_daily_loss(daily_pnl, current_equity)
    if not allowed:
        return True, reason

    allowed, reason = risk_manager.check_drawdown(current_equity, peak_equity)
    if not allowed:
        return True, reason

    return False, None
