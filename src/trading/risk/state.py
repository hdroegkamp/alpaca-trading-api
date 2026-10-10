"""Persisted circuit-breaker state for the live rebalance loop.

``RiskManager`` (see ``rules.py``) is pure, in-process logic with no file
I/O, so a tripped circuit breaker doesn't survive between separate
invocations of ``scripts/run_live_rebalance.py`` — e.g. across daily Task
Scheduler runs. This module persists the trip decision to a small JSON file
so a bad day actually halts *future* scheduled runs until a human
deliberately resets it.
"""

import json
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
    if not path.exists():
        return {"triggered": False, "reason": None, "triggered_at": None}
    with open(path) as f:
        return json.load(f)


def save_breaker_state(
    path: PathLike = DEFAULT_STATE_PATH,
    triggered: bool = True,
    reason: Optional[str] = None,
    triggered_at: Optional[str] = None,
) -> None:
    """Persist breaker state to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {"triggered": triggered, "reason": reason, "triggered_at": triggered_at},
            f,
            indent=2,
        )


def reset_breaker_state(path: PathLike = DEFAULT_STATE_PATH) -> None:
    """Clear a tripped breaker — the deliberate manual-reset step."""
    save_breaker_state(path, triggered=False, reason=None, triggered_at=None)


def evaluate_circuit_breaker(
    equity_log_path: PathLike,
    current_equity: float,
    risk_manager: RiskManager,
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
    peak_equity = max(float(history["equity"].max()), current_equity)

    daily_pnl = current_equity - prior_equity
    allowed, reason = risk_manager.check_daily_loss(daily_pnl, current_equity)
    if not allowed:
        return True, reason

    allowed, reason = risk_manager.check_drawdown(current_equity, peak_equity)
    if not allowed:
        return True, reason

    return False, None
