"""Data-quality checks run before training or evaluating on the archive.

Two failure modes have bitten this project:

- Unadjusted prices: every split shows up as a fake one-day crash (or, for
  reverse splits, a fake rally), corrupting returns, features and labels.
  Files saved without an ``adjustment`` in their metadata predate adjusted
  downloads and are treated as unadjusted.
- Implausible one-day moves: even in adjusted data, a |return| above ~50%
  is either a genuine event (biotech readout, buyout) or a bad print. Either
  way it deserves a look before it drives a model.
"""

from typing import List

import pandas as pd

from .storage import DataStore

#: |close-to-close return| above which a bar is flagged.
DEFAULT_MAX_ABS_RETURN = 0.5

AUDIT_COLUMNS = ["symbol", "adjustment", "n_suspect", "max_abs_return", "worst_date"]


def suspect_moves(
    data: pd.DataFrame, max_abs_return: float = DEFAULT_MAX_ABS_RETURN
) -> pd.Series:
    """Close-to-close returns whose magnitude exceeds ``max_abs_return``.

    Args:
        data: OHLCV DataFrame with a ``close`` column.
        max_abs_return: Flag threshold (0.5 = a 50% one-day move).

    Returns:
        The offending returns, indexed by the date of the move.
    """
    returns = data["close"].astype(float).pct_change()
    return returns[returns.abs() > max_abs_return]


def audit_symbols(
    store: DataStore,
    symbols: List[str],
    timeframe: str = "1Day",
    max_abs_return: float = DEFAULT_MAX_ABS_RETURN,
) -> pd.DataFrame:
    """Audit stored data for ``symbols``: adjustment and suspect moves.

    Args:
        store: Archive to read.
        symbols: Symbols to audit (missing ones are skipped).
        timeframe: Bar timeframe.
        max_abs_return: See :func:`suspect_moves`.

    Returns:
        One row per symbol with stored data: ``symbol``, ``adjustment``
        (from metadata, None if unknown), ``n_suspect``, ``max_abs_return``
        and ``worst_date`` (date of the largest move).
    """
    rows = []
    for symbol in symbols:
        path = store.data_dir / timeframe / f"{symbol}.parquet"
        if not path.exists():
            continue
        data = pd.read_parquet(path, columns=["close"])
        returns = data["close"].astype(float).pct_change().abs()
        meta = store.get_metadata(symbol, timeframe) or {}
        rows.append(
            {
                "symbol": symbol,
                "adjustment": meta.get("adjustment"),
                "n_suspect": int((returns > max_abs_return).sum()),
                "max_abs_return": float(returns.max()) if returns.notna().any() else 0.0,
                "worst_date": returns.idxmax() if returns.notna().any() else pd.NaT,
            }
        )
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def unadjusted_symbols(audit: pd.DataFrame) -> List[str]:
    """Symbols in an :func:`audit_symbols` result not stored with ``adjustment="all"``."""
    return audit.loc[audit["adjustment"] != "all", "symbol"].tolist()


def summarize_audit(audit: pd.DataFrame, max_listed: int = 10) -> List[str]:
    """Human-readable warning lines for an :func:`audit_symbols` result (empty if clean)."""
    lines = []
    unadjusted = unadjusted_symbols(audit)
    if unadjusted:
        lines.append(
            f"{len(unadjusted)}/{len(audit)} symbols are not split/dividend-adjusted "
            f"(run scripts/update_archive.py): {', '.join(unadjusted[:max_listed])}"
            + (" ..." if len(unadjusted) > max_listed else "")
        )
    suspect = audit[audit["n_suspect"] > 0].sort_values("max_abs_return", ascending=False)
    if not suspect.empty:
        listed = ", ".join(
            f"{r.symbol} {r.max_abs_return:+.0%} on {pd.Timestamp(r.worst_date).date()}"
            for r in suspect.head(max_listed).itertuples()
        )
        lines.append(
            f"{len(suspect)}/{len(audit)} symbols have one-day moves beyond the "
            f"suspect threshold: {listed}" + (" ..." if len(suspect) > max_listed else "")
        )
    return lines
