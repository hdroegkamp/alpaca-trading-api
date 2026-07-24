"""Predefined and file-based stock universes for downloads and panel construction."""

from pathlib import Path
from typing import List, Optional

from .storage import DataStore

# Predefined symbol universes
UNIVERSES = {
    "indices": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO"],
    "sectors": [
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLI",
        "XLP",
        "XLY",
        "XLU",
        "XLB",
        "XLRE",
        "XLC",
    ],
    "mega_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B"],
    "tech": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "NVDA",
        "AMD",
        "INTC",
        "CRM",
        "ORCL",
        "CSCO",
    ],
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "HES", "OXY"],
    "healthcare": [
        "UNH",
        "JNJ",
        "LLY",
        "PFE",
        "ABBV",
        "TMO",
        "MRK",
        "DHR",
        "ABT",
        "BMY",
    ],
    "starter": [
        "SPY",
        "QQQ",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "JPM",
    ],
}


def load_universe_file(filepath: str) -> List[str]:
    """Load symbols from a text file (one per line).

    Args:
        filepath: Path to universe file

    Returns:
        List of symbols
    """
    symbols = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                symbols.append(line.upper())
    return symbols


def symbols_from_inventory(data_store: DataStore, timeframe: str = "1Day") -> List[str]:
    """List every symbol with locally stored data for a timeframe.

    Unlike the curated ``UNIVERSES``/universe-file lists, this reflects
    whatever is actually available in ``data_store`` right now — the natural
    choice for a "full universe" panel run.

    Args:
        data_store: Store to inspect.
        timeframe: Bar timeframe to filter by.

    Returns:
        Sorted list of symbols with an existing data file.
    """
    return data_store.list_symbols(timeframe=timeframe)


def resolve_universe(
    spec: str,
    data_store: Optional[DataStore] = None,
    timeframe: str = "1Day",
) -> List[str]:
    """Resolve a single CLI-friendly string into a list of symbols.

    Shared by scripts that need a universe from a ``--universe`` flag
    (``train_pooled_model.py``, ``run_live_rebalance.py``) so the resolution
    rules live in one place. Tried in order:

    1. A key in ``UNIVERSES`` (e.g. ``"starter"``).
    2. The literal ``"inventory"`` — every symbol with locally stored data
       for ``timeframe`` (requires ``data_store``).
    3. A path to an existing universe file (one symbol per line).
    4. A comma-separated manual list (e.g. ``"AAPL,MSFT,GOOGL"``).

    Args:
        spec: The universe specifier.
        data_store: Required only if ``spec == "inventory"``.
        timeframe: Bar timeframe, used only for the ``"inventory"`` case.

    Returns:
        List of symbols.

    Raises:
        ValueError: If ``spec == "inventory"`` but ``data_store`` is None.
    """
    if spec in UNIVERSES:
        return UNIVERSES[spec]

    if spec == "inventory":
        if data_store is None:
            raise ValueError("--universe inventory requires a data_store")
        return symbols_from_inventory(data_store, timeframe=timeframe)

    if Path(spec).exists():
        return load_universe_file(spec)

    return [s.strip().upper() for s in spec.split(",") if s.strip()]
