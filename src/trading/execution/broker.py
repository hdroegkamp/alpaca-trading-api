"""Live/paper order execution against Alpaca via alpaca-py.

Everything else in ``src.trading`` (data ingestion, features, the pooled
model, the backtest) only ever reads market data. This is the one place that
reads account state and submits real orders — kept deliberately small and
defensive: a bad response for one symbol must never take down the rest of a
rebalance.
"""

import os
from typing import Any, Dict, List, Optional

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, PositionSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
except ImportError as e:
    raise ImportError(f"alpaca-py not installed. Run: pip install alpaca-py ({e})")

from ..risk.rules import RiskManager


class AlpacaBroker:
    """Thin wrapper around alpaca-py's ``TradingClient`` for rebalancing a live/paper account.

    Attributes:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        client: The underlying ``TradingClient`` (or a test double — see
            the ``client`` constructor argument).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: Optional[bool] = None,
        client: Optional[Any] = None,
    ):
        """Initialize the broker.

        Args:
            api_key: API key (reads ``APCA_API_KEY_ID`` from env if omitted).
            api_secret: API secret (reads ``APCA_API_SECRET_KEY`` if omitted).
            paper: Paper vs. live account. Reads ``APCA_PAPER`` from env
                (default ``true``) if omitted — one explicit switch instead
                of the hardcoded ``paper=True`` scattered across scripts.
            client: Inject a pre-built client (e.g. a test double) instead
                of constructing a real ``TradingClient`` from credentials.
        """
        self.api_key = api_key or os.getenv("APCA_API_KEY_ID")
        self.api_secret = api_secret or os.getenv("APCA_API_SECRET_KEY")

        if client is not None:
            self.client = client
            return

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "API credentials not found. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY"
            )
        if paper is None:
            paper = os.getenv("APCA_PAPER", "true").strip().lower() != "false"
        self.client = TradingClient(self.api_key, self.api_secret, paper=paper)

    def get_equity(self) -> float:
        """Current account equity (cash + market value of all positions)."""
        return float(self.client.get_account().equity)

    def get_cash(self) -> float:
        """Current uninvested cash."""
        return float(self.client.get_account().cash)

    def get_positions(self) -> Dict[str, float]:
        """Current positions.

        Returns:
            Dict mapping symbol to signed share count (negative = short),
            derived from each position's ``side`` rather than trusting the
            API's own sign convention on ``qty``.
        """
        positions: Dict[str, float] = {}
        for p in self.client.get_all_positions():
            qty = abs(float(p.qty))
            positions[p.symbol] = qty if p.side == PositionSide.LONG else -qty
        return positions

    def is_market_open(self) -> bool:
        """Whether the market is open right now."""
        return bool(self.client.get_clock().is_open)

    def submit_target_weights(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        dry_run: bool = True,
        min_trade_dollars: float = 50.0,
        risk_manager: Optional[RiskManager] = None,
    ) -> List[Dict[str, Any]]:
        """Rebalance the account toward ``target_weights``.

        Diffs the requested book against every *current* position, not just
        the symbols in ``target_weights`` — a symbol held today but absent
        from the new book is treated as a target weight of 0 and closed out.

        Args:
            target_weights: symbol -> fraction of equity (e.g. from
                ``build_target_weights``); positive = long, negative = short.
            prices: symbol -> price used to convert each dollar target into
                whole shares. Must cover every symbol in ``target_weights``
                and every currently held symbol; a missing/non-positive
                price causes that symbol to be skipped, not closed.
            dry_run: If True (default), compute and return the intended
                orders without submitting anything.
            min_trade_dollars: Skip any rebalance whose dollar delta is
                smaller than this — avoids churn on rounding-sized deltas.
            risk_manager: If given, ``check_position_size`` gates each
                *target* position before it is sized; a rejected symbol is
                skipped (with the reason recorded), never raised.

        Returns:
            List of ``{symbol, side, qty, status, detail}`` dicts, one per
            symbol considered — including skipped/rejected/no-change ones —
            suitable for appending straight to a trade log.
        """
        equity = self.get_equity()
        current_positions = self.get_positions()
        all_symbols = sorted(set(target_weights) | set(current_positions))

        results: List[Dict[str, Any]] = []
        for symbol in all_symbols:
            target_weight = target_weights.get(symbol, 0.0)
            price = prices.get(symbol)
            current_qty = current_positions.get(symbol, 0.0)

            if price is None or price <= 0:
                results.append(
                    _row(symbol, None, 0, "skipped", "no price available")
                )
                continue

            target_value = target_weight * equity
            if risk_manager is not None and target_weight != 0.0:
                allowed, reason = risk_manager.check_position_size(target_value, equity)
                if not allowed:
                    results.append(
                        _row(symbol, None, 0, "rejected_by_risk_manager", reason)
                    )
                    continue

            target_shares = int(target_value / price)
            delta_shares = target_shares - int(current_qty)
            delta_value = abs(delta_shares) * price

            if delta_shares == 0 or delta_value < min_trade_dollars:
                results.append(
                    _row(
                        symbol,
                        None,
                        0,
                        "no_change",
                        f"delta ${delta_value:,.2f} below threshold or zero",
                    )
                )
                continue

            side = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL
            qty = abs(delta_shares)

            if dry_run:
                results.append(
                    _row(
                        symbol,
                        side.value,
                        qty,
                        "dry_run",
                        f"would {side.value} {qty} shares (~${delta_value:,.2f})",
                    )
                )
                continue

            try:
                order = self.client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                results.append(
                    _row(symbol, side.value, qty, "submitted", str(order.id))
                )
            except Exception as e:
                results.append(_row(symbol, side.value, qty, "error", str(e)))

        return results


def _row(symbol: str, side: Optional[str], qty: int, status: str, detail: str) -> Dict[str, Any]:
    return {"symbol": symbol, "side": side, "qty": qty, "status": status, "detail": detail}
