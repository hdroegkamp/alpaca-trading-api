"""Unit tests for the Alpaca broker execution wrapper (no live network calls).

``AlpacaBroker`` accepts an injected ``client``, so every test here runs
against a ``MagicMock`` standing in for ``alpaca.trading.client.TradingClient``
— nothing here touches the network or requires credentials.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alpaca.trading.enums import PositionSide

from src.trading.execution.broker import AlpacaBroker
from src.trading.risk.rules import RiskManager


def _position(symbol, qty, side=PositionSide.LONG):
    return SimpleNamespace(symbol=symbol, qty=str(qty), side=side)


def _make_broker(equity=100_000.0, cash=50_000.0, positions=None, is_open=True):
    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity=str(equity), cash=str(cash))
    client.get_all_positions.return_value = positions or []
    client.get_clock.return_value = SimpleNamespace(is_open=is_open)
    client.submit_order.return_value = SimpleNamespace(id="order-123")
    return AlpacaBroker(client=client), client


class TestAccountState:
    def test_get_equity_and_cash(self):
        broker, _ = _make_broker(equity=123456.78, cash=999.0)
        assert broker.get_equity() == pytest.approx(123456.78)
        assert broker.get_cash() == pytest.approx(999.0)

    def test_get_positions_derives_sign_from_side(self):
        broker, _ = _make_broker(
            positions=[
                _position("AAPL", 10, PositionSide.LONG),
                _position("TSLA", 5, PositionSide.SHORT),
            ]
        )
        assert broker.get_positions() == {"AAPL": 10.0, "TSLA": -5.0}

    def test_is_market_open(self):
        broker, _ = _make_broker(is_open=False)
        assert broker.is_market_open() is False


class TestSubmitTargetWeights:
    def test_dry_run_computes_but_does_not_submit(self):
        broker, client = _make_broker(equity=100_000.0, positions=[])

        results = broker.submit_target_weights(
            target_weights={"AAPL": 0.2},
            prices={"AAPL": 100.0},
            dry_run=True,
        )

        row = next(r for r in results if r["symbol"] == "AAPL")
        assert row["status"] == "dry_run"
        assert row["side"] == "buy"
        assert row["qty"] == 200  # 0.2 * 100_000 / 100
        client.submit_order.assert_not_called()

    def test_live_submits_market_order(self):
        broker, client = _make_broker(equity=100_000.0, positions=[])

        results = broker.submit_target_weights(
            target_weights={"AAPL": 0.2},
            prices={"AAPL": 100.0},
            dry_run=False,
        )

        row = next(r for r in results if r["symbol"] == "AAPL")
        assert row["status"] == "submitted"
        client.submit_order.assert_called_once()
        submitted_request = client.submit_order.call_args[0][0]
        assert submitted_request.symbol == "AAPL"
        assert submitted_request.qty == 200

    def test_symbol_dropped_from_book_is_closed_out(self):
        broker, client = _make_broker(
            equity=100_000.0, positions=[_position("TSLA", 50, PositionSide.LONG)]
        )

        results = broker.submit_target_weights(
            target_weights={},  # TSLA no longer in the book
            prices={"TSLA": 200.0},
            dry_run=False,
        )

        row = next(r for r in results if r["symbol"] == "TSLA")
        assert row["side"] == "sell"
        assert row["qty"] == 50
        assert row["status"] == "submitted"

    def test_small_dollar_delta_below_threshold_is_skipped(self):
        broker, client = _make_broker(equity=100_000.0, positions=[])

        results = broker.submit_target_weights(
            target_weights={"AAPL": 0.00006},  # $6 target at $5/share -> 1 share, $5 delta
            prices={"AAPL": 5.0},
            dry_run=False,
            min_trade_dollars=50.0,
        )

        row = next(r for r in results if r["symbol"] == "AAPL")
        assert row["status"] == "no_change"
        client.submit_order.assert_not_called()

    def test_risk_manager_rejection_is_skipped_not_raised(self):
        broker, client = _make_broker(equity=100_000.0, positions=[])
        risk_manager = RiskManager(max_position_size=0.1)  # 10% cap

        results = broker.submit_target_weights(
            target_weights={"AAPL": 0.5},  # way over the 10% cap
            prices={"AAPL": 100.0},
            dry_run=False,
            risk_manager=risk_manager,
        )

        row = next(r for r in results if r["symbol"] == "AAPL")
        assert row["status"] == "rejected_by_risk_manager"
        client.submit_order.assert_not_called()

    def test_missing_price_is_skipped(self):
        broker, client = _make_broker(equity=100_000.0, positions=[])

        results = broker.submit_target_weights(
            target_weights={"AAPL": 0.2},
            prices={},
            dry_run=False,
        )

        row = next(r for r in results if r["symbol"] == "AAPL")
        assert row["status"] == "skipped"
        client.submit_order.assert_not_called()

    def test_short_target_on_new_symbol_sells(self):
        broker, client = _make_broker(equity=100_000.0, positions=[])

        results = broker.submit_target_weights(
            target_weights={"TSLA": -0.2},
            prices={"TSLA": 200.0},
            dry_run=False,
        )

        row = next(r for r in results if r["symbol"] == "TSLA")
        assert row["side"] == "sell"
        assert row["qty"] == 100  # 0.2 * 100_000 / 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
