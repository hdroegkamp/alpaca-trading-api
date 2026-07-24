"""Unit tests for single-cross-section target portfolio construction."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.execution.target_portfolio import build_target_weights


def _scored(scores):
    """Build a one-date scored frame from {symbol: score}."""
    return pd.DataFrame(
        {"symbol": list(scores.keys()), "probability_up": list(scores.values())}
    )


class TestBuildTargetWeights:
    def test_top_and_bottom_legs_selected_correctly(self):
        scored = _scored({f"SYM{i}": i / 10 for i in range(10)})  # scores 0.0..0.9

        weights = build_target_weights(scored, top_pct=0.2, bottom_pct=0.2)

        assert set(weights) == {"SYM9", "SYM8", "SYM0", "SYM1"}
        assert weights["SYM9"] == pytest.approx(0.25)
        assert weights["SYM8"] == pytest.approx(0.25)
        assert weights["SYM0"] == pytest.approx(-0.25)
        assert weights["SYM1"] == pytest.approx(-0.25)

    def test_book_is_dollar_neutral(self):
        scored = _scored({f"SYM{i}": i for i in range(20)})

        weights = build_target_weights(scored, top_pct=0.25, bottom_pct=0.25)

        assert sum(weights.values()) == pytest.approx(0.0)

    def test_unselected_symbols_absent_from_book(self):
        scored = _scored({f"SYM{i}": i for i in range(10)})

        weights = build_target_weights(scored, top_pct=0.2, bottom_pct=0.2)

        assert "SYM5" not in weights
        assert len(weights) == 4

    def test_small_universe_still_gets_at_least_one_per_leg(self):
        scored = _scored({"A": 1, "B": 2, "C": 3})

        weights = build_target_weights(scored, top_pct=0.2, bottom_pct=0.2)

        assert weights == {"C": pytest.approx(0.5), "A": pytest.approx(-0.5)}

    def test_quantiles_summing_over_one_raises(self):
        scored = _scored({"A": 1, "B": 2})

        with pytest.raises(ValueError, match="must not exceed 1.0"):
            build_target_weights(scored, top_pct=0.7, bottom_pct=0.6)

    def test_universe_too_small_for_both_legs_raises(self):
        scored = _scored({"A": 1})

        with pytest.raises(ValueError, match="too small"):
            build_target_weights(scored, top_pct=0.5, bottom_pct=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
