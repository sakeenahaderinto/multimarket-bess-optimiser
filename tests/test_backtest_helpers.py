import pytest

pytest.importorskip("pyomo.environ")

import pandas as pd

from backtest.engine import (
    _normalise_to_30min,
    _require_exact_grid,
    _seed_for_date,
    settle_revenue,
)


def test_da_normalisation_fills_only_the_matching_second_half_hour():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-01 00:00", tz="UTC"),
            pd.Timestamp("2025-01-01 01:00", tz="UTC"),
            pd.Timestamp("2025-01-01 03:00", tz="UTC"),
        ]
    )
    frame = pd.DataFrame({"value": [10.0, 20.0, 40.0]}, index=index)
    output = _normalise_to_30min(frame, "da_actual", fill_limit=1)
    assert output.loc["2025-01-01 00:30", "value"] == 10.0
    assert output.loc["2025-01-01 01:30", "value"] == 20.0
    assert pd.isna(output.loc["2025-01-01 02:00", "value"])
    assert pd.isna(output.loc["2025-01-01 02:30", "value"])


def test_exact_grid_rejects_missing_value():
    index = pd.date_range("2025-01-01", periods=3, freq="30min", tz="UTC")
    frame = pd.DataFrame({"value": [1.0, None, 3.0]}, index=index)
    with pytest.raises(ValueError, match="missing periods"):
        _require_exact_grid(frame, index, "test")


def test_daily_seed_is_stable_and_date_specific():
    first = _seed_for_date(42, pd.Timestamp("2025-01-01").date())
    repeat = _seed_for_date(42, pd.Timestamp("2025-01-01").date())
    next_day = _seed_for_date(42, pd.Timestamp("2025-01-02").date())
    assert first == repeat
    assert first != next_day


def test_settlement_rejects_negative_degradation_cost():
    actual = pd.Series([1.0])

    with pytest.raises(ValueError, match="replacement_cost_per_mwh must be non-negative"):
        settle_revenue(
            object(),
            actual,
            actual,
            actual,
            actual,
            "battery",
            1,
            opt_settings={"replacement_cost_per_mwh": -1.0},
        )


def test_settlement_rejects_infinite_actual_price():
    finite = pd.Series([1.0])
    infinite = pd.Series([float("inf")])

    with pytest.raises(ValueError, match="da actual contains non-finite values"):
        settle_revenue(
            object(),
            infinite,
            finite,
            finite,
            finite,
            "battery",
            1,
        )
