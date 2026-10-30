import numpy as np
import pandas as pd
import pytest

from baselines.scenarios import build_median_scenario
from market_calendar import efa_block_bounds_utc
from run_baselines import _forecast_metrics_one_market


@pytest.mark.parametrize(
    ("window_start", "expected_dc"),
    [("2023-10-15", 0.0), ("2025-01-15", -2.0)],
)
def test_median_baseline_applies_dc_price_regime(window_start: str, expected_dc: float):
    start = pd.Timestamp(window_start, tz="UTC")
    index = pd.date_range(start, periods=8, freq="30min")
    energy = pd.DataFrame({"q50": np.full(len(index), 50.0)}, index=index)
    negative_dc = pd.DataFrame({"q50": np.full(len(index), -2.0)}, index=index)

    output = build_median_scenario(
        start,
        len(index),
        energy,
        energy,
        negative_dc,
        negative_dc,
    )

    assert np.all(np.asarray(output["dc_low"]) == expected_dc)
    assert np.all(np.asarray(output["dc_high"]) == expected_dc)


def test_dc_forecast_metrics_include_complete_dst_short_block():
    key = (pd.Timestamp("2025-03-29").date(), 23)
    index = pd.date_range(*efa_block_bounds_utc(key), freq="30min", inclusive="left")
    actual = pd.Series(np.full(len(index), 10.0), index=index)
    forecast = pd.DataFrame(
        {
            "q10": np.full(len(index), 8.0),
            "q50": np.full(len(index), 10.0),
            "q90": np.full(len(index), 12.0),
        },
        index=index,
    )

    metrics, count = _forecast_metrics_one_market(
        actual,
        forecast,
        clean_dates=set(index.date),
        block_level=True,
    )

    assert len(index) == 6
    assert count == 1
    assert metrics["mae"] == 0.0
