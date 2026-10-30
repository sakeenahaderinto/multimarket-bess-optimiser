import numpy as np
import pandas as pd
import pytest

from run_fixed_month_comparison import (
    make_perfect_foresight_builder,
    make_q50_anchored_builder,
)


@pytest.mark.parametrize(
    ("window_start", "expected_dc"),
    [("2023-10-15", 0.0), ("2025-01-15", -2.0)],
)
def test_perfect_foresight_builder_applies_dc_price_regime(
    window_start: str,
    expected_dc: float,
):
    start = pd.Timestamp(window_start, tz="UTC")
    index = pd.date_range(start, periods=8, freq="30min")
    energy = pd.DataFrame({"value": np.full(len(index), 50.0)}, index=index)
    negative_dc = pd.DataFrame({"value": np.full(len(index), -2.0)}, index=index)
    forecast = pd.DataFrame({"q50": np.full(len(index), 1.0)}, index=index)
    builder = make_perfect_foresight_builder(
        energy,
        energy,
        negative_dc,
        negative_dc,
    )

    output = builder(start, len(index), forecast, forecast, forecast, forecast)

    assert np.all(np.asarray(output["dc_low"]) == expected_dc)
    assert np.all(np.asarray(output["dc_high"]) == expected_dc)


@pytest.mark.parametrize(
    ("window_start", "expected_anchor"),
    [("2023-10-15", 0.0), ("2025-01-15", -2.0)],
)
def test_q50_anchor_applies_dc_price_regime(
    window_start: str,
    expected_anchor: float,
):
    start = pd.Timestamp(window_start, tz="UTC")
    index = pd.date_range(start, periods=8, freq="30min")
    energy_forecast = pd.DataFrame({"q50": np.full(len(index), 50.0)}, index=index)
    negative_dc_forecast = pd.DataFrame(
        {"q50": np.full(len(index), -2.0)},
        index=index,
    )

    def base_builder(
        window_start,
        horizon,
        da_fc_win,
        bm_fc_win,
        dc_low_fc_win,
        dc_high_fc_win,
        seed=None,
    ):
        del (
            window_start,
            da_fc_win,
            bm_fc_win,
            dc_low_fc_win,
            dc_high_fc_win,
        )
        return {
            "da": np.full((2, horizon), 50.0),
            "bm": np.full((2, horizon), 50.0),
            "dc_low": np.full((2, horizon), 5.0),
            "dc_high": np.full((2, horizon), 5.0),
            "seed": seed,
        }

    builder = make_q50_anchored_builder(base_builder, anchor_fraction=0.5)
    output = builder(
        start,
        len(index),
        energy_forecast,
        energy_forecast,
        negative_dc_forecast,
        negative_dc_forecast,
    )

    assert np.all(np.asarray(output["dc_low"])[0] == expected_anchor)
    assert np.all(np.asarray(output["dc_high"])[0] == expected_anchor)
    assert np.all(np.asarray(output["dc_low"])[1] == 5.0)
    assert np.all(np.asarray(output["dc_high"])[1] == 5.0)
