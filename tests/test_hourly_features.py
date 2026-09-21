import numpy as np
import pandas as pd

from features.pipeline_da_hourly import (
    add_lag_features_hourly,
    complete_hour_mean,
    complete_hour_sum,
    upsample_da_forecast,
)


def test_complete_hour_aggregations_require_two_half_hours():
    index = pd.date_range("2025-01-01", periods=6, freq="30min", tz="UTC")
    values = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 7.0], index=index)

    summed = complete_hour_sum(values)
    averaged = complete_hour_mean(values)

    assert summed.iloc[0] == 3.0
    assert np.isnan(summed.iloc[1])
    assert summed.iloc[2] == 12.0
    assert averaged.iloc[0] == 1.5
    assert np.isnan(averaged.iloc[1])
    assert averaged.iloc[2] == 6.0


def test_restricted_hourly_feature_is_constant_within_delivery_day():
    index = pd.date_range("2025-01-01", periods=24 * 5, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "accepted_offer_volume": np.arange(len(index), dtype=float),
            "price": 100.0 + np.arange(len(index), dtype=float),
        },
        index=index,
    )
    output = add_lag_features_hourly(frame.copy())

    for day in sorted(set(output.index.date))[2:]:
        values = output.loc[str(day), "accepted_offer_volume_lag_24"].dropna()
        if len(values):
            assert values.nunique() == 1

    # DA price lag retains target-hour shape because the whole previous-day DA
    # auction curve is known before the current delivery-day gate.
    assert output["price_lag_24"].equals(output["price"].shift(24))


def test_hourly_forecast_upsampling_exactly_duplicates_each_hour():
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    hourly = pd.DataFrame(
        {
            "q10": [1.0, 2.0, 3.0],
            "q50": [4.0, 5.0, 6.0],
            "q90": [7.0, 8.0, 9.0],
        },
        index=index,
    )
    output = upsample_da_forecast(hourly)
    assert len(output) == 6
    for hour in range(3):
        pair = output.iloc[2 * hour : 2 * hour + 2]
        assert pair.nunique().max() == 1
