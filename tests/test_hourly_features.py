import numpy as np
import pandas as pd
import pytest

from features.pipeline_da_hourly import (
    _complete_hour_mean,
    _complete_hour_sum,
    _latest_bm_targets,
    add_lag_features_hourly,
    upsample_da_forecast,
)
from train_da_hourly import _prepare_bm_records
from verify_inputs import REQUIRED, validate_one


def test_complete_hour_aggregations_require_two_half_hours():
    index = pd.date_range("2025-01-01", periods=6, freq="30min", tz="UTC")
    values = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 7.0], index=index)

    summed = _complete_hour_sum(values)
    averaged = _complete_hour_mean(values)

    assert summed.iloc[0] == 3.0
    assert np.isnan(summed.iloc[1])
    assert summed.iloc[2] == 12.0
    assert averaged.iloc[0] == 1.5
    assert np.isnan(averaged.iloc[1])
    assert averaged.iloc[2] == 6.0


def test_hourly_da_price_lag_retains_target_hour_shape():
    index = pd.date_range("2025-01-01", periods=24 * 5, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "accepted_offer_volume": np.arange(len(index), dtype=float),
            "price": 100.0 + np.arange(len(index), dtype=float),
        },
        index=index,
    )
    output = add_lag_features_hourly(frame.copy())

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


def test_hourly_entry_point_preserves_bm_publication_vintages():
    frame = pd.DataFrame(
        {
            "startTime": ["2025-01-01 00:00Z", "2025-01-01 00:00Z"],
            "createdDateTime": ["2025-01-01 00:20Z", "2025-01-02 10:00Z"],
            "systemSellPrice": [50.0, 500.0],
            "totalAcceptedOfferVolume": [1.0, 9.0],
            "netImbalanceVolume": [2.0, 8.0],
        }
    )

    records = _prepare_bm_records(frame)
    latest = _latest_bm_targets(records)

    assert len(records) == 2
    assert records["startTime"].nunique() == 1
    assert latest.loc[pd.Timestamp("2025-01-01 00:00Z"), "systemSellPrice"] == 500.0


def test_input_manifest_requires_bm_publication_metadata():
    assert "createdDateTime" in REQUIRED["bmrs"]["columns"]


@pytest.mark.parametrize(
    "created_time",
    ["not-a-time", "2024-12-31 23:59Z"],
)
def test_input_validation_rejects_invalid_bm_publication_time(
    tmp_path,
    created_time,
):
    path = tmp_path / "bm.parquet"
    pd.DataFrame(
        {
            "startTime": ["2025-01-01 00:00Z"],
            "createdDateTime": [created_time],
            "systemSellPrice": [50.0],
            "totalAcceptedOfferVolume": [1.0],
            "netImbalanceVolume": [2.0],
        }
    ).to_parquet(path)
    specification = {**REQUIRED["bmrs"], "path": path}

    with pytest.raises(ValueError, match="publication"):
        validate_one("bmrs", specification)
