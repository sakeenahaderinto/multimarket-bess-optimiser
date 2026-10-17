import numpy as np
import pandas as pd
import pytest

import historical_error_scenarios as historical
from historical_error_scenarios import (
    build_error_path_pool,
    make_historical_error_scenario_builder,
)
from market_calendar import realised_data_cutoff_time


def forecast(index: pd.DatetimeIndex, value: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({"q50": np.full(len(index), value)}, index=index)


def test_error_pool_records_when_each_complete_bm_path_was_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(historical, "MIN_POOL_SIZE", 2)
    index = pd.date_range("2024-01-08", periods=96, freq="30min", tz="UTC")
    values = np.arange(len(index), dtype=float)
    actual = pd.DataFrame({"value": values}, index=index)
    bm_actual = actual.copy()
    bm_actual["bm_target_created_time"] = index + pd.Timedelta(hours=1)
    bm_actual.loc[index[-1], "bm_target_created_time"] = index[-1] + pd.Timedelta(days=2)
    zero_forecast = forecast(index)

    pool = build_error_path_pool(
        actual,
        bm_actual,
        actual,
        actual,
        zero_forecast,
        zero_forecast,
        zero_forecast,
        zero_forecast,
    )

    availability = pd.DatetimeIndex(pool["bm_available_time"])
    assert availability[0] == index[47] + pd.Timedelta(hours=1)
    assert availability[1] == index[-1] + pd.Timedelta(days=2)


def test_historical_sampler_excludes_path_unpublished_at_forecast_origin() -> None:
    pool_dates = np.asarray(
        [timestamp.date() for timestamp in pd.date_range("2024-01-01", periods=31, tz="UTC")],
        dtype=object,
    )
    target_start = pd.Timestamp("2024-03-05", tz="UTC")
    cutoff = realised_data_cutoff_time(target_start)
    path_ids = np.arange(31, dtype=float)[:, np.newaxis]
    paths = np.repeat(path_ids, 48, axis=1)
    pool = {
        "dates": pool_dates,
        "da": paths,
        "bm": paths,
        "dc_low": paths,
        "dc_high": paths,
        "da_vol_7d": np.full(31, np.nan),
        "utc_offset_minutes": np.zeros(31, dtype=int),
        "bm_available_time": pd.DatetimeIndex(
            [cutoff - pd.Timedelta(days=1)] * 30 + [cutoff + pd.Timedelta(hours=1)]
        ),
    }
    target_index = pd.date_range(target_start, periods=48, freq="30min")
    zero_forecast = forecast(target_index)
    builder = make_historical_error_scenario_builder(
        pool,
        n=500,
        error_cap_pct=1.0,
        match_dst_regime=False,
    )

    scenarios = builder(
        target_start,
        48,
        zero_forecast,
        zero_forecast,
        zero_forecast,
        zero_forecast,
        seed=42,
    )

    assert not np.any(np.asarray(scenarios["da"]) == 30.0)


def test_historical_sampler_requires_bm_availability_metadata() -> None:
    pool = {
        "dates": np.asarray([], dtype=object),
        "da_vol_7d": np.asarray([], dtype=float),
        "utc_offset_minutes": np.asarray([], dtype=int),
    }

    with pytest.raises(ValueError, match="missing 'bm_available_time'"):
        make_historical_error_scenario_builder(pool)


def test_historical_sampler_rejects_availability_before_path_settlement() -> None:
    pool = {
        "dates": np.asarray([pd.Timestamp("2024-01-01").date()], dtype=object),
        "da_vol_7d": np.asarray([np.nan]),
        "utc_offset_minutes": np.asarray([0]),
        "bm_available_time": pd.DatetimeIndex(["2024-01-01 12:00Z"]),
    }

    with pytest.raises(ValueError, match="available before it settled"):
        make_historical_error_scenario_builder(pool)
