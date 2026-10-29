"""Native-hourly feature pipeline for the day-ahead price forecaster."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from features.point_in_time import build_bm_asof_features
from market_calendar import last_complete_hour_start_times

logger = logging.getLogger(__name__)
STUDY_START = "2022-01-01"
STUDY_END = "2025-12-31"


def _check_hourly_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    inferred = pd.infer_freq(index)
    if inferred not in ("h", "H", "1h", "1H"):
        raise ValueError(
            "An hourly DatetimeIndex is required. "
            f"Inferred frequency: {inferred!r}"
        )


def _complete_hour_sum(series: pd.Series) -> pd.Series:
    """Sum two half-hours and return NaN when either half-hour is missing."""
    resampler = series.sort_index().resample("1h", closed="left", label="left")
    values = resampler.sum(min_count=2)
    counts = resampler.count()
    return values.where(counts == 2)


def _complete_hour_mean(series: pd.Series) -> pd.Series:
    """Average two half-hours and return NaN when either is missing."""
    resampler = series.sort_index().resample("1h", closed="left", label="left")
    values = resampler.mean()
    counts = resampler.count()
    return values.where(counts == 2)


def _latest_bm_targets(records: pd.DataFrame) -> pd.DataFrame:
    """Select the final stored vintage only for realised hourly target fields."""
    required = {"startTime", "createdDateTime"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"BM data is missing columns: {sorted(missing)}")
    frame = records.copy()
    frame["startTime"] = pd.to_datetime(frame["startTime"], utc=True, errors="coerce")
    frame["createdDateTime"] = pd.to_datetime(
        frame["createdDateTime"],
        utc=True,
        errors="coerce",
    )
    if frame[["startTime", "createdDateTime"]].isna().any().any():
        raise ValueError("BM data contains invalid event or publication timestamps")
    if (frame["createdDateTime"] < frame["startTime"]).any():
        raise ValueError("BM data contains publication times before settlement start")
    return (
        frame.sort_values(["startTime", "createdDateTime"])
        .drop_duplicates(subset="startTime", keep="last")
        .set_index("startTime")
        .sort_index()
    )


def add_cyclical_time_hourly(df: pd.DataFrame) -> pd.DataFrame:
    _check_hourly_index(df.index)
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    df["year_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.2425)
    df["year_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.2425)
    return df


def add_lag_features_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Add hourly DA-price lags, which are published before delivery.

    BM-derived lags are added separately from publication-vintage-aware data.
    """
    _check_hourly_index(df.index)
    required = ("price",)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for lag in (24, 48, 72, 168):
        df[f"price_lag_{lag}"] = df["price"].shift(lag)
    return df


def add_rolling_features_hourly(df: pd.DataFrame) -> pd.DataFrame:
    required = ("price",)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for window in (4, 24, 168):
        price_history = df["price"].shift(24)
        df[f"price_roll_mean_{window}"] = price_history.rolling(
            window=window,
            min_periods=max(1, window // 2),
        ).mean()
        df[f"price_roll_std_{window}"] = price_history.rolling(
            window=window,
            min_periods=max(2, window // 2),
        ).std()
    return df


def add_weather_features_hourly(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    required = ("wind_speed_10m", "cloud_cover")
    missing = [column for column in required if column not in weather_df.columns]
    if missing:
        raise ValueError(f"Weather data is missing columns: {missing}")
    snapshot_times = last_complete_hour_start_times(df.index)
    weather = weather_df.sort_index()
    weather = weather[~weather.index.duplicated(keep="last")]
    df["wind_speed"] = (
        weather["wind_speed_10m"]
        .reindex(snapshot_times, method="ffill")
        .to_numpy(dtype=float)
    )
    df["cloud_cover"] = (
        weather["cloud_cover"]
        .reindex(snapshot_times, method="ffill")
        .to_numpy(dtype=float)
    )
    return df


def add_market_features_hourly(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
    bm_asof: pd.DataFrame,
) -> pd.DataFrame:
    if "price" not in df.columns:
        raise ValueError("The hourly DA price column is missing")
    snapshot_times = last_complete_hour_start_times(df.index)
    weather = weather_df.sort_index()
    weather = weather[~weather.index.duplicated(keep="last")]
    wind = weather["wind_speed_10m"]
    wind_now = wind.reindex(snapshot_times, method="ffill").to_numpy(dtype=float)
    wind_previous = wind.reindex(
        snapshot_times - pd.Timedelta(hours=1),
        method="ffill",
    ).to_numpy(dtype=float)
    df["wind_change_lag_24"] = wind_now - wind_previous
    snapshot_starts = pd.DatetimeIndex(bm_asof["bm_snapshot_start_time"])
    da_at_snapshot = df["price"].reindex(snapshot_starts).to_numpy(dtype=float)
    bm_at_snapshot = bm_asof["bm_price_lag_24"].to_numpy(dtype=float)
    df["spread_lag_24"] = bm_at_snapshot - da_at_snapshot
    return df


def build_da_hourly_features(
    da_df_hourly: pd.DataFrame,
    bm_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    canonical_index = pd.date_range(
        start=pd.Timestamp(STUDY_START, tz="UTC"),
        end=pd.Timestamp(f"{STUDY_END} 23:00", tz="UTC"),
        freq="1h",
    )
    da_source = da_df_hourly["value"].sort_index()
    if da_source.index.has_duplicates:
        da_source = da_source.groupby(level=0).last()
    da_aligned = da_source.reindex(canonical_index)
    bm_records = bm_df.copy()
    if "startTime" not in bm_records.columns:
        if not isinstance(bm_records.index, pd.DatetimeIndex):
            raise ValueError("BM data requires a startTime column or DatetimeIndex")
        bm_records = bm_records.reset_index().rename(
            columns={bm_records.index.name or "index": "startTime"}
        )
    bm_records["startTime"] = pd.to_datetime(bm_records["startTime"], utc=True)
    bm_indexed = _latest_bm_targets(bm_records)
    accepted_offer_volume = _complete_hour_sum(
        bm_indexed["totalAcceptedOfferVolume"]
    ).reindex(canonical_index)
    missing_rate = float(da_aligned.isna().mean())
    if missing_rate > 0.05:
        logger.warning("Hourly DA price has %.1f%% missing values.", missing_rate * 100.0)
    df = pd.DataFrame(index=canonical_index)
    df["price"] = da_aligned
    df["accepted_offer_volume"] = accepted_offer_volume
    bm_asof = build_bm_asof_features(
        bm_records,
        canonical_index,
        resolution="1h",
        snapshot_lag=24,
        long_lags=(48, 72, 168),
        rolling_windows=(4, 24, 168),
    )
    df = add_cyclical_time_hourly(df)
    df = add_lag_features_hourly(df)
    df = add_rolling_features_hourly(df)
    df = add_weather_features_hourly(df, weather_df)
    df = pd.concat([df, bm_asof], axis=1)
    df = add_market_features_hourly(df, weather_df, bm_asof)
    return df


def upsample_da_forecast(hourly_fc: pd.DataFrame) -> pd.DataFrame:
    """Duplicate every hourly forecast exactly across its two half-hours."""
    if not isinstance(hourly_fc.index, pd.DatetimeIndex):
        raise ValueError("hourly_fc must have a DatetimeIndex")
    if hourly_fc.index.tz is None:
        raise ValueError("hourly_fc index must be timezone-aware")
    if hourly_fc.empty:
        return hourly_fc.copy()
    required = {"q10", "q50", "q90"}
    missing = required - set(hourly_fc.columns)
    if missing:
        raise ValueError(f"hourly_fc is missing forecast columns: {sorted(missing)}")
    hourly = hourly_fc.sort_index()
    if hourly.index.has_duplicates:
        raise ValueError("hourly_fc contains duplicate timestamps")
    repeated = hourly.loc[hourly.index.repeat(2)].copy()
    new_index_values = []
    for timestamp in hourly.index:
        new_index_values.extend([timestamp, timestamp + pd.Timedelta(minutes=30)])
    repeated.index = pd.DatetimeIndex(new_index_values, name=hourly.index.name)
    return repeated
