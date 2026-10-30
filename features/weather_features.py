"""Observed-weather features available before the common market cutoff."""

from __future__ import annotations

import pandas as pd

from data_integrity import collapse_identical_duplicates
from market_calendar import last_complete_hour_start_times


def _check_required_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def weather_at_exact_hours(
    weather_df: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    columns: list[str],
    source: str,
) -> pd.DataFrame:
    """Return exact hourly observations without carrying stale values forward."""
    _check_required_columns(weather_df, columns, source)
    if not isinstance(weather_df.index, pd.DatetimeIndex):
        raise TypeError(f"{source} must use a DatetimeIndex")
    if weather_df.index.tz is None:
        raise ValueError(f"{source} index must be timezone-aware")
    if not isinstance(timestamps, pd.DatetimeIndex) or timestamps.tz is None:
        raise ValueError("Weather lookup timestamps must be timezone-aware")

    weather = collapse_identical_duplicates(weather_df, source).copy()
    weather.index = weather.index.tz_convert("UTC")
    requested = timestamps.tz_convert("UTC")
    return weather.loc[:, columns].reindex(requested)


def add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Add the latest complete observed weather hour before 09:00 UTC D-1."""
    required = [
        "temperature_2m",
        "wind_speed_10m",
        "cloud_cover",
        "shortwave_radiation",
    ]
    snapshot_times = last_complete_hour_start_times(df.index)
    weather = weather_at_exact_hours(
        weather_df,
        snapshot_times,
        required,
        "weather features",
    )
    mapping = {
        "temperature": "temperature_2m",
        "wind_speed": "wind_speed_10m",
        "cloud_cover": "cloud_cover",
        "irradiance": "shortwave_radiation",
    }
    for output_column, source_column in mapping.items():
        df[output_column] = (
            weather[source_column].to_numpy(dtype=float)
        )
    return df
