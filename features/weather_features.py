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


def add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Add the latest complete observed weather hour before 09:00 UTC D-1."""
    required = [
        "temperature_2m",
        "wind_speed_10m",
        "cloud_cover",
        "shortwave_radiation",
    ]
    _check_required_columns(weather_df, required, "weather_df")
    weather = collapse_identical_duplicates(weather_df, "weather features")
    snapshot_times = last_complete_hour_start_times(df.index)
    mapping = {
        "temperature": "temperature_2m",
        "wind_speed": "wind_speed_10m",
        "cloud_cover": "cloud_cover",
        "irradiance": "shortwave_radiation",
    }
    for output_column, source_column in mapping.items():
        df[output_column] = (
            weather[source_column]
            .reindex(snapshot_times, method="ffill")
            .to_numpy(dtype=float)
        )
    return df
