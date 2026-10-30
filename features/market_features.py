"""Cross-market features observed before the common information cutoff."""

from __future__ import annotations

import pandas as pd

from features.weather_features import weather_at_exact_hours
from market_calendar import (
    last_complete_hour_start_times,
)


def _check_required_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def create_market_features(
    da_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    canonical_index: pd.DatetimeIndex,
    bm_asof: pd.DataFrame,
    bm_snapshot_lag: int,
) -> pd.DataFrame:
    """Create weather change and a publication-safe BM-minus-DA snapshot."""
    _check_required_columns(da_df, ["value"], "da_df")
    _check_required_columns(weather_df, ["wind_speed_10m"], "weather_df")
    _check_required_columns(
        bm_asof,
        ["bm_snapshot_start_time", f"bm_price_lag_{bm_snapshot_lag}"],
        "bm_asof",
    )
    market_df = pd.DataFrame(index=canonical_index)
    hour_times = last_complete_hour_start_times(canonical_index)

    wind_now = weather_at_exact_hours(
        weather_df,
        hour_times,
        ["wind_speed_10m"],
        "market weather features",
    )["wind_speed_10m"].to_numpy(dtype=float)
    wind_previous = weather_at_exact_hours(
        weather_df,
        hour_times - pd.Timedelta(hours=1),
        ["wind_speed_10m"],
        "market weather features",
    )["wind_speed_10m"].to_numpy(dtype=float)
    market_df["wind_change_lag_48"] = wind_now - wind_previous

    da_price = (
        da_df["value"]
        .resample("30min")
        .ffill(limit=1)
    )
    da_at_snapshot = da_price.reindex(
        pd.DatetimeIndex(bm_asof["bm_snapshot_start_time"])
    ).to_numpy(dtype=float)
    bm_at_snapshot = bm_asof[f"bm_price_lag_{bm_snapshot_lag}"].to_numpy(dtype=float)
    market_df[f"spread_lag_{bm_snapshot_lag}"] = bm_at_snapshot - da_at_snapshot
    return market_df
