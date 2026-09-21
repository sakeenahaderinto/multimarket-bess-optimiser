"""Cross-market features observed before the common information cutoff."""

from __future__ import annotations

import pandas as pd

from market_calendar import (
    last_complete_half_hour_start_times,
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
    bm_df: pd.DataFrame,
    da_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    canonical_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create BM, wind-change and BM-minus-DA spread snapshot features."""
    _check_required_columns(
        bm_df,
        ["netImbalanceVolume", "systemSellPrice"],
        "bm_df",
    )
    _check_required_columns(da_df, ["value"], "da_df")
    _check_required_columns(weather_df, ["wind_speed_10m"], "weather_df")
    market_df = pd.DataFrame(index=canonical_index)
    half_hour_times = last_complete_half_hour_start_times(canonical_index)
    hour_times = last_complete_hour_start_times(canonical_index)

    bm_imbalance = bm_df["netImbalanceVolume"].reindex(canonical_index)
    market_df["bm_imbalance_volume_lag_48"] = (
        bm_imbalance.reindex(half_hour_times).to_numpy(dtype=float)
    )

    weather = weather_df.sort_index()
    weather = weather[~weather.index.duplicated(keep="last")]
    wind = weather["wind_speed_10m"]
    wind_now = wind.reindex(hour_times, method="ffill").to_numpy(dtype=float)
    wind_previous = wind.reindex(
        hour_times - pd.Timedelta(hours=1),
        method="ffill",
    ).to_numpy(dtype=float)
    market_df["wind_change_lag_48"] = wind_now - wind_previous

    da_price = (
        da_df["value"]
        .resample("30min")
        .ffill(limit=1)
        .reindex(canonical_index)
    )
    bm_price = bm_df["systemSellPrice"].reindex(canonical_index)
    da_at_cutoff = da_price.reindex(half_hour_times).to_numpy(dtype=float)
    bm_at_cutoff = bm_price.reindex(half_hour_times).to_numpy(dtype=float)
    market_df["spread_lag_48"] = bm_at_cutoff - da_at_cutoff
    return market_df
