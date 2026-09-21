"""Leakage-safe rolling statistics for the 30-minute feature table."""

from __future__ import annotations

import pandas as pd

from market_calendar import last_complete_half_hour_start_times


def _check_required_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price and accepted-offer-volume rolling statistics."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    _check_required_columns(
        df,
        ["accepted_offer_volume", "price"],
        "add_rolling_features",
    )
    snapshot_times = last_complete_half_hour_start_times(df.index)
    for window in (8, 48, 336):
        price_history = df["price"].shift(48)
        df[f"price_roll_mean_{window}"] = price_history.rolling(
            window=window,
            min_periods=max(1, window // 2),
        ).mean()
        df[f"price_roll_std_{window}"] = price_history.rolling(
            window=window,
            min_periods=max(2, window // 2),
        ).std()

        volume_mean = df["accepted_offer_volume"].rolling(
            window=window,
            min_periods=max(1, window // 2),
        ).mean()
        volume_std = df["accepted_offer_volume"].rolling(
            window=window,
            min_periods=max(2, window // 2),
        ).std()
        df[f"accepted_offer_volume_roll_mean_{window}"] = (
            volume_mean.reindex(snapshot_times).to_numpy(dtype=float)
        )
        df[f"accepted_offer_volume_roll_std_{window}"] = (
            volume_std.reindex(snapshot_times).to_numpy(dtype=float)
        )
    return df
