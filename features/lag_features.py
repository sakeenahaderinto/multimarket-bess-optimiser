"""Leakage-safe lag construction for the 30-minute feature table."""

from __future__ import annotations

import pandas as pd

from market_calendar import last_complete_half_hour_start_times

SAFE_LONG_LAGS = (96, 144, 336)


def _check_required_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _check_frequency(index: pd.DatetimeIndex) -> None:
    inferred = pd.infer_freq(index)
    if inferred not in ("30min", "30T"):
        raise ValueError(
            "add_lag_features expects a regular 30-minute DatetimeIndex. "
            f"Inferred frequency: {inferred!r}"
        )


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lags while respecting the 09:00 UTC D-1 information cutoff."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    _check_frequency(df.index)
    _check_required_columns(df, ["accepted_offer_volume", "price"], "add_lag_features")
    snapshot_times = last_complete_half_hour_start_times(df.index)

    published_ahead = ["price"]
    for column in ("dc_low_price", "dc_high_price"):
        if column in df.columns:
            published_ahead.append(column)
    for column in published_ahead:
        df[f"{column}_lag_48"] = df[column].shift(48)
        for lag in SAFE_LONG_LAGS:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)

    observed = ["accepted_offer_volume"]
    if "bm_price" in df.columns:
        observed.append("bm_price")
    for column in observed:
        df[f"{column}_lag_48"] = df[column].reindex(snapshot_times).to_numpy(dtype=float)
        for lag in SAFE_LONG_LAGS:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df
