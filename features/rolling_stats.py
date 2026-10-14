"""Leakage-safe rolling statistics for the 30-minute feature table."""

from __future__ import annotations

import pandas as pd


def _check_required_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add published-ahead DA-price rolling statistics.

    Accepted-offer-volume statistics are constructed point in time from BMRS
    publication vintages in ``features.point_in_time``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    _check_required_columns(
        df,
        ["price"],
        "add_rolling_features",
    )
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
    return df
