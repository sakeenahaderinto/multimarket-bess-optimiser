"""Dates excluded from scenario pools and publication backtests."""

from __future__ import annotations

import pandas as pd

CLOCK_CHANGE_DATES = pd.to_datetime(
    [
        "2021-03-28",
        "2021-10-31",
        "2022-03-27",
        "2022-10-30",
        "2023-03-26",
        "2023-10-29",
        "2024-03-31",
        "2024-10-27",
        "2025-03-30",
        "2025-10-26",
    ]
).date

BM_ANOMALY_DATES = pd.to_datetime(
    [
        "2021-01-07",
        "2022-05-31",
        "2022-10-22",
        "2023-01-17",
        "2023-01-22",
        "2023-03-18",
    ]
).date

EXCLUDED_DATES = set(CLOCK_CHANGE_DATES) | set(BM_ANOMALY_DATES)


def is_excluded(value: object) -> bool:
    return pd.Timestamp(value).date() in EXCLUDED_DATES


def filter_valid_dates(dates) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(dates)
    mask = ~pd.Series(index.date, index=index).isin(EXCLUDED_DATES)
    return index[mask.to_numpy()]
