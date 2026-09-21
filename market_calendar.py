"""Shared market timing and EFA-block calendar utilities.

The project uses one conservative information timestamp for realised inputs:
09:00 UTC on D-1 for delivery day D. The N2EX day-ahead gate closes at
09:50 GMT. A 09:00 cutoff leaves a publication buffer and means that the
last complete half-hourly observation starts at 08:30 UTC.

Day-ahead prices are an exception to ordinary observation-time logic. A DA
price for delivery on D-1 was published before D-1 began, so a target-period
24-hour DA-price lag is known at the D-1 gate for delivery day D.
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime, time, timedelta

import pandas as pd

UTC = "UTC"
LONDON_TZ = "Europe/London"
N2EX_GATE_TIME_UTC = time(9, 50)
REALISED_DATA_CUTOFF_TIME_UTC = time(9, 0)
PERIOD = pd.Timedelta(minutes=30)


def ensure_utc_index(index: pd.DatetimeIndex, name: str = "index") -> pd.DatetimeIndex:
    """Return a UTC-aware copy of ``index`` and reject timezone-naive input."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} must be a pandas DatetimeIndex")
    if index.tz is None:
        raise ValueError(f"{name} must be timezone-aware")
    return index.tz_convert(UTC)


def ensure_utc_timestamp(value: object, name: str = "timestamp") -> pd.Timestamp:
    """Parse ``value`` and return a UTC-aware timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp


def _delivery_midnights(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return ensure_utc_index(index).normalize()


def n2ex_gate_times(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return 09:50 UTC on D-1 for every delivery timestamp in ``index``."""
    return (
        _delivery_midnights(index)
        - pd.Timedelta(days=1)
        + pd.Timedelta(
            hours=N2EX_GATE_TIME_UTC.hour,
            minutes=N2EX_GATE_TIME_UTC.minute,
        )
    )


def realised_data_cutoff_times(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return 09:00 UTC on D-1 for every delivery timestamp in ``index``."""
    return (
        _delivery_midnights(index)
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=REALISED_DATA_CUTOFF_TIME_UTC.hour)
    )


def realised_data_cutoff_time(delivery_timestamp: object) -> pd.Timestamp:
    """Scalar version of :func:`realised_data_cutoff_times`."""
    timestamp = ensure_utc_timestamp(delivery_timestamp)
    return realised_data_cutoff_times(pd.DatetimeIndex([timestamp]))[0]


def last_complete_half_hour_start_times(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return 08:30 UTC on D-1 for each delivery timestamp."""
    return realised_data_cutoff_times(index) - PERIOD


def last_complete_hour_start_times(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return 08:00 UTC on D-1 for each delivery timestamp."""
    return realised_data_cutoff_times(index) - pd.Timedelta(hours=1)


def _efa_key(timestamp: pd.Timestamp) -> tuple[date_type, int]:
    """Return ``(local block-start date, local block-start hour)``."""
    local = ensure_utc_timestamp(timestamp).tz_convert(LONDON_TZ)
    hour = local.hour
    if hour >= 23:
        return local.date(), 23
    if hour < 3:
        return local.date() - timedelta(days=1), 23
    return local.date(), 3 + 4 * ((hour - 3) // 4)


def efa_block_keys(index: pd.DatetimeIndex) -> pd.Index:
    """Return a stable EFA-block key for each timestamp in ``index``."""
    utc_index = ensure_utc_index(index)
    return pd.Index([_efa_key(ts) for ts in utc_index], dtype="object")


def efa_block_groups(window_start: object, horizon: int) -> list[tuple[int, int]]:
    """Return consecutive ``[start, end)`` period groups for real EFA blocks."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    start = ensure_utc_timestamp(window_start, "window_start")
    index = pd.date_range(start=start, periods=horizon, freq=PERIOD)
    keys = efa_block_keys(index)
    groups: list[tuple[int, int]] = []
    group_start = 0
    for position in range(1, horizon):
        if keys[position] != keys[group_start]:
            groups.append((group_start, position))
            group_start = position
    groups.append((group_start, horizon))
    return groups


def _local_boundary(block_date: date_type, hour: int) -> pd.Timestamp:
    naive = pd.Timestamp(datetime.combine(block_date, time(hour=hour)))
    return naive.tz_localize(LONDON_TZ).tz_convert(UTC)


def efa_block_bounds_utc(key: tuple[date_type, int]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the exact UTC start and end of an EFA block key."""
    block_date, start_hour = key
    start = _local_boundary(block_date, start_hour)
    if start_hour == 23:
        end_date = block_date + timedelta(days=1)
        end_hour = 3
    else:
        end_date = block_date
        end_hour = start_hour + 4
    end = _local_boundary(end_date, end_hour)
    return start, end


def is_complete_efa_group(index: pd.DatetimeIndex, start: int, end: int) -> bool:
    """Return whether ``index[start:end]`` covers one full EFA block."""
    utc_index = ensure_utc_index(index)
    if not (0 <= start < end <= len(utc_index)):
        raise ValueError(f"Invalid group [{start}, {end}) for index length {len(index)}")
    key = _efa_key(utc_index[start])
    expected_start, expected_end = efa_block_bounds_utc(key)
    actual_start = utc_index[start]
    actual_end = utc_index[end - 1] + PERIOD
    return actual_start == expected_start and actual_end == expected_end


def aggregate_to_efa_blocks(
    values: pd.Series | pd.DataFrame,
    how: str = "mean",
) -> pd.Series | pd.DataFrame:
    """Aggregate a timezone-aware series or frame by actual local EFA blocks."""
    if values.empty:
        return values.copy()
    keys = efa_block_keys(values.index)
    grouped = values.groupby(keys, sort=True)
    if how == "mean":
        result = grouped.mean()
    elif how == "sum":
        result = grouped.sum(min_count=1)
    else:
        raise ValueError("how must be 'mean' or 'sum'")
    block_starts = [efa_block_bounds_utc(key)[0] for key in result.index]
    result.index = pd.DatetimeIndex(block_starts, name="efa_block_start")
    return result.sort_index()
