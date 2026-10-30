"""Point-in-time construction of BM-derived forecasting features.

BMRS rows describe a settlement period (``startTime``) but are not necessarily
published at that time.  Forecast features must therefore be selected using
both the settlement timestamp and the row's publication timestamp
(``createdDateTime``).

The processed dataset currently contains the latest/final row for each period,
not a complete archive of every preliminary revision.  This module follows a
conservative rule: a stored row becomes usable only at its own
``createdDateTime``.  It never backdates a final value to ``startTime``.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_integrity import collapse_bm_vintage_duplicates
from market_calendar import realised_data_cutoff_times

BM_VALUE_COLUMNS = (
    "totalAcceptedOfferVolume",
    "systemSellPrice",
    "netImbalanceVolume",
)


@dataclass(frozen=True)
class _AvailableValue:
    value: float
    created_at: pd.Timestamp


def _normalise_bm_records(bm_df: pd.DataFrame) -> pd.DataFrame:
    """Return validated BM records with explicit UTC event and publication times."""
    frame = bm_df.copy()
    if "startTime" not in frame.columns:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("BM records require a startTime column or DatetimeIndex")
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "startTime"})

    required = {"startTime", "createdDateTime", *BM_VALUE_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Point-in-time BM features require publication metadata. "
            f"Missing columns: {sorted(missing)}"
        )

    frame = collapse_bm_vintage_duplicates(
        frame,
        "Point-in-time BM features",
        comparison_columns=BM_VALUE_COLUMNS,
    )
    return frame.sort_values(["createdDateTime", "startTime"]).reset_index(drop=True)


def _aggregate_value(
    active: dict[pd.Timestamp, pd.Series],
    start: pd.Timestamp,
    resolution: str,
    column: str,
) -> _AvailableValue | None:
    if resolution == "30min":
        row = active.get(start)
        if row is None or pd.isna(row[column]):
            return None
        return _AvailableValue(float(row[column]), row["createdDateTime"])

    if resolution != "1h":
        raise ValueError("resolution must be '30min' or '1h'")

    first = active.get(start)
    second = active.get(start + pd.Timedelta(minutes=30))
    if first is None or second is None:
        return None
    values = [first[column], second[column]]
    if any(pd.isna(value) for value in values):
        return None
    value = (
        float(np.mean(values))
        if column == "systemSellPrice"
        else float(np.sum(values))
    )
    return _AvailableValue(
        value,
        max(first["createdDateTime"], second["createdDateTime"]),
    )


def _latest_complete_start(
    active: dict[pd.Timestamp, pd.Series],
    sorted_starts: list[pd.Timestamp],
    origin: pd.Timestamp,
    resolution: str,
) -> pd.Timestamp | None:
    position = bisect_left(sorted_starts, origin) - 1
    while position >= 0:
        candidate = sorted_starts[position]
        if resolution == "30min":
            return candidate
        hour_start = candidate.floor("h")
        if (
            hour_start < origin
            and hour_start in active
            and hour_start + pd.Timedelta(minutes=30) in active
        ):
            return hour_start
        position -= 1
    return None


def _rolling_stat(
    active: dict[pd.Timestamp, pd.Series],
    end: pd.Timestamp | None,
    resolution: str,
    column: str,
    window: int,
    statistic: str,
) -> tuple[float, pd.Timestamp | None]:
    if end is None:
        return np.nan, None
    step = pd.Timedelta(minutes=30) if resolution == "30min" else pd.Timedelta(hours=1)
    available: list[_AvailableValue] = []
    for offset in range(window):
        value = _aggregate_value(active, end - offset * step, resolution, column)
        if value is not None:
            available.append(value)

    minimum = max(1, window // 2) if statistic == "mean" else max(2, window // 2)
    if len(available) < minimum:
        return np.nan, None
    values = np.asarray([item.value for item in available], dtype=float)
    result = float(values.mean()) if statistic == "mean" else float(values.std(ddof=1))
    return result, max(item.created_at for item in available)


def build_bm_asof_features(
    bm_df: pd.DataFrame,
    delivery_index: pd.DatetimeIndex,
    *,
    resolution: str,
    snapshot_lag: int,
    long_lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
) -> pd.DataFrame:
    """Build BM features using only vintages published by each D-1 origin.

    ``snapshot_lag`` retains the existing public feature name for compatibility,
    but its value is the latest complete observation actually published by the
    decision origin.  The lineage columns expose its true source and age.
    """
    if not isinstance(delivery_index, pd.DatetimeIndex) or delivery_index.tz is None:
        raise ValueError("delivery_index must be a timezone-aware DatetimeIndex")
    if resolution not in {"30min", "1h"}:
        raise ValueError("resolution must be '30min' or '1h'")

    records = _normalise_bm_records(bm_df)
    index = delivery_index.tz_convert("UTC")
    origins = realised_data_cutoff_times(index)
    unique_origins = pd.DatetimeIndex(origins.unique()).sort_values()

    columns = [
        f"accepted_offer_volume_lag_{snapshot_lag}",
        f"bm_price_lag_{snapshot_lag}",
        f"bm_imbalance_volume_lag_{snapshot_lag}",
        "bm_snapshot_start_time",
        "bm_snapshot_created_time",
        "bm_snapshot_age_hours",
        "bm_feature_origin",
        "bm_feature_max_created_time",
    ]
    for lag in long_lags:
        columns.extend(
            [
                f"accepted_offer_volume_lag_{lag}",
                f"bm_price_lag_{lag}",
            ]
        )
    for window in rolling_windows:
        columns.extend(
            [
                f"accepted_offer_volume_roll_mean_{window}",
                f"accepted_offer_volume_roll_std_{window}",
            ]
        )

    output = pd.DataFrame(index=index, columns=columns)
    active: dict[pd.Timestamp, pd.Series] = {}
    sorted_starts: list[pd.Timestamp] = []
    record_position = 0
    step = pd.Timedelta(minutes=30) if resolution == "30min" else pd.Timedelta(hours=1)

    for origin in unique_origins:
        while (
            record_position < len(records)
            and records.iloc[record_position]["createdDateTime"] <= origin
        ):
            row = records.iloc[record_position]
            start = row["startTime"]
            if start not in active:
                insort(sorted_starts, start)
            active[start] = row
            record_position += 1

        row_positions = np.flatnonzero(origins == origin)
        snapshot_start = _latest_complete_start(
            active,
            sorted_starts,
            origin,
            resolution,
        )
        snapshot_values = {
            column: (
                _aggregate_value(active, snapshot_start, resolution, column)
                if snapshot_start is not None
                else None
            )
            for column in BM_VALUE_COLUMNS
        }

        rolling_values: dict[tuple[int, str], tuple[float, pd.Timestamp | None]] = {}
        for window in rolling_windows:
            for statistic in ("mean", "std"):
                rolling_values[(window, statistic)] = _rolling_stat(
                    active,
                    snapshot_start,
                    resolution,
                    "totalAcceptedOfferVolume",
                    window,
                    statistic,
                )

        for position in row_positions:
            timestamp = index[position]
            max_created: list[pd.Timestamp] = []
            row_output: dict[str, object] = {
                "bm_feature_origin": origin,
                "bm_snapshot_start_time": snapshot_start,
            }
            snapshot_mapping = {
                f"accepted_offer_volume_lag_{snapshot_lag}": "totalAcceptedOfferVolume",
                f"bm_price_lag_{snapshot_lag}": "systemSellPrice",
                f"bm_imbalance_volume_lag_{snapshot_lag}": "netImbalanceVolume",
            }
            for output_column, source_column in snapshot_mapping.items():
                item = snapshot_values[source_column]
                row_output[output_column] = item.value if item is not None else np.nan
                if item is not None:
                    max_created.append(item.created_at)

            snapshot_created = (
                max(
                    item.created_at
                    for item in snapshot_values.values()
                    if item is not None
                )
                if any(item is not None for item in snapshot_values.values())
                else pd.NaT
            )
            row_output["bm_snapshot_created_time"] = snapshot_created
            row_output["bm_snapshot_age_hours"] = (
                (origin - snapshot_start).total_seconds() / 3600.0
                if snapshot_start is not None
                else np.nan
            )

            for lag in long_lags:
                source_start = timestamp - lag * step
                for output_prefix, source_column in (
                    ("accepted_offer_volume", "totalAcceptedOfferVolume"),
                    ("bm_price", "systemSellPrice"),
                ):
                    item = _aggregate_value(active, source_start, resolution, source_column)
                    row_output[f"{output_prefix}_lag_{lag}"] = (
                        item.value if item is not None else np.nan
                    )
                    if item is not None:
                        max_created.append(item.created_at)

            for window in rolling_windows:
                for statistic in ("mean", "std"):
                    value, created_at = rolling_values[(window, statistic)]
                    row_output[
                        f"accepted_offer_volume_roll_{statistic}_{window}"
                    ] = value
                    if created_at is not None:
                        max_created.append(created_at)

            row_output["bm_feature_max_created_time"] = (
                max(max_created) if max_created else pd.NaT
            )
            for column, value in row_output.items():
                output.iat[position, output.columns.get_loc(column)] = value

    for column in output.columns:
        if column.endswith("_time") or column == "bm_feature_origin":
            output[column] = pd.to_datetime(output[column], utc=True)
        elif column not in {"bm_snapshot_start_time"}:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    violation = output["bm_feature_max_created_time"] > output["bm_feature_origin"]
    if violation.fillna(False).any():
        raise RuntimeError("A BM feature used a vintage published after its decision origin")
    return output
