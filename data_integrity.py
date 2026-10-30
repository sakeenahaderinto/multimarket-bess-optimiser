"""Shared validation for timestamp-indexed research inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

QUANTILE_COLUMNS = ("q10", "q50", "q90")
BM_VINTAGE_KEY = ("startTime", "createdDateTime")


def collapse_bm_vintage_duplicates(
    records: pd.DataFrame,
    source: str,
    comparison_columns: Sequence[str],
) -> pd.DataFrame:
    """Collapse identical BM composite-key copies and reject conflicts."""
    required = {*BM_VINTAGE_KEY, *comparison_columns}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")

    frame = records.copy()
    for column in BM_VINTAGE_KEY:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if frame[list(BM_VINTAGE_KEY)].isna().any().any():
        raise ValueError(f"{source} contains invalid settlement or publication timestamps")
    if (frame["createdDateTime"] < frame["startTime"]).any():
        raise ValueError(f"{source} contains publication times before settlement start")

    duplicated = frame.duplicated(subset=list(BM_VINTAGE_KEY), keep=False)
    if duplicated.any():
        duplicate_rows = frame.loc[
            duplicated,
            [*BM_VINTAGE_KEY, *comparison_columns],
        ]
        counts = duplicate_rows.groupby(
            list(BM_VINTAGE_KEY),
            sort=False,
        )[list(comparison_columns)].nunique(dropna=False)
        conflicts = counts.gt(1)
        if conflicts.to_numpy().any():
            row, column = np.argwhere(conflicts.to_numpy())[0]
            settlement, publication = conflicts.index[row]
            field = conflicts.columns[column]
            raise ValueError(
                f"{source} contains conflicting duplicate BM vintage at "
                f"settlement {settlement}, publication {publication}, "
                f"column {field!r}"
            )
        frame = frame.drop_duplicates(subset=list(BM_VINTAGE_KEY), keep="first")

    return frame.sort_values(list(BM_VINTAGE_KEY)).reset_index(drop=True)


def select_latest_bm_vintages(
    records: pd.DataFrame,
    source: str,
    comparison_columns: Sequence[str],
) -> pd.DataFrame:
    """Return the latest validated BM vintage for each settlement period."""
    frame = collapse_bm_vintage_duplicates(records, source, comparison_columns)
    return (
        frame.drop_duplicates(subset="startTime", keep="last")
        .set_index("startTime")
        .sort_index()
    )


def validate_quantile_forecast(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate that a forecast is finite and ordered at every timestamp."""
    missing = [column for column in QUANTILE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing quantile columns: {missing}")

    try:
        values = frame.loc[:, list(QUANTILE_COLUMNS)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} quantiles must be numeric") from exc

    non_finite = ~np.isfinite(values)
    if non_finite.any():
        row, column = np.argwhere(non_finite)[0]
        raise ValueError(
            f"{source} contains a non-finite {QUANTILE_COLUMNS[column]} value "
            f"at {frame.index[row]}"
        )

    crossed = (values[:, 0] > values[:, 1]) | (values[:, 1] > values[:, 2])
    if crossed.any():
        row = int(np.flatnonzero(crossed)[0])
        q10, q50, q90 = values[row]
        raise ValueError(
            f"{source} contains crossed quantiles at {frame.index[row]}: "
            f"q10={q10}, q50={q50}, q90={q90}"
        )
    return frame


def collapse_identical_duplicates(
    values: pd.Series | pd.DataFrame,
    source: str,
    comparison_columns: Sequence[str] | None = None,
) -> pd.Series | pd.DataFrame:
    """Collapse identical timestamp duplicates and reject conflicting observations.

    Selecting a duplicate by row order is not reproducible when input file order changes.
    This helper permits repeated copies of the same observation, but requires callers to
    define an explicit revision or composite-key policy for genuinely different values.
    """
    if not values.index.has_duplicates:
        return values.sort_index()

    if isinstance(values, pd.Series):
        comparison = values.to_frame(name=values.name or "value")
    else:
        columns = list(comparison_columns) if comparison_columns is not None else list(values)
        missing = [column for column in columns if column not in values.columns]
        if missing:
            raise ValueError(
                f"{source} duplicate validation is missing columns: {missing}"
            )
        comparison = values[columns]

    duplicate_rows = comparison.loc[comparison.index.duplicated(keep=False)]
    unique_counts = duplicate_rows.groupby(level=0, sort=False).nunique(dropna=False)
    conflicts = unique_counts.gt(1)
    if conflicts.to_numpy().any():
        conflict_timestamp, conflict_column_position = conflicts.to_numpy().nonzero()
        timestamp = conflicts.index[conflict_timestamp[0]]
        column = conflicts.columns[conflict_column_position[0]]
        raise ValueError(
            f"{source} contains conflicting duplicate values at {timestamp} "
            f"for column {column!r}; resolve the source/revision policy explicitly"
        )

    return values.loc[~values.index.duplicated(keep="first")].sort_index()
