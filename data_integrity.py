"""Shared validation for timestamp-indexed research inputs."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


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
