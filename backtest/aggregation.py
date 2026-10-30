"""Auditable aggregation of repeated backtest runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

QUALITY_EXCLUSION_COLUMNS = ("da_missing", "da_imputed", "state_reset")


def clean_result_dates(frame: pd.DataFrame) -> set:
    """Return solved dates without known data-quality or continuity flags."""
    if "solve_failed" not in frame.columns:
        raise ValueError("Backtest result is missing the 'solve_failed' column")

    clean = ~frame["solve_failed"].fillna(True).astype(bool)
    for column in QUALITY_EXCLUSION_COLUMNS:
        if column in frame.columns:
            clean &= ~frame[column].fillna(False).astype(bool)
    return set(frame.index[clean])


def shared_clean_dates(results: Mapping[str, pd.DataFrame]) -> set:
    """Intersect clean dates across every supplied strategy result."""
    if not results:
        return set()

    date_sets = [clean_result_dates(frame) for frame in results.values()]
    common = date_sets[0].copy()
    for dates in date_sets[1:]:
        common &= dates
    return common


def aggregate_seed_results(
    frames: Sequence[pd.DataFrame],
    revenue_columns: Sequence[str],
    std_columns: Sequence[str] = ("net_revenue",),
) -> pd.DataFrame:
    """Average dates solved by every seed and retain all attempted dates.

    A date is not considered solved in the aggregate when it is absent or
    failed in any seed run. This prevents seed-specific failures from being
    hidden by an intersection of successful dates.
    """
    if not frames:
        return pd.DataFrame()

    for position, frame in enumerate(frames):
        missing = {"solve_failed", *revenue_columns} - set(frame.columns)
        if missing:
            raise ValueError(
                f"Seed frame {position} is missing columns: {sorted(missing)}"
            )
        if frame.index.has_duplicates:
            raise ValueError(f"Seed frame {position} contains duplicate dates")

    all_dates = sorted(set().union(*(set(frame.index) for frame in frames)))
    if not all_dates:
        return pd.DataFrame()
    index = pd.Index(all_dates, name=frames[0].index.name)
    aligned = [frame.reindex(index) for frame in frames]

    success = np.column_stack(
        [
            (~frame["solve_failed"].fillna(True).astype(bool)).to_numpy()
            for frame in aligned
        ]
    )
    success_count = success.sum(axis=1)
    all_succeeded = success_count == len(frames)

    result = aligned[0].copy()
    for column in revenue_columns:
        matrix = np.column_stack(
            [frame[column].to_numpy(dtype=float) for frame in aligned]
        )
        means = np.mean(matrix, axis=1)
        result[column] = np.where(all_succeeded, means, np.nan)
        if column in std_columns:
            deviations = (
                np.std(matrix, axis=1, ddof=1)
                if len(frames) > 1
                else np.zeros(len(index), dtype=float)
            )
            result[f"{column}_seed_std"] = np.where(
                all_succeeded,
                deviations,
                np.nan,
            )

    result["seed_run_count"] = len(frames)
    result["seed_success_count"] = success_count

    if len(frames) == 1:
        result["solve_failed"] = ~all_succeeded
    else:
        result["solve_failed"] = ~all_succeeded
        result["failure_reason"] = None
        result["failure_stage"] = None
        failed = ~all_succeeded
        result.loc[failed, "failure_reason"] = [
            f"Solved in {count} of {len(frames)} seed runs; "
            "inspect per-seed artifacts for underlying failures"
            for count in success_count[failed]
        ]
        result.loc[failed, "failure_stage"] = "seed_aggregation"

    for column in QUALITY_EXCLUSION_COLUMNS:
        if any(column in frame.columns for frame in aligned):
            flags = np.column_stack(
                [
                    frame[column].fillna(False).astype(bool).to_numpy()
                    if column in frame.columns
                    else np.zeros(len(index), dtype=bool)
                    for frame in aligned
                ]
            )
            result[column] = flags.any(axis=1)

    if "ending_soc" in result.columns:
        result.loc[~all_succeeded, "ending_soc"] = np.nan
    return result
