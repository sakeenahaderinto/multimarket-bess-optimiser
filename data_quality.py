"""Audit safe DA expansion from native hourly data to a 30-minute grid."""

from __future__ import annotations

import logging

import pandas as pd

from config import settings
from data_integrity import collapse_identical_duplicates

logger = logging.getLogger(__name__)
STUDY_START = "2022-01-01"
STUDY_END = "2025-12-31"
DA_FORWARD_FILL_LIMIT = 1


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = settings.data_dir / "raw" / "em_day_ahead"
    files = sorted(directory.glob("*.parquet"))
    if not files:
        processed = settings.data_dir / "processed" / "day_ahead_processed.parquet"
        if not processed.exists():
            raise FileNotFoundError("No raw or processed DA parquet file was found")
        frame = pd.read_parquet(processed)
    else:
        frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)

    if "datetime" not in frame.columns or "value" not in frame.columns:
        raise ValueError("DA data must contain datetime and value columns")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    raw = collapse_identical_duplicates(
        frame.set_index("datetime")["value"],
        "DA quality input",
    )

    canonical = pd.date_range(
        start=pd.Timestamp(STUDY_START, tz="UTC"),
        end=pd.Timestamp(f"{STUDY_END} 23:30", tz="UTC"),
        freq="30min",
    )
    raw_on_grid = raw.reindex(canonical)
    expanded = raw.resample("30min").ffill(limit=DA_FORWARD_FILL_LIMIT).reindex(canonical)

    audit = pd.DataFrame(index=canonical)
    audit.index.name = "timestamp"
    audit["raw_observed"] = raw_on_grid.notna()
    audit["frequency_aligned"] = (
        raw_on_grid.isna()
        & expanded.notna()
        & raw_on_grid.shift(1).notna()
    )
    audit["da_missing"] = expanded.isna()
    audit["da_imputed"] = False
    audit["safe_value"] = expanded

    summary_source = audit.copy()
    summary_source["date"] = summary_source.index.normalize()
    summary = summary_source.groupby("date").agg(
        n_raw_observed=("raw_observed", "sum"),
        n_frequency_aligned=("frequency_aligned", "sum"),
        n_missing=("da_missing", "sum"),
        n_imputed=("da_imputed", "sum"),
    )
    summary = summary.astype(int)
    summary["da_missing"] = summary["n_missing"] > 0
    summary["da_imputed"] = summary["n_imputed"] > 0

    output_directory = settings.data_dir / "quality"
    output_directory.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(output_directory / "da_quality_slots.parquet")
    summary.to_parquet(output_directory / "da_quality_by_date.parquet")
    summary[["da_missing", "da_imputed"]].to_parquet(
        output_directory / "da_imputed_by_date.parquet"
    )

    logger.info(
        "DA quality audit: %d missing slots across %d affected UTC dates",
        int(audit["da_missing"].sum()),
        int(summary["da_missing"].sum()),
    )
    return audit, summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_audit()
