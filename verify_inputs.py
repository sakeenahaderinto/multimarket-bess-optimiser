"""Validate controlled parquet inputs and print reproducibility hashes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {
    "bmrs": {
        "path": Path("data/processed/bmrs_processed.parquet"),
        "columns": {
            "startTime",
            "createdDateTime",
            "systemSellPrice",
            "totalAcceptedOfferVolume",
            "netImbalanceVolume",
        },
        "time": "startTime",
        "numeric": {
            "systemSellPrice",
            "totalAcceptedOfferVolume",
            "netImbalanceVolume",
        },
    },
    "day_ahead": {
        "path": Path("data/processed/day_ahead_processed.parquet"),
        "columns": {"datetime", "value"},
        "time": "datetime",
        "numeric": {"value"},
    },
    "dc": {
        "path": Path("data/processed/dc_auction_processed.parquet"),
        "columns": {"delivery_start", "service", "clearing_price"},
        "time": "delivery_start",
        "numeric": {"clearing_price"},
    },
    "weather": {
        "path": Path("data/raw/weather/weather_backfill.parquet"),
        "columns": {
            "date",
            "temperature_2m",
            "shortwave_radiation",
            "wind_speed_10m",
            "cloud_cover",
        },
        "time": "date",
        "numeric": {
            "temperature_2m",
            "shortwave_radiation",
            "wind_speed_10m",
            "cloud_cover",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_one(name: str, specification: dict) -> dict:
    path: Path = specification["path"]
    if not path.exists():
        raise FileNotFoundError(f"{name}: missing file {path}")

    frame = pd.read_parquet(path)
    missing = set(specification["columns"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name}: file contains no rows")

    time_column = specification["time"]
    timestamps = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{name}: {int(timestamps.isna().sum())} invalid timestamps")

    for column in specification["numeric"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = values.isna() & frame[column].notna()
        if bad.any():
            raise ValueError(f"{name}: column {column} has non-numeric values")
        finite = values.dropna().to_numpy(dtype=float)
        if finite.size and not np.isfinite(finite).all():
            raise ValueError(f"{name}: column {column} contains infinite values")

    if name == "dc":
        services = set(frame["service"].dropna().astype(str))
        absent = {"DCL", "DCH"} - services
        if absent:
            raise ValueError(f"dc: required services are missing: {sorted(absent)}")

    if name == "bmrs":
        publication = pd.to_datetime(
            frame["createdDateTime"],
            utc=True,
            errors="coerce",
        )
        if publication.isna().any():
            raise ValueError(
                "bmrs: "
                f"{int(publication.isna().sum())} invalid publication timestamps"
            )
        published_before_settlement = publication < timestamps
        if published_before_settlement.any():
            first = published_before_settlement[published_before_settlement].index[0]
            raise ValueError(
                "bmrs: publication time precedes settlement start "
                f"at row {first}"
            )

    duplicate_rows = int(timestamps.duplicated(keep=False).sum())
    return {
        "input": name,
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(frame),
        "start_utc": timestamps.min(),
        "end_utc": timestamps.max(),
        "duplicate_timestamp_rows": duplicate_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/input_manifest.csv",
        help="CSV path for the validated input manifest",
    )
    args = parser.parse_args()

    rows = [validate_one(name, specification) for name, specification in REQUIRED.items()]
    manifest = pd.DataFrame(rows).set_index("input")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output)
    print(manifest.to_string())
    print(f"\nInput manifest written to {output}")


if __name__ == "__main__":
    main()
