"""Manual leakage checks for the native-hourly DA feature pipeline."""

from __future__ import annotations

import pandas as pd

from config import settings
from features.pipeline_da_hourly import build_da_hourly_features


def _maximum_residual(actual: pd.Series, expected: pd.Series) -> float:
    aligned = pd.concat([actual.rename("actual"), expected.rename("expected")], axis=1).dropna()
    if aligned.empty:
        raise AssertionError("No observations were available for the alignment check")
    return float((aligned["actual"] - aligned["expected"]).abs().max())


def main() -> None:
    bm = pd.read_parquet(settings.data_dir / "processed" / "bmrs_processed.parquet")
    da = pd.read_parquet(settings.data_dir / "processed" / "day_ahead_processed.parquet")
    weather_files = sorted((settings.data_dir / "raw" / "weather").glob("*.parquet"))
    weather = pd.concat([pd.read_parquet(path) for path in weather_files], ignore_index=True)

    da["datetime"] = pd.to_datetime(da["datetime"], utc=True)
    da = da.set_index("datetime").sort_index()
    bm["startTime"] = pd.to_datetime(bm["startTime"], utc=True)
    bm = bm.set_index("startTime").sort_index()
    weather["date"] = pd.to_datetime(weather["date"], utc=True)
    weather = weather.set_index("date").sort_index()

    features = build_da_hourly_features(
        da,
        bm.reset_index(),
        weather,
    )
    # DA price lag retains target-hour shape because the entire previous-day DA
    # auction curve was cleared and published before the current delivery-day gate.
    expected_price = features["price"].shift(24)
    residual = _maximum_residual(features["price_lag_24"], expected_price)
    print(f"price_lag_24 maximum residual: {residual:.8f}")
    assert residual < 1e-6

    # Every BM-derived value must come from a vintage published by its origin.
    violations = (
        features["bm_feature_max_created_time"] > features["bm_feature_origin"]
    ).fillna(False)
    print(f"BM publication-origin violations: {int(violations.sum())}")
    assert not violations.any()

    # The spread combines the as-of BM snapshot with DA at the snapshot's true
    # settlement hour, rather than assuming an unavailable 08:00/08:30 value.
    snapshot_starts = pd.DatetimeIndex(features["bm_snapshot_start_time"])
    expected_spread = pd.Series(
        features["bm_price_lag_24"].to_numpy(float)
        - features["price"].reindex(snapshot_starts).to_numpy(float),
        index=features.index,
    )
    residual = _maximum_residual(features["spread_lag_24"], expected_spread)
    print(f"spread_lag_24 maximum residual: {residual:.8f}")
    assert residual < 1e-6

    # Every row for a delivery UTC day must use the same restricted snapshot.
    for column in (
        "accepted_offer_volume_lag_24",
        "bm_imbalance_volume_lag_24",
        "wind_change_lag_24",
        "spread_lag_24",
    ):
        maximum_unique = int(features[column].groupby(features.index.date).nunique(dropna=False).max())
        print(f"{column} maximum unique values within a day: {maximum_unique}")
        assert maximum_unique <= 1

    print("All hourly lag checks passed")


if __name__ == "__main__":
    main()
