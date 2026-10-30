"""Build the main leakage-safe 30-minute feature table."""

from __future__ import annotations

import logging

import pandas as pd

from config import settings
from data_integrity import collapse_identical_duplicates, select_latest_bm_vintages
from features.cyclical_time import add_cyclical_time
from features.lag_features import add_lag_features
from features.market_features import create_market_features
from features.point_in_time import build_bm_asof_features
from features.rolling_stats import add_rolling_features
from features.weather_features import add_weather_features
from market_calendar import expand_efa_block_values

logger = logging.getLogger(__name__)
STUDY_START = "2022-01-01"
STUDY_END = "2025-12-31"
DA_FORWARD_FILL_LIMIT = 1


def _deduplicate_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return collapse_identical_duplicates(df, name)  # type: ignore[return-value]


def _check_coverage(series: pd.Series, name: str, threshold: float = 0.05) -> None:
    missing_rate = float(series.isna().mean())
    if missing_rate > threshold:
        logger.warning(
            "%s has %.1f%% missing values after canonical alignment.",
            name,
            missing_rate * 100.0,
        )


def _dc_service_series(dc_df: pd.DataFrame, service: str) -> pd.Series:
    service_df = dc_df[dc_df["service"] == service]
    if service_df.empty:
        raise ValueError(f"No DC rows found for service {service!r}")
    series = service_df["clearing_price"].copy()
    if series.index.has_duplicates:
        conflicting = series.groupby(level=0).nunique(dropna=False)
        if (conflicting > 1).any():
            first = conflicting[conflicting > 1].index[0]
            raise ValueError(
                f"Conflicting {service} clearing prices at {first}. "
                "Resolve duplicate auction rows before feature construction."
            )
        series = series.groupby(level=0).last()
    expanded = expand_efa_block_values(series.sort_index())
    assert isinstance(expanded, pd.Series)
    return expanded


def _latest_bm_targets(bm_records: pd.DataFrame) -> pd.DataFrame:
    """Return the latest published BM target vintage for each settlement period."""
    comparison_columns = ["systemSellPrice"]
    if "totalAcceptedOfferVolume" in bm_records.columns:
        comparison_columns.append("totalAcceptedOfferVolume")
    return select_latest_bm_vintages(
        bm_records,
        "BMRS target data",
        comparison_columns=comparison_columns,
    )


def build_features() -> pd.DataFrame:
    """Build and save the main 30-minute target and feature table."""
    bm_df = pd.read_parquet(settings.data_dir / "processed" / "bmrs_processed.parquet")
    da_df = pd.read_parquet(settings.data_dir / "processed" / "day_ahead_processed.parquet")
    dc_df = pd.read_parquet(settings.data_dir / "processed" / "dc_auction_processed.parquet")
    weather_files = sorted((settings.data_dir / "raw" / "weather").glob("*.parquet"))
    if not weather_files:
        raise FileNotFoundError("No weather parquet files were found")
    weather_df = pd.concat(
        [pd.read_parquet(path) for path in weather_files],
        ignore_index=True,
    )

    da_df["datetime"] = pd.to_datetime(da_df["datetime"], utc=True)
    da_df = _deduplicate_index(da_df.set_index("datetime"), "DA")
    bm_df["startTime"] = pd.to_datetime(bm_df["startTime"], utc=True)
    bm_df["createdDateTime"] = pd.to_datetime(
        bm_df["createdDateTime"],
        utc=True,
        errors="coerce",
    )
    if bm_df["createdDateTime"].isna().any():
        raise ValueError("BMRS data contains invalid createdDateTime values")
    bm_records = bm_df.copy()
    bm_df = _latest_bm_targets(bm_records)
    weather_df["date"] = pd.to_datetime(weather_df["date"], utc=True)
    weather_df = _deduplicate_index(weather_df.set_index("date"), "weather")
    dc_df["delivery_start"] = pd.to_datetime(dc_df["delivery_start"], utc=True)
    dc_df = dc_df.set_index("delivery_start").sort_index()

    canonical_index = pd.date_range(
        start=pd.Timestamp(STUDY_START, tz="UTC"),
        end=pd.Timestamp(f"{STUDY_END} 23:30", tz="UTC"),
        freq="30min",
    )
    da_aligned = (
        da_df["value"]
        .resample("30min")
        .ffill(limit=DA_FORWARD_FILL_LIMIT)
        .reindex(canonical_index)
    )
    bm_price_aligned = bm_df["systemSellPrice"].reindex(canonical_index)
    bm_target_created_time_aligned = bm_df["createdDateTime"].reindex(canonical_index)
    accepted_offer_volume_aligned = bm_df["totalAcceptedOfferVolume"].reindex(
        canonical_index
    )
    dcl_aligned = _dc_service_series(dc_df, "DCL").reindex(canonical_index)
    dch_aligned = _dc_service_series(dc_df, "DCH").reindex(canonical_index)

    _check_coverage(da_aligned, "DA price")
    _check_coverage(bm_price_aligned, "BM system price")
    _check_coverage(accepted_offer_volume_aligned, "accepted offer volume")
    _check_coverage(dcl_aligned, "DC Low clearing price")
    _check_coverage(dch_aligned, "DC High clearing price")

    df = pd.DataFrame(index=canonical_index)
    df["price"] = da_aligned
    df["accepted_offer_volume"] = accepted_offer_volume_aligned
    df["bm_price"] = bm_price_aligned
    df["bm_target_created_time"] = bm_target_created_time_aligned
    df["dc_low_price"] = dcl_aligned
    df["dc_high_price"] = dch_aligned
    bm_asof = build_bm_asof_features(
        bm_records,
        canonical_index,
        resolution="30min",
        snapshot_lag=48,
        long_lags=(96, 144, 336),
        rolling_windows=(8, 48, 336),
    )
    df = add_cyclical_time(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_weather_features(df, weather_df)
    df = pd.concat(
        [
            df,
            bm_asof,
            create_market_features(
                da_df,
                weather_df,
                canonical_index,
                bm_asof,
                bm_snapshot_lag=48,
            ),
        ],
        axis=1,
    )

    output_path = settings.data_dir / "features" / "features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    logger.info("Feature table written to %s with shape %s", output_path, df.shape)
    return df
