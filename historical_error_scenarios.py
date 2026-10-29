"""Historical joint forecast-error-path scenario generator."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from baselines.scenarios import _block_avg_1d, _col
from excluded_dates import is_excluded
from market_calendar import (
    apply_dc_price_domain,
    last_complete_half_hour_start_times,
    realised_data_cutoff_time,
)

logger = logging.getLogger(__name__)
MIN_POOL_SIZE = 30


def _date_offset_minutes(value) -> int:
    timestamp = pd.Timestamp(value, tz="UTC").tz_convert("Europe/London")
    offset = timestamp.utcoffset()
    return int(offset.total_seconds() // 60) if offset is not None else 0


def _compute_da_vol_7d(da_actual: pd.DataFrame, dates: list) -> np.ndarray:
    """Compute trailing DA volatility from data available before each cutoff."""
    output = []
    price = _col(da_actual)
    for value in dates:
        delivery_index = pd.DatetimeIndex([pd.Timestamp(value, tz="UTC")])
        last_period = last_complete_half_hour_start_times(delivery_index)[0]
        lookback = pd.date_range(end=last_period, periods=7 * 48, freq="30min")
        window = price.reindex(lookback).dropna()
        output.append(float(window.std()) if len(window) >= 48 else np.nan)
    return np.asarray(output, dtype=float)


def build_error_path_pool(
    da_actual: pd.DataFrame,
    bm_actual: pd.DataFrame,
    dc_low_actual_30: pd.DataFrame,
    dc_high_actual_30: pd.DataFrame,
    da_oof: pd.DataFrame,
    bm_oof: pd.DataFrame,
    dc_low_oof: pd.DataFrame,
    dc_high_oof: pd.DataFrame,
) -> dict:
    """Build complete 48-period joint OOF residual paths."""
    publication_column = "bm_target_created_time"
    if publication_column not in bm_actual.columns:
        raise ValueError(
            f"BM actuals must include publication metadata in {publication_column!r}"
        )
    errors = {
        "da": (_col(da_actual) - da_oof["q50"]).dropna(),
        "bm": (_col(bm_actual) - bm_oof["q50"]).dropna(),
        "dc_low": (_col(dc_low_actual_30) - dc_low_oof["q50"]).dropna(),
        "dc_high": (_col(dc_high_actual_30) - dc_high_oof["q50"]).dropna(),
    }

    bm_publication = pd.to_datetime(
        bm_actual[publication_column],
        utc=True,
        errors="coerce",
    ).reindex(errors["bm"].index)
    if bm_publication.isna().any():
        first = bm_publication[bm_publication.isna()].index[0]
        raise ValueError(f"BM residual at {first} has no valid publication time")
    published_before_settlement = bm_publication < bm_publication.index
    if published_before_settlement.any():
        first = published_before_settlement[published_before_settlement].index[0]
        raise ValueError(f"BM publication time precedes settlement start at {first}")

    def complete_dates(series: pd.Series) -> set:
        counts = series.groupby(series.index.date).size()
        return set(counts[counts == 48].index)

    common_dates = set.intersection(*(complete_dates(series) for series in errors.values()))
    valid_dates = sorted(value for value in common_dates if not is_excluded(value))
    if len(valid_dates) < MIN_POOL_SIZE:
        raise ValueError(
            f"Only {len(valid_dates)} complete error days are available. "
            f"At least {MIN_POOL_SIZE} are required."
        )

    def stack(series: pd.Series) -> np.ndarray:
        rows = []
        for value in valid_dates:
            row = series[series.index.date == value].sort_index().to_numpy(dtype=float)
            if row.shape != (48,):
                raise RuntimeError(f"Unexpected path shape {row.shape} for {value}")
            rows.append(row)
        return np.stack(rows)

    da_paths = stack(errors["da"])
    bm_paths = stack(errors["bm"])
    raw_dcl = stack(errors["dc_low"])
    raw_dch = stack(errors["dc_high"])
    dcl_paths = np.stack(
        [
            _block_avg_1d(row, 48, pd.Timestamp(value, tz="UTC"))
            for row, value in zip(raw_dcl, valid_dates, strict=True)
        ]
    )
    dch_paths = np.stack(
        [
            _block_avg_1d(row, 48, pd.Timestamp(value, tz="UTC"))
            for row, value in zip(raw_dch, valid_dates, strict=True)
        ]
    )
    bm_available_time = pd.DatetimeIndex(
        [
            bm_publication[bm_publication.index.date == value].max()
            for value in valid_dates
        ]
    )

    return {
        "dates": np.asarray(valid_dates, dtype=object),
        "da": da_paths,
        "bm": bm_paths,
        "dc_low": dcl_paths,
        "dc_high": dch_paths,
        "bm_available_time": bm_available_time,
        "da_vol_7d": _compute_da_vol_7d(da_actual, valid_dates),
        "utc_offset_minutes": np.asarray(
            [_date_offset_minutes(value) for value in valid_dates],
            dtype=int,
        ),
    }


def make_historical_error_scenario_builder(
    error_pool: dict,
    n: int = 20,
    da_actual: pd.DataFrame | None = None,
    recency_halflife: float = 90.0,
    error_cap_pct: float | None = None,
    match_dst_regime: bool = True,
):
    """Return a leakage-safe historical error-path scenario builder.

    Residual paths are uncensored by default. ``error_cap_pct`` enables an
    explicit absolute-residual winsorisation sensitivity when set in (0, 1).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if recency_halflife <= 0:
        raise ValueError("recency_halflife must be positive")
    if error_cap_pct is not None and not 0.0 < error_cap_pct < 1.0:
        raise ValueError("error_cap_pct must be strictly between 0 and 1, or None")

    pool_dates = np.asarray(error_pool["dates"], dtype=object)
    pool_volumes = np.asarray(error_pool.get("da_vol_7d"), dtype=float)
    pool_offsets = np.asarray(
        error_pool.get(
            "utc_offset_minutes",
            [_date_offset_minutes(value) for value in pool_dates],
        ),
        dtype=int,
    )
    if "bm_available_time" not in error_pool:
        raise ValueError("Historical error pool is missing 'bm_available_time'")
    pool_availability = pd.to_datetime(
        error_pool["bm_available_time"],
        utc=True,
        errors="coerce",
    )
    if len(pool_availability) != len(pool_dates) or pool_availability.isna().any():
        raise ValueError("Historical error pool has invalid BM availability metadata")
    path_end_times = pd.to_datetime(pool_dates, utc=True) + pd.Timedelta(hours=23, minutes=30)
    if (pool_availability < path_end_times).any():
        raise ValueError("Historical error pool marks a BM path available before it settled")
    use_regime_weights = da_actual is not None and pool_volumes.size == len(pool_dates)

    def builder(
        window_start,
        horizon,
        da_fc_win,
        bm_fc_win,
        dc_low_fc_win,
        dc_high_fc_win,
        seed=None,
    ):
        if horizon % 48 != 0:
            raise ValueError("Historical error scenarios require a multiple of 48 periods")

        target_timestamp = pd.Timestamp(window_start)
        target_date = target_timestamp.date()
        delivery_index = pd.date_range(
            target_timestamp,
            periods=horizon,
            freq="30min",
        )
        information_cutoff = realised_data_cutoff_time(target_timestamp)
        available = np.where(pool_availability <= information_cutoff)[0]

        if match_dst_regime:
            target_offset = _date_offset_minutes(target_date)
            available = available[pool_offsets[available] == target_offset]

        if len(available) < MIN_POOL_SIZE:
            raise ValueError(
                f"Only {len(available)} eligible historical error days exist before "
                f"{target_date}. At least {MIN_POOL_SIZE} are required."
            )

        caps: dict[str, float | None] = {}
        for market in ("da", "bm", "dc_low", "dc_high"):
            if error_cap_pct is not None:
                caps[market] = float(
                    np.percentile(
                        np.abs(np.asarray(error_pool[market])[available]),
                        error_cap_pct * 100,
                    )
                )
            else:
                caps[market] = None

        weights = None
        if use_regime_weights:
            target_day_index = pd.DatetimeIndex([pd.Timestamp(target_date, tz="UTC")])
            last_period = last_complete_half_hour_start_times(target_day_index)[0]
            lookback = pd.date_range(end=last_period, periods=7 * 48, freq="30min")
            target_window = _col(da_actual).reindex(lookback).dropna()
            target_volatility = (
                float(target_window.std()) if len(target_window) >= 48 else np.nan
            )

            available_volumes = pool_volumes[available]
            valid_volumes = available_volumes[np.isfinite(available_volumes)]
            if np.isfinite(target_volatility) and len(valid_volumes) >= 2:
                pairwise = np.abs(valid_volumes[:, None] - valid_volumes[None, :])
                positive = pairwise[pairwise > 0]
                bandwidth = float(np.median(positive)) if positive.size else 10.0
                bandwidth = max(bandwidth, 1e-6)

                similarity = np.where(
                    np.isfinite(available_volumes),
                    np.exp(-np.abs(available_volumes - target_volatility) / bandwidth),
                    1.0,
                )
                days_ago = np.asarray(
                    [(target_date - value).days for value in pool_dates[available]],
                    dtype=float,
                )
                recency = np.exp(-np.log(2.0) * days_ago / recency_halflife)
                raw = similarity * recency
                if raw.sum() > 0:
                    raw = raw / raw.sum()
                    effective_sample_size = 1.0 / float(np.sum(raw**2))
                    threshold = min(float(n), 0.5 * len(available))
                    if effective_sample_size < threshold:
                        blend = effective_sample_size / threshold
                        uniform = np.full(len(available), 1.0 / len(available))
                        raw = blend * raw + (1.0 - blend) * uniform
                    weights = raw / raw.sum()

        rng = np.random.default_rng(seed)
        scenarios = {
            market: np.zeros((n, horizon), dtype=float)
            for market in ("da", "bm", "dc_low", "dc_high")
        }

        q50 = {
            "da": da_fc_win["q50"].iloc[:horizon].to_numpy(dtype=float),
            "bm": bm_fc_win["q50"].iloc[:horizon].to_numpy(dtype=float),
            "dc_low": dc_low_fc_win["q50"].iloc[:horizon].to_numpy(dtype=float),
            "dc_high": dc_high_fc_win["q50"].iloc[:horizon].to_numpy(dtype=float),
        }
        q50["dc_low"] = _block_avg_1d(q50["dc_low"], horizon, window_start)
        q50["dc_high"] = _block_avg_1d(q50["dc_high"], horizon, window_start)

        for day_number in range(horizon // 48):
            target_slice = slice(day_number * 48, (day_number + 1) * 48)
            sampled = rng.choice(available, size=n, replace=True, p=weights)

            for market in scenarios:
                residual = np.asarray(error_pool[market])[sampled]
                cap = caps[market]
                if cap is not None:
                    residual = np.clip(residual, -cap, cap)
                scenarios[market][:, target_slice] = (
                    q50[market][target_slice][np.newaxis, :] + residual
                )

        scenarios["dc_low"] = apply_dc_price_domain(
            scenarios["dc_low"],
            delivery_index,
        )
        scenarios["dc_high"] = apply_dc_price_domain(
            scenarios["dc_high"],
            delivery_index,
        )
        scenarios["seed"] = seed
        return scenarios

    builder.__name__ = "historical_error_scenario_builder"
    return builder
