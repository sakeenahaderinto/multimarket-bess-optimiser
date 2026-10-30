"""Scenario generation for the multi-market BESS model."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import settings
from data_integrity import (
    collapse_identical_duplicates,
    select_latest_bm_vintages,
    validate_quantile_forecast,
)
from market_calendar import (
    apply_dc_price_domain,
    efa_block_groups,
    ensure_utc_timestamp,
    realised_data_cutoff_time,
)

logger = logging.getLogger(__name__)
N_SCENARIOS = 20
_Z10 = float(norm.ppf(0.10))
_Z90 = float(norm.ppf(0.90))
BM_TARGET_AVAILABILITY_COL = "bm_target_created_time"
MIN_DEPENDENCE_OBSERVATIONS = 500
MAX_TEMPORAL_CORRELATION = 0.95


def _efa_block_groups(window_start: pd.Timestamp, horizon: int) -> list[tuple[int, int]]:
    """Compatibility wrapper around the shared market calendar."""
    return efa_block_groups(window_start, horizon)


def _default_correlation_matrix(use_spread: bool = False) -> np.ndarray:
    if use_spread:
        return np.array(
            [
                [1.00, 0.20, 0.20, 0.15],
                [0.20, 1.00, 0.15, 0.10],
                [0.20, 0.15, 1.00, 0.55],
                [0.15, 0.10, 0.55, 1.00],
            ],
            dtype=float,
        )
    return np.array(
        [
            [1.00, 0.65, 0.20, 0.15],
            [0.65, 1.00, 0.20, 0.15],
            [0.20, 0.20, 1.00, 0.55],
            [0.15, 0.15, 0.55, 1.00],
        ],
        dtype=float,
    )


def _default_dependence_parameters(
    use_spread: bool,
    reason: str,
) -> tuple[np.ndarray, float]:
    """Return documented fallback parameters for an expected lack of history."""
    logger.warning("Using default copula dependence because %s", reason)
    return _default_correlation_matrix(use_spread), 0.0


def _nearest_correlation(matrix: np.ndarray, minimum_eigenvalue: float = 1e-6) -> np.ndarray:
    """Return a symmetric positive-definite correlation matrix."""
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    values = np.clip(values, minimum_eigenvalue, None)
    positive = vectors @ np.diag(values) @ vectors.T
    scale = np.sqrt(np.diag(positive))
    correlation = positive / np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _read_forecast_q50(model_id: str) -> pd.Series | None:
    path = settings.data_dir / "forecasts" / f"{model_id}_oof_2022_2024.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "q50" not in frame.columns:
        return None
    index = pd.to_datetime(frame.index, utc=True)
    result = pd.Series(frame["q50"].to_numpy(dtype=float), index=index, name=model_id)
    result = collapse_identical_duplicates(result, f"{model_id} median forecast")
    if model_id == "da_hourly":
        result = result.resample("30min").ffill(limit=1)
    return result


def _dc_actual(service: str) -> pd.Series:
    path = settings.data_dir / "processed" / "dc_auction_processed.parquet"
    frame = pd.read_parquet(path)
    frame = frame[frame["service"] == service].copy()
    frame["delivery_start"] = pd.to_datetime(frame["delivery_start"], utc=True)
    series = frame.set_index("delivery_start")["clearing_price"].sort_index()
    if series.index.has_duplicates:
        unique_counts = series.groupby(level=0).nunique(dropna=False)
        if (unique_counts > 1).any():
            raise ValueError(f"Conflicting duplicate {service} clearing prices")
        series = series.groupby(level=0).last()
    return series.resample("30min").ffill(limit=7)


@lru_cache(maxsize=4)
def _historical_residual_frame(
    use_spread: bool,
    da_model_id: str,
) -> pd.DataFrame | None:
    """Load OOF residuals once for leakage-safe copula dependence estimation."""
    da_path = settings.data_dir / "processed" / "day_ahead_processed.parquet"
    bm_path = settings.data_dir / "processed" / "bmrs_processed.parquet"
    dc_path = settings.data_dir / "processed" / "dc_auction_processed.parquet"
    if not da_path.exists() or not bm_path.exists() or not dc_path.exists():
        return None

    da_q50 = _read_forecast_q50(da_model_id)
    bm_q50 = _read_forecast_q50("bm")
    dcl_q50 = _read_forecast_q50("dc_low")
    dch_q50 = _read_forecast_q50("dc_high")
    spread_q50 = _read_forecast_q50("spread") if use_spread else None

    required = [da_q50, dcl_q50, dch_q50, spread_q50 if use_spread else bm_q50]
    if any(series is None for series in required):
        return None

    da_frame = pd.read_parquet(da_path)
    da_frame["datetime"] = pd.to_datetime(da_frame["datetime"], utc=True)
    da_actual = collapse_identical_duplicates(
        da_frame.set_index("datetime")["value"],
        "dependence-estimation DA actuals",
    ).resample("30min").ffill(limit=1)

    bm_frame = pd.read_parquet(bm_path)
    bm_latest = select_latest_bm_vintages(
        bm_frame,
        "Dependence-estimation BM actuals",
        comparison_columns=["systemSellPrice"],
    )
    bm_actual = bm_latest["systemSellPrice"]
    bm_target_created_time = bm_latest["createdDateTime"]

    dcl_actual = _dc_actual("DCL")
    dch_actual = _dc_actual("DCH")

    actual_spread = bm_actual - da_actual
    second_actual = actual_spread if use_spread else bm_actual
    second_forecast = spread_q50 if use_spread else bm_q50
    second_name = "spread" if use_spread else "bm"

    residuals = pd.DataFrame(
        {
            "da": da_actual - da_q50,
            second_name: second_actual - second_forecast,
            "dc_low": dcl_actual - dcl_q50,
            "dc_high": dch_actual - dch_q50,
            BM_TARGET_AVAILABILITY_COL: bm_target_created_time,
        }
    ).dropna()
    return residuals.sort_index()


def _residual_history_available_by(
    residuals: pd.DataFrame,
    cutoff_date: object,
) -> pd.DataFrame:
    """Return residual observations whose final BM target was published by cutoff."""
    cutoff = ensure_utc_timestamp(cutoff_date, "cutoff_date")
    if BM_TARGET_AVAILABILITY_COL not in residuals.columns:
        raise ValueError(
            f"Residual history is missing {BM_TARGET_AVAILABILITY_COL!r}"
        )
    if not isinstance(residuals.index, pd.DatetimeIndex) or residuals.index.tz is None:
        raise ValueError("Residual history must use a timezone-aware DatetimeIndex")

    publication = pd.to_datetime(
        residuals[BM_TARGET_AVAILABILITY_COL],
        utc=True,
        errors="coerce",
    )
    if publication.isna().any():
        first = publication[publication.isna()].index[0]
        raise ValueError(f"Residual at {first} has no valid BM publication time")
    published_before_settlement = publication < residuals.index
    if published_before_settlement.any():
        first = published_before_settlement[published_before_settlement].index[0]
        raise ValueError(f"BM publication time precedes settlement start at {first}")

    residual_columns = [
        column
        for column in ("da", "bm", "spread", "dc_low", "dc_high")
        if column in residuals.columns
    ]
    if len(residual_columns) != 4 or not {"da", "dc_low", "dc_high"}.issubset(
        residual_columns
    ):
        raise ValueError("Residual history does not contain one complete market set")

    available = (residuals.index < cutoff) & publication.le(cutoff)
    return residuals.loc[available, residual_columns]


def _estimate_temporal_correlation_from_history(
    history: pd.DataFrame,
    minimum_pairs: int = MIN_DEPENDENCE_OBSERVATIONS,
) -> float:
    """Estimate a common lag-one Gaussian-copula correlation.

    Only exact adjacent half-hours are paired. This prevents gaps between OOF
    folds or missing observations from being treated as consecutive periods.
    A common coefficient preserves the supplied contemporaneous correlation
    matrix under the stationary AR(1) construction used by the sampler.
    """
    if minimum_pairs <= 0:
        raise ValueError("minimum_pairs must be positive")
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None:
        raise ValueError("Residual history must use a timezone-aware DatetimeIndex")
    if history.index.has_duplicates:
        raise ValueError("Residual history contains duplicate timestamps")
    if not history.index.is_monotonic_increasing:
        raise ValueError("Residual history must be sorted by timestamp")

    second_column = "spread" if "spread" in history.columns else "bm"
    required = {"da", second_column}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(
            f"Residual history is missing temporal columns: {sorted(missing)}"
        )

    consecutive = history.index.to_series().diff().eq(pd.Timedelta(minutes=30))
    correlations: list[float] = []
    pair_counts: dict[str, int] = {}
    for column in ("da", second_column):
        current = history[column]
        previous = current.shift(1)
        valid = consecutive & current.notna() & previous.notna()
        pair_counts[column] = int(valid.sum())
        if pair_counts[column] < minimum_pairs:
            continue
        spearman = current.loc[valid].corr(previous.loc[valid], method="spearman")
        if pd.notna(spearman):
            gaussian = 2.0 * np.sin(np.pi * float(spearman) / 6.0)
            correlations.append(gaussian)

    if not correlations:
        logger.warning(
            "Too few consecutive residual pairs for temporal dependence: %s. "
            "Using zero persistence.",
            pair_counts,
        )
        return 0.0

    estimate = float(np.median(correlations))
    clipped = float(
        np.clip(estimate, -MAX_TEMPORAL_CORRELATION, MAX_TEMPORAL_CORRELATION)
    )
    if clipped != estimate:
        logger.warning(
            "Clipped temporal correlation from %.3f to %.3f for stability",
            estimate,
            clipped,
        )
    logger.info(
        "Temporal residual correlation %.3f estimated from consecutive pairs %s",
        clipped,
        pair_counts,
    )
    return clipped


def _estimate_dependence_parameters(
    cutoff_date: str | pd.Timestamp = "2025-01-01",
    use_spread: bool = False,
    da_model_id: str = "da",
) -> tuple[np.ndarray, float]:
    """Estimate contemporaneous and lag-one copula dependence before cutoff."""
    cutoff = ensure_utc_timestamp(cutoff_date, "cutoff_date")
    residuals = _historical_residual_frame(use_spread, da_model_id)
    if residuals is None:
        return _default_dependence_parameters(
            use_spread,
            "OOF residual inputs are unavailable",
        )

    settlement_history_count = int((residuals.index < cutoff).sum())
    history = _residual_history_available_by(residuals, cutoff)
    excluded_unpublished = settlement_history_count - len(history)
    if excluded_unpublished:
        logger.info(
            "Excluded %d OOF residuals whose BM targets were unpublished at %s",
            excluded_unpublished,
            cutoff,
        )
    if len(history) < MIN_DEPENDENCE_OBSERVATIONS:
        return _default_dependence_parameters(
            use_spread,
            f"only {len(history)} publication-eligible residual observations exist "
            f"before {cutoff}; at least {MIN_DEPENDENCE_OBSERVATIONS} are required",
        )

    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        row, column = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(
            "Dependence residual history contains a non-finite value at "
            f"{history.index[row]} for {history.columns[column]!r}"
        )

    spearman = history.corr(method="spearman").to_numpy(dtype=float)
    if not np.isfinite(spearman).all():
        raise ValueError(
            "Dependence correlation is undefined; check residual columns for "
            "constant or otherwise degenerate values"
        )
    gaussian = 2.0 * np.sin(np.pi * spearman / 6.0)
    correlation = _nearest_correlation(gaussian)
    logger.info(
        "Residual correlation estimated from %d observations before %s",
        len(history),
        cutoff,
    )
    temporal_correlation = _estimate_temporal_correlation_from_history(history)
    return correlation, temporal_correlation


def _estimate_correlation_matrix(
    cutoff_date: str | pd.Timestamp = "2025-01-01",
    use_spread: bool = False,
    da_model_id: str = "da",
) -> np.ndarray:
    """Compatibility wrapper returning contemporaneous copula correlation."""
    return _estimate_dependence_parameters(
        cutoff_date=cutoff_date,
        use_spread=use_spread,
        da_model_id=da_model_id,
    )[0]


def _validate_forecast(frame: pd.DataFrame, name: str, horizon: int) -> None:
    validate_quantile_forecast(frame, f"{name} forecast")
    if len(frame) != horizon:
        raise ValueError(f"{name} forecast length is {len(frame)}, expected {horizon}")


def _block_average(values: np.ndarray, groups: list[tuple[int, int]]) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for start, end in groups:
        result[start:end] = float(np.mean(values[start:end]))
    return result


def _two_piece_normal_quantile(
    latent_normal: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    tail_probability: float = 0.005,
) -> np.ndarray:
    """Map normal scores to a distribution matching q10, q50 and q90."""
    if not 0.0 < tail_probability < 0.1:
        raise ValueError("tail_probability must be between 0 and 0.1")

    limit = abs(float(norm.ppf(tail_probability)))
    z = np.clip(latent_normal, -limit, limit)
    lower_scale = (q50 - q10) / abs(_Z10)
    upper_scale = (q90 - q50) / _Z90
    return q50 + np.where(z < 0.0, z * lower_scale, z * upper_scale)


def sample_scenarios_multimarket(
    da_fc: pd.DataFrame,
    bm_fc: pd.DataFrame,
    dc_low_fc: pd.DataFrame,
    dc_high_fc: pd.DataFrame,
    n: int = N_SCENARIOS,
    seed: int | None = None,
    corr_matrix: np.ndarray | None = None,
    spread_fc: pd.DataFrame | None = None,
    temporal_correlation: float = 0.0,
) -> dict[str, np.ndarray | int | None]:
    """Generate temporally persistent copula scenarios with calibrated marginals."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if da_fc.empty:
        raise ValueError("Forecast horizon is empty")
    if not np.isfinite(temporal_correlation) or not -1.0 < temporal_correlation < 1.0:
        raise ValueError(
            "temporal_correlation must be finite and strictly between -1 and 1"
        )

    horizon = len(da_fc)
    for frame, name in (
        (da_fc, "DA"),
        (bm_fc, "BM"),
        (dc_low_fc, "DC Low"),
        (dc_high_fc, "DC High"),
    ):
        _validate_forecast(frame, name, horizon)
    if spread_fc is not None:
        _validate_forecast(spread_fc, "spread", horizon)

    reference_index = da_fc.index
    for frame, name in (
        (bm_fc, "BM"),
        (dc_low_fc, "DC Low"),
        (dc_high_fc, "DC High"),
    ):
        if not frame.index.equals(reference_index):
            raise ValueError(f"{name} forecast index does not match the DA index")
    if spread_fc is not None and not spread_fc.index.equals(reference_index):
        raise ValueError("Spread forecast index does not match the DA index")

    use_spread = spread_fc is not None
    correlation = (
        _estimate_correlation_matrix(use_spread=use_spread)
        if corr_matrix is None
        else np.asarray(corr_matrix, dtype=float)
    )
    if correlation.shape != (4, 4):
        raise ValueError(f"corr_matrix must have shape (4, 4), got {correlation.shape}")
    correlation = _nearest_correlation(correlation)
    cholesky = np.linalg.cholesky(correlation)

    rng = np.random.default_rng(seed)
    independent = rng.standard_normal((n, horizon, 4))
    innovations = independent @ cholesky.T
    latent = innovations.copy()
    innovation_scale = float(np.sqrt(1.0 - temporal_correlation**2))
    for period in range(1, horizon):
        latent[:, period, :] = (
            temporal_correlation * latent[:, period - 1, :]
            + innovation_scale * innovations[:, period, :]
        )

    window_start = reference_index[0]
    groups = efa_block_groups(window_start, horizon)

    # DC clears by EFA block. Use one latent draw per block for each DC product.
    for dimension in (2, 3):
        for start, end in groups:
            latent[:, start:end, dimension] = latent[:, start : start + 1, dimension]

    second_frame = spread_fc if spread_fc is not None else bm_fc
    assert second_frame is not None

    quantiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "da": tuple(
            da_fc[column].to_numpy(dtype=float) for column in ("q10", "q50", "q90")
        ),
        "second": tuple(
            second_frame[column].to_numpy(dtype=float)
            for column in ("q10", "q50", "q90")
        ),
        "dc_low": tuple(
            _block_average(dc_low_fc[column].to_numpy(dtype=float), groups)
            for column in ("q10", "q50", "q90")
        ),
        "dc_high": tuple(
            _block_average(dc_high_fc[column].to_numpy(dtype=float), groups)
            for column in ("q10", "q50", "q90")
        ),
    }

    output: dict[str, np.ndarray | int | None] = {}
    for dimension, key in enumerate(("da", "second", "dc_low", "dc_high")):
        q10, q50, q90 = quantiles[key]
        output[key] = _two_piece_normal_quantile(
            latent[:, :, dimension],
            q10[np.newaxis, :],
            q50[np.newaxis, :],
            q90[np.newaxis, :],
        )

    output["bm"] = (
        np.asarray(output["da"]) + np.asarray(output["second"])
        if use_spread
        else np.asarray(output["second"])
    )
    output["dc_low"] = apply_dc_price_domain(output["dc_low"], reference_index)
    output["dc_high"] = apply_dc_price_domain(output["dc_high"], reference_index)
    del output["second"]
    output["seed"] = seed
    return output


def load_latest_forecast(model_id: str) -> pd.DataFrame:
    directory = settings.data_dir / "forecasts"
    paths = sorted(directory.glob(f"{model_id}_oos_*.parquet"))
    if not paths:
        paths = sorted(directory.glob(f"{model_id}_oof_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No forecast was found for {model_id}")
    frame = pd.read_parquet(paths[-1])
    frame.index = pd.to_datetime(frame.index, utc=True)
    required = ["q10", "q50", "q90"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{paths[-1]} is missing forecast columns: {missing}")
    forecast = collapse_identical_duplicates(
        frame[required],
        f"{model_id} forecast",
    )
    return validate_quantile_forecast(forecast, f"{model_id} forecast")


def build_scenarios(horizon: int = 48, seed: int | None = None) -> dict:
    da = load_latest_forecast("da").iloc[:horizon]
    bm = load_latest_forecast("bm").reindex(da.index)
    dc_low = load_latest_forecast("dc_low").reindex(da.index)
    dc_high = load_latest_forecast("dc_high").reindex(da.index)
    correlation, temporal_correlation = _estimate_dependence_parameters(
        cutoff_date=realised_data_cutoff_time(da.index[0])
    )
    return sample_scenarios_multimarket(
        da,
        bm,
        dc_low,
        dc_high,
        n=N_SCENARIOS,
        seed=seed,
        corr_matrix=correlation,
        temporal_correlation=temporal_correlation,
    )
