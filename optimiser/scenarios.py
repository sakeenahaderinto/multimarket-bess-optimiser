"""Scenario generation for the multi-market BESS model."""

from __future__ import annotations

from functools import lru_cache
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import settings
from market_calendar import efa_block_groups, ensure_utc_timestamp

logger = logging.getLogger(__name__)
N_SCENARIOS = 20
_Z10 = float(norm.ppf(0.10))
_Z90 = float(norm.ppf(0.90))


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
    result = result[~result.index.duplicated(keep="last")].sort_index()
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
    da_actual = (
        da_frame.set_index("datetime")["value"]
        .sort_index()
        .groupby(level=0)
        .last()
        .resample("30min")
        .ffill(limit=1)
    )

    bm_frame = pd.read_parquet(bm_path)
    bm_frame["startTime"] = pd.to_datetime(bm_frame["startTime"], utc=True)
    bm_actual = (
        bm_frame.set_index("startTime")["systemSellPrice"]
        .sort_index()
        .groupby(level=0)
        .last()
    )

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
        }
    ).dropna()
    return residuals.sort_index()


def _estimate_correlation_matrix(
    cutoff_date: str | pd.Timestamp = "2025-01-01",
    use_spread: bool = False,
    da_model_id: str = "da",
) -> np.ndarray:
    """Estimate copula correlation from OOF forecast residuals before cutoff."""
    cutoff = ensure_utc_timestamp(cutoff_date, "cutoff_date")

    try:
        residuals = _historical_residual_frame(use_spread, da_model_id)
        if residuals is None:
            logger.warning("OOF residuals are unavailable. Using default correlation.")
            return _default_correlation_matrix(use_spread)

        history = residuals[residuals.index < cutoff]
        if len(history) < 500:
            logger.warning(
                "Only %d residual observations exist before %s. Using defaults.",
                len(history),
                cutoff,
            )
            return _default_correlation_matrix(use_spread)

        spearman = history.corr(method="spearman").to_numpy(dtype=float)
        gaussian = 2.0 * np.sin(np.pi * spearman / 6.0)
        correlation = _nearest_correlation(gaussian)
        logger.info(
            "Residual correlation estimated from %d observations before %s",
            len(history),
            cutoff,
        )
        return correlation
    except Exception as exc:
        logger.exception("Correlation estimation failed: %s", exc)
        return _default_correlation_matrix(use_spread)


def _validate_forecast(frame: pd.DataFrame, name: str, horizon: int) -> None:
    required = {"q10", "q50", "q90"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} forecast is missing columns: {sorted(missing)}")
    if len(frame) != horizon:
        raise ValueError(f"{name} forecast length is {len(frame)}, expected {horizon}")
    values = frame[["q10", "q50", "q90"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} forecast contains non-finite values")
    if np.any(values[:, 0] > values[:, 1]) or np.any(values[:, 1] > values[:, 2]):
        raise ValueError(f"{name} forecast contains crossed quantiles")


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
) -> dict[str, np.ndarray | int | None]:
    """Generate Gaussian-copula scenarios with calibrated marginal quantiles."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if da_fc.empty:
        raise ValueError("Forecast horizon is empty")

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
    latent = independent @ cholesky.T

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
    output["dc_low"] = np.maximum(np.asarray(output["dc_low"]), 0.0)
    output["dc_high"] = np.maximum(np.asarray(output["dc_high"]), 0.0)
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
    frame = pd.read_parquet(paths[-1]).sort_index()
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame[~frame.index.duplicated(keep="last")]


def build_scenarios(horizon: int = 48, seed: int | None = None) -> dict:
    da = load_latest_forecast("da").iloc[:horizon]
    bm = load_latest_forecast("bm").reindex(da.index)
    dc_low = load_latest_forecast("dc_low").reindex(da.index)
    dc_high = load_latest_forecast("dc_high").reindex(da.index)
    correlation = _estimate_correlation_matrix(cutoff_date=da.index[0])
    return sample_scenarios_multimarket(
        da,
        bm,
        dc_low,
        dc_high,
        n=N_SCENARIOS,
        seed=seed,
        corr_matrix=correlation,
    )
