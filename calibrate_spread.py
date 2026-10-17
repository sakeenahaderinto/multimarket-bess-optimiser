"""Leakage-safe post-hoc calibration for BM-minus-DA spread quantiles."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import _normalise_to_30min
from market_calendar import ensure_utc_timestamp, realised_data_cutoff_time
from run_backtest import _load_actual, _load_bm_actual, _load_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
DATA_DIR = Path("data")
TARGET_TAIL_RATE = 0.10
MIN_HISTORY_DAYS = 90


def _pinball(actual: np.ndarray, forecast: np.ndarray, quantile: float) -> float:
    error = actual - forecast
    return float(np.mean(np.where(error >= 0, quantile * error, (quantile - 1) * error)))


def _lower_scale(
    actual: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    target_tail_rate: float = TARGET_TAIL_RATE,
) -> float:
    """Find scale so the lower-tail miss rate is close to target."""
    width = np.maximum(q50 - q10, 0.0)
    low, high = 0.0, 8.0
    for _ in range(80):
        scale = (low + high) / 2.0
        lower = q50 - scale * width
        miss_rate = float(np.mean(actual < lower))
        if miss_rate > target_tail_rate:
            low = scale
        else:
            high = scale
    return (low + high) / 2.0


def _upper_scale(
    actual: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    target_tail_rate: float = TARGET_TAIL_RATE,
) -> float:
    """Find scale so the upper-tail miss rate is close to target."""
    width = np.maximum(q90 - q50, 0.0)
    low, high = 0.0, 8.0
    for _ in range(80):
        scale = (low + high) / 2.0
        upper = q50 + scale * width
        miss_rate = float(np.mean(actual > upper))
        if miss_rate > target_tail_rate:
            low = scale
        else:
            high = scale
    return (low + high) / 2.0


def find_tail_scales(
    actual: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> tuple[float, float]:
    arrays = [np.asarray(values, dtype=float) for values in (actual, q10, q50, q90)]
    finite = np.logical_and.reduce([np.isfinite(values) for values in arrays])
    if finite.sum() < 48:
        raise ValueError("At least 48 finite observations are required for calibration")
    a, lower, median, upper = [values[finite] for values in arrays]
    return _lower_scale(a, lower, median), _upper_scale(a, median, upper)


def apply_scales(frame: pd.DataFrame, lower_scale: float, upper_scale: float) -> pd.DataFrame:
    calibrated = frame.copy()
    lower_width = np.maximum(calibrated["q50"] - calibrated["q10"], 0.0)
    upper_width = np.maximum(calibrated["q90"] - calibrated["q50"], 0.0)
    calibrated["q10"] = calibrated["q50"] - lower_scale * lower_width
    calibrated["q90"] = calibrated["q50"] + upper_scale * upper_width
    return calibrated


def _aligned_calibration_history(
    forecasts: pd.DataFrame,
    actual: pd.Series,
    actual_available_time: pd.Series,
) -> pd.DataFrame:
    """Align spread outcomes, forecasts, and their BM publication timestamps."""
    if not isinstance(forecasts.index, pd.DatetimeIndex) or forecasts.index.tz is None:
        raise ValueError("Calibration forecasts must use a timezone-aware DatetimeIndex")
    if forecasts.index.has_duplicates:
        raise ValueError("Calibration forecasts contain duplicate timestamps")

    frame = pd.concat(
        [
            actual.rename("actual"),
            forecasts[["q10", "q50", "q90"]],
            actual_available_time.rename("actual_available_time"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        raise ValueError("No complete observations are available for calibration")
    frame = frame.sort_index()
    availability = pd.to_datetime(
        frame["actual_available_time"],
        utc=True,
        errors="coerce",
    )
    if availability.isna().any():
        first = availability[availability.isna()].index[0]
        raise ValueError(f"Calibration outcome at {first} has no valid publication time")
    published_before_settlement = availability < frame.index
    if published_before_settlement.any():
        first = published_before_settlement[published_before_settlement].index[0]
        raise ValueError(f"Calibration publication precedes settlement start at {first}")
    frame["actual_available_time"] = availability
    return frame


def calibration_history_available_by(
    forecasts: pd.DataFrame,
    actual: pd.Series,
    actual_available_time: pd.Series,
    information_cutoff: object,
) -> pd.DataFrame:
    """Return aligned calibration observations known by an information cutoff."""
    cutoff = ensure_utc_timestamp(information_cutoff, "information_cutoff")
    frame = _aligned_calibration_history(forecasts, actual, actual_available_time)
    eligible = (frame.index < cutoff) & frame["actual_available_time"].le(cutoff)
    return frame.loc[eligible]


def rolling_calibrate(
    forecasts: pd.DataFrame,
    actual: pd.Series,
    actual_available_time: pd.Series,
    min_history_days: int = MIN_HISTORY_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate each UTC date using only outcomes published by its origin."""
    frame = _aligned_calibration_history(forecasts, actual, actual_available_time)
    forecast = frame[["q10", "q50", "q90"]]
    result = forecast.copy()
    date_values = np.asarray(result.index.date, dtype=object)
    scale_rows = []

    for current_date in sorted(set(date_values)):
        current_midnight = pd.Timestamp(current_date, tz="UTC")
        information_cutoff = realised_data_cutoff_time(current_midnight)
        past = (result.index < current_midnight) & frame["actual_available_time"].le(
            information_cutoff
        )
        past_dates = len(set(date_values[past]))
        if past_dates < min_history_days:
            lower_scale = upper_scale = 1.0
        else:
            lower_scale, upper_scale = find_tail_scales(
                frame["actual"].to_numpy(dtype=float)[past],
                forecast["q10"].to_numpy(dtype=float)[past],
                forecast["q50"].to_numpy(dtype=float)[past],
                forecast["q90"].to_numpy(dtype=float)[past],
            )

        current = date_values == current_date
        lower_width = (
            forecast["q50"].to_numpy(dtype=float)[current]
            - forecast["q10"].to_numpy(dtype=float)[current]
        )
        upper_width = (
            forecast["q90"].to_numpy(dtype=float)[current]
            - forecast["q50"].to_numpy(dtype=float)[current]
        )
        result.loc[current, "q10"] = (
            forecast["q50"].to_numpy(dtype=float)[current] - lower_scale * lower_width
        )
        result.loc[current, "q90"] = (
            forecast["q50"].to_numpy(dtype=float)[current] + upper_scale * upper_width
        )
        scale_rows.append(
            {
                "date": current_date,
                "lower_scale": lower_scale,
                "upper_scale": upper_scale,
                "history_days": past_dates,
                "history_observations": int(past.sum()),
                "information_cutoff": information_cutoff,
            }
        )

    return result, pd.DataFrame(scale_rows).set_index("date")


def _report(label: str, actual: pd.Series, forecast: pd.DataFrame) -> None:
    aligned = pd.concat([actual.rename("actual"), forecast], axis=1).dropna()
    a = aligned["actual"].to_numpy(float)
    q10 = aligned["q10"].to_numpy(float)
    q90 = aligned["q90"].to_numpy(float)
    logger.info("%s lower miss: %.2f%%", label, 100 * np.mean(a < q10))
    logger.info("%s upper miss: %.2f%%", label, 100 * np.mean(a > q90))
    logger.info("%s coverage: %.2f%%", label, 100 * np.mean((a >= q10) & (a <= q90)))
    logger.info("%s pinball q10: %.4f", label, _pinball(a, q10, 0.10))
    logger.info("%s pinball q90: %.4f", label, _pinball(a, q90, 0.90))


def main() -> None:
    da = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    bm_frame = _load_bm_actual()
    bm = bm_frame["value"]
    bm_available_time = bm_frame["bm_target_created_time"]
    da_30 = _normalise_to_30min(da, "da_actual", fill_limit=1)["value"]
    spread_actual = (bm - da_30).dropna()

    oof = _load_forecast("spread", "cv")
    common = oof.index.intersection(spread_actual.index)
    oof = oof.reindex(common)
    actual_oof = spread_actual.reindex(common)
    _report("OOF before", actual_oof, oof)

    calibrated_oof, scales = rolling_calibrate(
        oof,
        actual_oof,
        bm_available_time,
    )
    _report("OOF rolling calibrated", actual_oof, calibrated_oof)

    output_directory = DATA_DIR / "forecasts"
    calibrated_oof.to_parquet(output_directory / "spread_calibrated_oof_2022_2024.parquet")
    scales.to_csv(output_directory / "spread_calibration_scales_by_date.csv")

    oos = _load_forecast("spread", "oos")
    if oos.empty:
        raise ValueError("OOS spread forecast is empty")
    oos_information_cutoff = realised_data_cutoff_time(oos.index.min())
    final_history = calibration_history_available_by(
        oof,
        actual_oof,
        bm_available_time,
        oos_information_cutoff,
    )
    lower_scale, upper_scale = find_tail_scales(
        final_history["actual"].to_numpy(float),
        final_history["q10"].to_numpy(float),
        final_history["q50"].to_numpy(float),
        final_history["q90"].to_numpy(float),
    )
    logger.info(
        "Final OOF scales at %s from %d published observations: lower %.5f, upper %.5f",
        oos_information_cutoff,
        len(final_history),
        lower_scale,
        upper_scale,
    )

    calibrated_oos = apply_scales(oos, lower_scale, upper_scale)
    calibrated_oos.to_parquet(output_directory / "spread_calibrated_oos_2025.parquet")


if __name__ == "__main__":
    main()
