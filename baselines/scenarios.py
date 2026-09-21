"""Deterministic scenario builders used as backtest baselines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from market_calendar import efa_block_groups

ScenarioBuilder = Callable[..., dict]


def _col(df: pd.DataFrame) -> pd.Series:
    if "value" in df.columns:
        return df["value"]
    if df.shape[1] != 1:
        raise ValueError(
            "Actual-price frame must have a 'value' column or exactly one column"
        )
    return df.iloc[:, 0]


def _block_avg_1d(
    values: np.ndarray,
    horizon: int,
    window_start: pd.Timestamp | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (horizon,):
        raise ValueError(f"Expected shape {(horizon,)}, got {array.shape}")

    groups = (
        efa_block_groups(window_start, horizon)
        if window_start is not None
        else [(start, min(start + 8, horizon)) for start in range(0, horizon, 8)]
    )
    result = np.empty_like(array)
    for start, end in groups:
        result[start:end] = float(np.mean(array[start:end]))
    return result


def _q50(frame: pd.DataFrame, name: str, horizon: int) -> np.ndarray:
    if "q50" not in frame.columns:
        raise ValueError(f"{name} forecast does not contain q50")
    if len(frame) < horizon:
        raise ValueError(f"{name} forecast has {len(frame)} rows, expected {horizon}")
    values = frame["q50"].iloc[:horizon].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} q50 contains non-finite values")
    return values


def build_median_scenario(
    window_start: pd.Timestamp,
    horizon: int,
    da_fc_win: pd.DataFrame,
    bm_fc_win: pd.DataFrame,
    dc_low_fc_win: pd.DataFrame,
    dc_high_fc_win: pd.DataFrame,
    seed=None,
) -> dict:
    """Return a single deterministic scenario from forecast medians."""
    da = _q50(da_fc_win, "DA", horizon)
    bm = _q50(bm_fc_win, "BM", horizon)
    dc_low = _block_avg_1d(
        _q50(dc_low_fc_win, "DC Low", horizon),
        horizon,
        window_start,
    )
    dc_high = _block_avg_1d(
        _q50(dc_high_fc_win, "DC High", horizon),
        horizon,
        window_start,
    )
    return {
        "da": da[np.newaxis, :],
        "bm": bm[np.newaxis, :],
        "dc_low": np.maximum(dc_low, 0.0)[np.newaxis, :],
        "dc_high": np.maximum(dc_high, 0.0)[np.newaxis, :],
        "seed": seed,
    }


def make_naive_scenario_builder(
    da_actual: pd.DataFrame,
    bm_actual: pd.DataFrame,
    dc_low_actual: pd.DataFrame,
    dc_high_actual: pd.DataFrame,
    lag_periods: int,
) -> ScenarioBuilder:
    """Build a strict historical-lag baseline with no filling or zero fallback.

    The recommended values are 96 periods for the previous complete comparable
    day and 336 periods for the prior week.
    """
    if lag_periods <= 0:
        raise ValueError("lag_periods must be positive")
    lag_offset = pd.Timedelta(minutes=30 * lag_periods)
    label = f"lag_{lag_periods}"

    actuals = {
        "da": _col(da_actual).sort_index(),
        "bm": _col(bm_actual).sort_index(),
        "dc_low": _col(dc_low_actual).sort_index(),
        "dc_high": _col(dc_high_actual).sort_index(),
    }

    def builder(
        window_start: pd.Timestamp,
        horizon: int,
        da_fc_win: pd.DataFrame,
        bm_fc_win: pd.DataFrame,
        dc_low_fc_win: pd.DataFrame,
        dc_high_fc_win: pd.DataFrame,
        seed=None,
    ) -> dict:
        del da_fc_win, bm_fc_win, dc_low_fc_win, dc_high_fc_win
        lag_index = pd.date_range(
            start=window_start - lag_offset,
            periods=horizon,
            freq="30min",
        )

        def extract(name: str) -> np.ndarray:
            values = actuals[name].reindex(lag_index)
            if values.isna().any():
                missing = values.index[values.isna()]
                raise ValueError(
                    f"{label} {name} has {len(missing)} missing periods. "
                    f"First missing period: {missing[0]}"
                )
            return values.to_numpy(dtype=float)

        da = extract("da")
        bm = extract("bm")
        dc_low = _block_avg_1d(extract("dc_low"), horizon, window_start)
        dc_high = _block_avg_1d(extract("dc_high"), horizon, window_start)
        return {
            "da": da[np.newaxis, :],
            "bm": bm[np.newaxis, :],
            "dc_low": np.maximum(dc_low, 0.0)[np.newaxis, :],
            "dc_high": np.maximum(dc_high, 0.0)[np.newaxis, :],
            "seed": seed,
        }

    builder.__name__ = f"naive_scenario_builder_{label}"
    return builder
