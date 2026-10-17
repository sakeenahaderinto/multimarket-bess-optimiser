"""Sanity checks for historical joint error-path scenarios."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.engine import _normalise_to_30min
from features.pipeline_da_hourly import upsample_da_forecast
from historical_error_scenarios import (
    build_error_path_pool,
    make_historical_error_scenario_builder,
)
from run_backtest import _load_actual, _load_bm_actual, _load_dc_actual, _load_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-hourly-da", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-date", type=str, default=None, help="Target delivery date for scenario generation test")
    args = parser.parse_args()

    da_actual = _normalise_to_30min(
        _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value"),
        "da_actual",
        fill_limit=1,
    )
    bm_actual = _load_bm_actual()
    dc_low = _normalise_to_30min(
        _load_dc_actual("DCL").to_frame("value"),
        "dc_low_actual",
        fill_limit=7,
    )
    dc_high = _normalise_to_30min(
        _load_dc_actual("DCH").to_frame("value"),
        "dc_high_actual",
        fill_limit=7,
    )

    da_forecast = (
        upsample_da_forecast(_load_forecast("da_hourly", "cv"))
        if args.use_hourly_da
        else _load_forecast("da", "cv")
    )
    bm_forecast = _load_forecast("bm", "cv")
    dc_low_forecast = _load_forecast("dc_low", "cv")
    dc_high_forecast = _load_forecast("dc_high", "cv")

    pool = build_error_path_pool(
        da_actual,
        bm_actual,
        dc_low,
        dc_high,
        da_forecast,
        bm_forecast,
        dc_low_forecast,
        dc_high_forecast,
    )

    print(f"Pool days: {len(pool['dates'])}")
    print(f"Date range: {pool['dates'][0]} to {pool['dates'][-1]}")
    for market in ("da", "bm", "dc_low", "dc_high"):
        values = np.asarray(pool[market], dtype=float)
        assert values.ndim == 2 and values.shape[1] == 48
        assert np.isfinite(values).all()
        print(
            f"{market:8s} shape={values.shape}, mean_abs={np.abs(values).mean():.3f}, "
            f"std={values.std():.3f}, min={values.min():.3f}, max={values.max():.3f}"
        )

    if args.target_date is not None:
        target_start = pd.Timestamp(args.target_date, tz="UTC")
    else:
        # A 45-day warm-up lands on 2022-03-15 in GMT, with >40 eligible GMT days in the pool.
        # (A 60-day offset lands on 2022-03-30, immediately after the spring clock change into BST,
        # where only 1 BST day exists in the pool before the cutoff).
        target_start = da_forecast.index.min().normalize() + np.timedelta64(45, "D")
    index = da_forecast.loc[target_start:].index[:48]
    if len(index) != 48:
        raise RuntimeError("No complete target day exists after the pool warm-up")

    builder = make_historical_error_scenario_builder(
        pool,
        n=20,
        da_actual=da_actual,
    )
    scenarios = builder(
        index[0],
        48,
        da_forecast.reindex(index),
        bm_forecast.reindex(index),
        dc_low_forecast.reindex(index),
        dc_high_forecast.reindex(index),
        args.seed,
    )
    for market in ("da", "bm", "dc_low", "dc_high"):
        values = np.asarray(scenarios[market])
        assert values.shape == (20, 48)
        assert np.isfinite(values).all()
        assert not np.allclose(values, values[0])
        print(f"Scenario {market} shape and diversity check passed")

    print("Historical-error pool checks passed")


if __name__ == "__main__":
    main()
