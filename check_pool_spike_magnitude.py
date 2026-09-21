"""Compare an historical-error pool with a selected realised spread event."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.engine import _normalise_to_30min
from historical_error_scenarios import build_error_path_pool
from run_backtest import _load_actual, _load_dc_actual, _load_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2022-12-15")
    parser.add_argument("--start-period", type=int, default=32)
    parser.add_argument("--end-period", type=int, default=39)
    parser.add_argument("--use-hourly-da", action="store_true")
    args = parser.parse_args()

    from features.pipeline_da_hourly import upsample_da_forecast

    da_actual = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    da_actual = _normalise_to_30min(da_actual, "da_actual", fill_limit=1)
    bm_actual = _load_actual("processed/bmrs_processed.parquet", "systemSellPrice").to_frame("value")
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

    day_start = pd.Timestamp(args.date, tz="UTC")
    index = pd.date_range(day_start, periods=48, freq="30min")
    start = args.start_period
    end = args.end_period + 1

    actual_spread = (
        bm_actual["value"].reindex(index).to_numpy(float)
        - da_actual["value"].reindex(index).to_numpy(float)
    )[start:end]
    q50_spread = (
        bm_forecast["q50"].reindex(index).to_numpy(float)
        - da_forecast["q50"].reindex(index).to_numpy(float)
    )[start:end]

    residual_spread = np.asarray(pool["bm"])[:, start:end] - np.asarray(pool["da"])[:, start:end]
    mean_shift = residual_spread.mean(axis=1)

    print(f"Target date: {args.date}")
    print(f"Periods: {start} to {end - 1}")
    print(f"Actual mean BM-minus-DA spread: {np.mean(actual_spread):.2f}")
    print(f"Forecast q50 mean spread: {np.mean(q50_spread):.2f}")
    print(f"Pool mean residual spread shift: {np.mean(mean_shift):.2f}")
    print(f"Pool 95th percentile residual shift: {np.percentile(mean_shift, 95):.2f}")
    print(f"Pool 99th percentile residual shift: {np.percentile(mean_shift, 99):.2f}")
    print(
        "Q50 plus 99th percentile pool shift: "
        f"{np.mean(q50_spread) + np.percentile(mean_shift, 99):.2f}"
    )


if __name__ == "__main__":
    main()
