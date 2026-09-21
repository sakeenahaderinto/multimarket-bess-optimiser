"""Solve and export a complete schedule for one UTC delivery day."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from backtest.engine import DEFAULT_BATTERY, DEFAULT_OPT_SETTINGS, _normalise_to_30min
from baselines.scenarios import build_median_scenario
from features.pipeline_da_hourly import upsample_da_forecast
from optimiser.model import build_model
from optimiser.scenarios import _estimate_correlation_matrix, sample_scenarios_multimarket
from run_backtest import _load_actual, _load_dc_actual, _load_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--period", choices=["cv", "oos"], required=True)
    parser.add_argument("--strategy", choices=["median", "copula"], default="copula")
    parser.add_argument("--use-hourly-da", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-scenarios", type=int, default=50)
    args = parser.parse_args()

    start = pd.Timestamp(args.date, tz="UTC")
    index = pd.date_range(start, periods=48, freq="30min")

    da_actual = _normalise_to_30min(
        _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value"),
        "da_actual",
        fill_limit=1,
    ).reindex(index)
    bm_actual = _load_actual("processed/bmrs_processed.parquet", "systemSellPrice").to_frame("value").reindex(index)

    da_forecast = (
        upsample_da_forecast(_load_forecast("da_hourly", args.period))
        if args.use_hourly_da
        else _load_forecast("da", args.period)
    ).reindex(index)
    bm_forecast = _load_forecast("bm", args.period).reindex(index)
    dc_low_forecast = _load_forecast("dc_low", args.period).reindex(index)
    dc_high_forecast = _load_forecast("dc_high", args.period).reindex(index)

    if any(frame.isna().any().any() for frame in (da_actual, bm_actual, da_forecast, bm_forecast, dc_low_forecast, dc_high_forecast)):
        raise ValueError("The selected day has missing actual or forecast data")

    if args.strategy == "median":
        scenarios = build_median_scenario(
            start,
            48,
            da_forecast,
            bm_forecast,
            dc_low_forecast,
            dc_high_forecast,
            args.seed,
        )
    else:
        correlation = _estimate_correlation_matrix(
            cutoff_date=start,
            da_model_id=("da_hourly" if args.use_hourly_da else "da"),
        )
        scenarios = sample_scenarios_multimarket(
            da_forecast,
            bm_forecast,
            dc_low_forecast,
            dc_high_forecast,
            n=args.n_scenarios,
            seed=args.seed,
            corr_matrix=correlation,
        )

    settings = DEFAULT_OPT_SETTINGS.copy()
    settings.update(
        {
            "n_scenarios": args.n_scenarios,
            "terminal_soc_mode": "initial",
            "disable_partial_dc_blocks": True,
        }
    )
    model = build_model([DEFAULT_BATTERY.copy()], scenarios, settings, window_start=start)
    solver = pyo.SolverFactory("appsi_highs")
    solver.options["time_limit"] = 120
    result = solver.solve(model)
    if result.solver.termination_condition != TerminationCondition.optimal:
        raise RuntimeError(f"Solver failed: {result.solver.termination_condition}")

    rows = []
    for t, timestamp in enumerate(index):
        dispatches = np.array(
            [pyo.value(model.bm_dispatch[t, DEFAULT_BATTERY["id"], s]) for s in model.S],
            dtype=float,
        )
        rows.append(
            {
                "datetime": timestamp,
                "da_actual": float(da_actual.iloc[t, 0]),
                "bm_ssp_actual": float(bm_actual.iloc[t, 0]),
                "da_charge_mw": pyo.value(model.da_charge[t, DEFAULT_BATTERY["id"]),
                "da_discharge_mw": pyo.value(model.da_discharge[t, DEFAULT_BATTERY["id"]),
                "bm_proxy_offer_mw": pyo.value(model.bm_offer[t, DEFAULT_BATTERY["id"]),
                "bm_dispatch_scenario_std": float(dispatches.std()),
                "dc_low_mw": pyo.value(model.dc_low[t, DEFAULT_BATTERY["id"]),
                "dc_high_mw": pyo.value(model.dc_high[t, DEFAULT_BATTERY["id"]),
                "soc": pyo.value(model.soc[t, DEFAULT_BATTERY["id"], 0]),
            }
        )

    schedule = pd.DataFrame(rows).set_index("datetime")
    output = f"data/results/schedule_{args.strategy}_{args.date}.csv"
    schedule.to_csv(output)
    print(schedule.to_string())
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
