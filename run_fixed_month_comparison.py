"""Compare scenario methods over a user-selected fixed date range."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import DEFAULT_OPT_SETTINGS, _normalise_to_30min, run_backtest
from baselines.scenarios import _block_avg_1d, _col, build_median_scenario
from features.pipeline_da_hourly import upsample_da_forecast
from historical_error_scenarios import (
    build_error_path_pool,
    make_historical_error_scenario_builder,
)
from run_backtest import _load_actual, _load_bm_actual, _load_dc_actual, _load_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
DATA_DIR = Path("data")
REVENUE_COLUMNS = [
    "da_revenue",
    "bm_proxy_revenue",
    "dc_low_revenue",
    "dc_high_revenue",
    "gross_revenue",
    "degradation_cost",
    "net_revenue",
]


def make_perfect_foresight_builder(
    da_actual: pd.DataFrame,
    bm_actual: pd.DataFrame,
    dc_low_actual: pd.DataFrame,
    dc_high_actual: pd.DataFrame,
):
    """Create a deterministic realised-price benchmark."""
    actuals = {
        "da": _col(da_actual),
        "bm": _col(bm_actual),
        "dc_low": _col(dc_low_actual),
        "dc_high": _col(dc_high_actual),
    }

    def builder(
        window_start,
        horizon,
        da_fc_win,
        bm_fc_win,
        dc_low_fc_win,
        dc_high_fc_win,
        seed=None,
    ):
        del da_fc_win, bm_fc_win, dc_low_fc_win, dc_high_fc_win
        index = pd.date_range(window_start, periods=horizon, freq="30min")
        paths = {}
        for market, series in actuals.items():
            values = series.reindex(index)
            if values.isna().any():
                missing = values.index[values.isna()]
                raise ValueError(
                    f"Perfect foresight {market} has missing data at {missing[0]}"
                )
            paths[market] = values.to_numpy(dtype=float)

        paths["dc_low"] = _block_avg_1d(paths["dc_low"], horizon, window_start)
        paths["dc_high"] = _block_avg_1d(paths["dc_high"], horizon, window_start)
        return {
            market: paths[market][np.newaxis, :]
            for market in ("da", "bm", "dc_low", "dc_high")
        } | {"seed": seed}

    return builder


def make_q50_anchored_builder(base_builder, anchor_fraction: float):
    if not 0 <= anchor_fraction <= 1:
        raise ValueError("anchor_fraction must be in [0, 1]")

    def builder(
        window_start,
        horizon,
        da_fc_win,
        bm_fc_win,
        dc_low_fc_win,
        dc_high_fc_win,
        seed=None,
    ):
        scenarios = base_builder(
            window_start,
            horizon,
            da_fc_win,
            bm_fc_win,
            dc_low_fc_win,
            dc_high_fc_win,
            seed,
        )
        medians = {
            "da": da_fc_win["q50"].iloc[:horizon].to_numpy(dtype=float),
            "bm": bm_fc_win["q50"].iloc[:horizon].to_numpy(dtype=float),
            "dc_low": _block_avg_1d(
                dc_low_fc_win["q50"].iloc[:horizon].to_numpy(dtype=float),
                horizon,
                window_start,
            ),
            "dc_high": _block_avg_1d(
                dc_high_fc_win["q50"].iloc[:horizon].to_numpy(dtype=float),
                horizon,
                window_start,
            ),
        }
        count = int(round(anchor_fraction * np.asarray(scenarios["da"]).shape[0]))
        if anchor_fraction > 0:
            count = max(1, count)
        for market in medians:
            np.asarray(scenarios[market])[:count] = medians[market]
        return scenarios

    builder.__name__ = f"q50_anchor_{int(anchor_fraction * 100):02d}"
    return builder


def _average(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    solved_sets = [set(frame[~frame["solve_failed"].fillna(True)].index) for frame in frames]
    common = solved_sets[0]
    for values in solved_sets[1:]:
        common &= values
    index = pd.Index(sorted(common))
    if index.empty:
        return pd.DataFrame()
    result = frames[0].reindex(index).copy()
    for column in REVENUE_COLUMNS:
        matrix = np.vstack([frame.reindex(index)[column].to_numpy(float) for frame in frames])
        result[column] = matrix.mean(axis=0)
        result[f"{column}_seed_std"] = matrix.std(axis=0, ddof=1) if len(frames) > 1 else 0.0
    result["solve_failed"] = False
    return result


def _metrics(frame: pd.DataFrame, label: str) -> dict:
    solved = frame[~frame["solve_failed"].fillna(True)]
    net = solved["net_revenue"]
    worst_count = max(1, int(np.ceil(0.1 * len(net)))) if len(net) else 0
    return {
        "strategy": label,
        "days": len(solved),
        "total_net_revenue": round(float(net.sum()), 2),
        "average_daily_net_revenue": round(float(net.mean()), 2),
        "worst_daily_revenue": round(float(net.min()), 2),
        "p05_daily_revenue": round(float(net.quantile(0.05)), 2),
        "average_worst_10pct": (
            round(float(net.nsmallest(worst_count).mean()), 2) if worst_count else np.nan
        ),
        **{
            f"total_{column}": round(float(solved[column].sum()), 2)
            for column in REVENUE_COLUMNS[:-1]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-range BESS comparison")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--period", choices=["cv", "oos"], default="cv")
    parser.add_argument("--use-hourly-da", action="store_true")
    parser.add_argument("--n-scenarios", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--error-cap-pct",
        type=float,
        default=None,
        help=(
            "Optional absolute-residual winsorisation percentile in (0, 1). "
            "By default historical error paths are not clipped."
        ),
    )
    args = parser.parse_args()

    da_actual = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    bm_actual = _load_bm_actual()
    dc_low = _load_dc_actual("DCL").to_frame("value")
    dc_high = _load_dc_actual("DCH").to_frame("value")
    da_actual_30 = _normalise_to_30min(da_actual, "da_actual", fill_limit=1)
    dc_low_30 = _normalise_to_30min(dc_low, "dc_low_actual", fill_limit=7)
    dc_high_30 = _normalise_to_30min(dc_high, "dc_high_actual", fill_limit=7)

    da_forecast = (
        upsample_da_forecast(_load_forecast("da_hourly", args.period))
        if args.use_hourly_da
        else _load_forecast("da", args.period)
    )
    bm_forecast = _load_forecast("bm", args.period)
    dc_low_forecast = _load_forecast("dc_low", args.period)
    dc_high_forecast = _load_forecast("dc_high", args.period)

    error_pool = build_error_path_pool(
        da_actual_30,
        bm_actual,
        dc_low_30,
        dc_high_30,
        da_forecast,
        bm_forecast,
        dc_low_forecast,
        dc_high_forecast,
    )
    historical = make_historical_error_scenario_builder(
        error_pool,
        n=args.n_scenarios,
        da_actual=da_actual_30,
        error_cap_pct=args.error_cap_pct,
    )

    strategies = [
        (
            "perfect_foresight",
            make_perfect_foresight_builder(
                da_actual_30,
                bm_actual,
                dc_low_30,
                dc_high_30,
            ),
            {"mean_weight": 1.0},
            True,
        ),
        ("median", build_median_scenario, {"mean_weight": 1.0}, True),
        ("gaussian_copula", None, {"mean_weight": 1.0}, False),
        ("historical_errors", historical, {"mean_weight": 1.0}, False),
        (
            "historical_errors_cvar",
            historical,
            {"mean_weight": 0.7, "cvar_alpha": 0.90},
            False,
        ),
        (
            "historical_errors_q50_anchor25",
            make_q50_anchored_builder(historical, 0.25),
            {"mean_weight": 1.0},
            False,
        ),
    ]

    shared = {
        "da_actual": da_actual,
        "bm_actual": bm_actual,
        "dc_low_actual": dc_low,
        "dc_high_actual": dc_high,
        "da_forecast": da_forecast,
        "bm_forecast": bm_forecast,
        "dc_low_forecast": dc_low_forecast,
        "dc_high_forecast": dc_high_forecast,
        "start_date": args.start,
        "end_date": args.end,
        "strict": False,
        "correlation_da_model_id": ("da_hourly" if args.use_hourly_da else "da"),
    }

    output_directory = DATA_DIR / "results" / f"fixed_{args.label}"
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {}

    for label, builder, extra_settings, deterministic in strategies:
        settings_data = DEFAULT_OPT_SETTINGS.copy()
        settings_data.update(
            {
                "n_scenarios": args.n_scenarios,
                "terminal_soc_mode": "initial",
                "disable_partial_dc_blocks": True,
                **extra_settings,
            }
        )
        strategy_seeds = [args.seeds[0]] if deterministic else args.seeds
        frames = [
            run_backtest(
                **shared,
                scenario_builder=builder,
                seed=seed,
                opt_settings=settings_data,
            )
            for seed in strategy_seeds
        ]
        frames = [frame for frame in frames if not frame.empty]
        averaged = _average(frames)
        if averaged.empty:
            continue
        outputs[label] = averaged

    common = None
    for frame in outputs.values():
        solved = set(frame[~frame["solve_failed"].fillna(True)].index)
        common = solved if common is None else common & solved
    common = common or set()
    if not common:
        raise RuntimeError("No shared solved dates across strategies")

    rows = []
    for label, frame in outputs.items():
        clean = frame[frame.index.isin(common)]
        clean.to_parquet(output_directory / f"backtest_{label}.parquet")
        rows.append(_metrics(clean, label))

    comparison = pd.DataFrame(rows).set_index("strategy")
    comparison.to_csv(output_directory / "comparison.csv")
    print(comparison.to_string())
    print(f"Saved to {output_directory}")


if __name__ == "__main__":
    main()
