"""Compare stochastic and deterministic BESS strategies on shared dates."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import DEFAULT_OPT_SETTINGS, _normalise_to_30min, run_backtest
from baselines.scenarios import build_median_scenario, make_naive_scenario_builder
from features.pipeline_da_hourly import upsample_da_forecast
from forecasting.base import BaseForecaster
from market_calendar import efa_block_keys
from optimiser.model import validate_cvar_configuration
from run_backtest import _load_actual, _load_dc_actual, _load_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
DATA_DIR = Path("data")
RESULTS_DIR = DATA_DIR / "results"
REVENUE_COLUMNS = [
    "net_revenue",
    "da_revenue",
    "bm_proxy_revenue",
    "dc_low_revenue",
    "dc_high_revenue",
    "degradation_cost",
    "gross_revenue",
]


def _annualise(total: float, days: int) -> float:
    return total / days * 365.0 if days else float("nan")


def _shared_clean_dates(results: dict[str, pd.DataFrame]) -> set:
    sets: list[set] = []
    for frame in results.values():
        clean = frame[~frame["solve_failed"].fillna(True)].copy()
        if "da_imputed" in clean.columns:
            clean = clean[~clean["da_imputed"].fillna(False)]
        if "da_missing" in clean.columns:
            clean = clean[~clean["da_missing"].fillna(False)]
        if "state_reset" in clean.columns:
            clean = clean[~clean["state_reset"].fillna(False)]
        sets.append(set(clean.index))

    if not sets:
        return set()
    common = sets[0]
    for values in sets[1:]:
        common &= values
    return common


def _metrics(
    frame: pd.DataFrame,
    label: str,
    clean_dates: set,
    seed_count: int,
    total_seed_failures: int,
) -> dict:
    solved = frame[~frame["solve_failed"].fillna(True)]
    clean = solved[solved.index.isin(clean_dates)]
    net = clean["net_revenue"].dropna()
    worst_count = max(1, int(np.ceil(0.10 * len(net)))) if len(net) else 0

    result = {
        "strategy": label,
        "n_evaluated_days": len(solved),
        "n_clean_days": len(clean),
        "n_seeds": seed_count,
        "total_seed_failures": total_seed_failures,
        "total_net_revenue_clean": round(float(net.sum()), 2),
        "annualised_net_revenue_clean": round(_annualise(float(net.sum()), len(net)), 2),
        "avg_net_revenue_per_day_clean": round(float(net.mean()), 2) if len(net) else np.nan,
        "worst_daily_revenue_clean": round(float(net.min()), 2) if len(net) else np.nan,
        "p05_daily_revenue_clean": round(float(net.quantile(0.05)), 2) if len(net) else np.nan,
        "avg_worst_10pct_clean": (
            round(float(net.nsmallest(worst_count).mean()), 2) if worst_count else np.nan
        ),
    }
    for column in REVENUE_COLUMNS[1:]:
        result[f"total_{column}_clean"] = round(float(clean[column].sum()), 2)
    if "net_revenue_seed_std" in clean.columns:
        result["mean_daily_seed_std"] = round(
            float(clean["net_revenue_seed_std"].mean()),
            3,
        )
    return result


def _average_seed_results(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        result = frames[0].copy()
        result["net_revenue_seed_std"] = 0.0
        return result

    solved_sets = [
        set(frame[~frame["solve_failed"].fillna(True)].index)
        for frame in frames
    ]
    common = solved_sets[0]
    for values in solved_sets[1:]:
        common &= values
    common_index = pd.Index(sorted(common))
    if common_index.empty:
        return pd.DataFrame()

    result = frames[0].reindex(common_index).copy()
    for column in REVENUE_COLUMNS:
        matrix = np.vstack(
            [frame.reindex(common_index)[column].to_numpy(dtype=float) for frame in frames]
        )
        result[column] = np.mean(matrix, axis=0)
        if column == "net_revenue":
            result["net_revenue_seed_std"] = np.std(matrix, axis=0, ddof=1)

    result["solve_failed"] = False
    result["failure_reason"] = None
    return result


def _filter_clean_timestamps(index: pd.DatetimeIndex, clean_dates: set) -> np.ndarray:
    return np.asarray([timestamp.date() in clean_dates for timestamp in index], dtype=bool)


def _forecast_metrics_one_market(
    actual: pd.Series,
    forecast: pd.DataFrame,
    clean_dates: set,
    block_level: bool,
) -> tuple[dict, int]:
    common = actual.index.intersection(forecast.index)
    aligned = pd.concat(
        [
            actual.reindex(common).rename("actual"),
            forecast.reindex(common)[["q10", "q50", "q90"]],
        ],
        axis=1,
    ).dropna()

    if block_level:
        aligned["block_key"] = efa_block_keys(aligned.index).to_numpy()
        aligned["clean_timestamp"] = _filter_clean_timestamps(aligned.index, clean_dates)
        grouped = aligned.groupby("block_key", sort=True)
        counts = grouped.size()
        all_clean = grouped["clean_timestamp"].all()
        aggregated = grouped[["actual", "q10", "q50", "q90"]].mean()
        valid = (counts == 8) & all_clean
        aggregated = aggregated.loc[valid]
    else:
        aggregated = aligned.loc[_filter_clean_timestamps(aligned.index, clean_dates)]

    if aggregated.empty:
        raise ValueError("No aligned clean observations are available")
    metrics = BaseForecaster.evaluate_forecast_metrics(
        aggregated["actual"].to_numpy(dtype=float),
        aggregated["q10"].to_numpy(dtype=float),
        aggregated["q50"].to_numpy(dtype=float),
        aggregated["q90"].to_numpy(dtype=float),
    )
    return metrics, len(aggregated)


def _forecast_metrics(
    da_actual_30: pd.DataFrame,
    bm_actual: pd.DataFrame,
    dc_low_30: pd.DataFrame,
    dc_high_30: pd.DataFrame,
    da_forecast: pd.DataFrame,
    bm_forecast: pd.DataFrame,
    dc_low_forecast: pd.DataFrame,
    dc_high_forecast: pd.DataFrame,
    clean_dates: set,
) -> pd.DataFrame:
    markets = [
        ("DA (£/MWh)", da_actual_30["value"], da_forecast, False),
        ("BM SSP proxy (£/MWh)", bm_actual["value"], bm_forecast, False),
        ("DC Low (£/MW/h)", dc_low_30["value"], dc_low_forecast, True),
        ("DC High (£/MW/h)", dc_high_30["value"], dc_high_forecast, True),
    ]
    rows = []
    for market, actual, forecast, block_level in markets:
        metrics, count = _forecast_metrics_one_market(
            actual,
            forecast,
            clean_dates,
            block_level,
        )
        rows.append({"market": market, "n_observations": count, **metrics})
    return pd.DataFrame(rows).set_index("market")


def run_all(
    period: str,
    start_date: str,
    end_date: str,
    use_hourly_da: bool,
    seeds: list[int],
    n_scenarios: int,
    mean_weight: float,
    cvar_alpha: float,
) -> None:
    da_actual = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    bm_actual = _load_actual("processed/bmrs_processed.parquet", "systemSellPrice").to_frame("value")
    dc_low_raw = _load_dc_actual("DCL")
    dc_high_raw = _load_dc_actual("DCH")
    dc_low_actual = dc_low_raw.to_frame("value")
    dc_high_actual = dc_high_raw.to_frame("value")

    da_actual_30 = _normalise_to_30min(da_actual, "da_actual", fill_limit=1)
    dc_low_30 = _normalise_to_30min(dc_low_actual, "dc_low_actual", fill_limit=7)
    dc_high_30 = _normalise_to_30min(dc_high_actual, "dc_high_actual", fill_limit=7)

    if use_hourly_da:
        da_forecast = upsample_da_forecast(_load_forecast("da_hourly", period))
    else:
        da_forecast = _load_forecast("da", period)
    bm_forecast = _load_forecast("bm", period)
    dc_low_forecast = _load_forecast("dc_low", period)
    dc_high_forecast = _load_forecast("dc_high", period)

    strategies = [
        ("main", None, False),
        ("median", build_median_scenario, True),
        (
            "prev_complete_day",
            make_naive_scenario_builder(
                da_actual_30,
                bm_actual,
                dc_low_30,
                dc_high_30,
                lag_periods=96,
            ),
            True,
        ),
        (
            "prev_week",
            make_naive_scenario_builder(
                da_actual_30,
                bm_actual,
                dc_low_30,
                dc_high_30,
                lag_periods=336,
            ),
            True,
        ),
    ]

    settings_data = DEFAULT_OPT_SETTINGS.copy()
    settings_data.update(
        {
            "n_scenarios": n_scenarios,
            "mean_weight": mean_weight,
            "cvar_alpha": cvar_alpha,
            "terminal_soc_mode": "initial",
            "disable_partial_dc_blocks": True,
        }
    )

    shared_arguments = {
        "da_actual": da_actual,
        "bm_actual": bm_actual,
        "dc_low_actual": dc_low_actual,
        "dc_high_actual": dc_high_actual,
        "da_forecast": da_forecast,
        "bm_forecast": bm_forecast,
        "dc_low_forecast": dc_low_forecast,
        "dc_high_forecast": dc_high_forecast,
        "start_date": start_date,
        "end_date": end_date,
        "opt_settings": settings_data,
        "strict": False,
        "correlation_da_model_id": ("da_hourly" if use_hourly_da else "da"),
    }

    da_label = "hourly_da" if use_hourly_da else "30min_da"
    seed_label = "-".join(str(seed) for seed in seeds)
    output_directory = RESULTS_DIR / (
        f"baseline_{period}_{da_label}_n{n_scenarios}_seeds{seed_label}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    seed_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}

    for label, builder, deterministic in strategies:
        strategy_seeds = [seeds[0]] if deterministic else seeds
        per_seed = []
        total_failures = 0
        for strategy_seed in strategy_seeds:
            frame = run_backtest(
                **shared_arguments,
                seed=strategy_seed,
                scenario_builder=builder,
            )
            if frame.empty:
                continue
            total_failures += int(frame["solve_failed"].fillna(True).sum())
            frame.to_parquet(
                output_directory / f"backtest_{label}_seed{strategy_seed}.parquet"
            )
            per_seed.append(frame)

        averaged = _average_seed_results(per_seed)
        if averaged.empty:
            logger.warning("No common solved output for strategy %s", label)
            continue
        averaged.to_parquet(output_directory / f"backtest_{label}.parquet")
        results[label] = averaged
        seed_counts[label] = len(per_seed)
        failure_counts[label] = total_failures

    if len(results) != len(strategies):
        raise RuntimeError("At least one strategy did not produce comparable output")

    clean_dates = _shared_clean_dates(results)
    if not clean_dates:
        raise RuntimeError("The shared clean-date set is empty")

    comparison = pd.DataFrame(
        [
            _metrics(
                frame,
                label,
                clean_dates,
                seed_counts[label],
                failure_counts[label],
            )
            for label, frame in results.items()
        ]
    ).set_index("strategy")
    comparison.to_csv(output_directory / "baseline_comparison.csv")
    comparison.to_parquet(output_directory / "baseline_comparison.parquet")

    forecast_metrics = _forecast_metrics(
        da_actual_30,
        bm_actual,
        dc_low_30,
        dc_high_30,
        da_forecast,
        bm_forecast,
        dc_low_forecast,
        dc_high_forecast,
        clean_dates,
    )
    forecast_metrics.to_csv(output_directory / "held_out_forecast_metrics.csv")

    print("\n" + "=" * 88)
    print(f"Baseline comparison {period.upper()} from {start_date} to {end_date}")
    print(f"Shared clean dates: {len(clean_dates)}")
    print("\nForecast performance")
    print(forecast_metrics.to_string())
    print("\nStrategy performance")
    print(comparison.to_string())
    print(f"\nSaved to: {output_directory}")
    print("=" * 88 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="BESS baseline comparison")
    parser.add_argument("--period", choices=["cv", "oos"], required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--use-hourly-da", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-scenarios", type=int, default=50)
    parser.add_argument("--mean-weight", type=float, default=1.0)
    parser.add_argument("--cvar-alpha", type=float, default=0.90)
    args = parser.parse_args()

    try:
        validate_cvar_configuration(
            args.n_scenarios,
            args.mean_weight,
            args.cvar_alpha,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.period == "cv":
        start = args.start or "2022-01-01"
        end = args.end or "2024-12-31"
    else:
        start = args.start or "2025-01-01"
        end = args.end or "2025-12-31"

    run_all(
        period=args.period,
        start_date=start,
        end_date=end,
        use_hourly_da=args.use_hourly_da,
        seeds=args.seeds,
        n_scenarios=args.n_scenarios,
        mean_weight=args.mean_weight,
        cvar_alpha=args.cvar_alpha,
    )


if __name__ == "__main__":
    main()
