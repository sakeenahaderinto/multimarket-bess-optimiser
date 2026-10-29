"""Run one leakage-safe daily BESS backtest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from backtest.engine import DEFAULT_OPT_SETTINGS, run_backtest
from features.pipeline_da_hourly import upsample_da_forecast
from optimiser.model import validate_cvar_configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
DATA_DIR = Path("data")


def _ensure_utc(index: pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a DatetimeIndex")
    if index.tz is None:
        logger.warning("Localising %s to UTC", name)
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def _load_actual(glob_pattern: str, value_col: str) -> pd.Series:
    files = sorted(DATA_DIR.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {DATA_DIR / glob_pattern}")

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        time_column = "datetime" if "datetime" in frame.columns else "startTime"
        if time_column not in frame.columns:
            raise KeyError(f"No recognised timestamp column in {path}")
        if value_col not in frame.columns:
            raise KeyError(f"Column {value_col!r} is missing from {path}")
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
        frames.append(frame.set_index(time_column)[[value_col]])

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.index = _ensure_utc(combined.index, glob_pattern)
    return combined[value_col]


def _load_bm_actual() -> pd.DataFrame:
    """Load latest BM targets with their paired publication timestamps."""
    path = DATA_DIR / "processed" / "bmrs_processed.parquet"
    if not path.exists():
        raise FileNotFoundError(f"BM actual file not found: {path}")

    frame = pd.read_parquet(path)
    required = {"startTime", "createdDateTime", "systemSellPrice"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["startTime"] = pd.to_datetime(frame["startTime"], utc=True, errors="coerce")
    frame["createdDateTime"] = pd.to_datetime(
        frame["createdDateTime"],
        utc=True,
        errors="coerce",
    )
    if frame[["startTime", "createdDateTime"]].isna().any().any():
        raise ValueError("BM actuals contain invalid settlement or publication timestamps")

    latest = (
        frame.sort_values(["startTime", "createdDateTime"])
        .drop_duplicates(subset="startTime", keep="last")
        .set_index("startTime")
        .sort_index()
    )
    if (latest["createdDateTime"] < latest.index).any():
        raise ValueError("BM actuals contain publication times before settlement start")
    latest.index = _ensure_utc(latest.index, "BM actual")
    return latest[["systemSellPrice", "createdDateTime"]].rename(
        columns={
            "systemSellPrice": "value",
            "createdDateTime": "bm_target_created_time",
        }
    )


def _load_forecast(model_id: str, period: str) -> pd.DataFrame:
    if period == "cv":
        path = DATA_DIR / "forecasts" / f"{model_id}_oof_2022_2024.parquet"
    elif period == "oos":
        path = DATA_DIR / "forecasts" / f"{model_id}_oos_2025.parquet"
    else:
        raise ValueError("period must be 'cv' or 'oos'")

    if not path.exists():
        raise FileNotFoundError(f"Forecast file not found: {path}")
    frame = pd.read_parquet(path)
    frame.index = _ensure_utc(pd.DatetimeIndex(frame.index), f"{model_id} forecast")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    required = {"q10", "q50", "q90"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame[["q10", "q50", "q90"]]


def _load_dc_actual(service: str) -> pd.Series:
    path = DATA_DIR / "processed" / "dc_auction_processed.parquet"
    frame = pd.read_parquet(path)
    frame = frame[frame["service"] == service].copy()
    if frame.empty:
        raise ValueError(f"No rows found for DC service {service}")
    frame["delivery_start"] = pd.to_datetime(frame["delivery_start"], utc=True)
    series = frame.set_index("delivery_start")["clearing_price"].sort_index()
    if series.index.has_duplicates:
        unique_counts = series.groupby(level=0).nunique(dropna=False)
        if (unique_counts > 1).any():
            raise ValueError(f"Conflicting duplicate clearing prices for {service}")
        series = series.groupby(level=0).last()
    series.index = _ensure_utc(series.index, service)
    return series


def _log_data_summary(name: str, values: pd.Series | pd.DataFrame) -> None:
    logger.info(
        "%-22s rows=%d, %s to %s, timezone=%s",
        name,
        len(values),
        values.index.min() if len(values) else "N/A",
        values.index.max() if len(values) else "N/A",
        values.index.tz,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily multi-market BESS backtest")
    parser.add_argument("--period", choices=["cv", "oos"], required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--use-hourly-da", action="store_true")
    parser.add_argument("--use-spread-model", action="store_true")
    parser.add_argument("--calibrated-spread", action="store_true")
    parser.add_argument("--n-scenarios", type=int, default=20)
    parser.add_argument("--mean-weight", type=float, default=1.0)
    parser.add_argument("--cvar-alpha", type=float, default=0.90)
    parser.add_argument(
        "--terminal-mode",
        choices=["initial", "floor", "band"],
        default="initial",
    )
    parser.add_argument("--terminal-soc-min", type=float, default=0.1)
    parser.add_argument("--terminal-tolerance", type=float, default=0.0)
    parser.add_argument("--allow-partial-dc", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.calibrated_spread and not args.use_spread_model:
        parser.error("--calibrated-spread requires --use-spread-model")
    if args.n_scenarios <= 0:
        parser.error("--n-scenarios must be positive")
    if not 0 <= args.mean_weight <= 1:
        parser.error("--mean-weight must be in [0, 1]")
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

    da_actual = _load_actual("processed/day_ahead_processed.parquet", "value")
    bm_actual_frame = _load_bm_actual()
    bm_actual = bm_actual_frame["value"]
    dc_low_actual = _load_dc_actual("DCL")
    dc_high_actual = _load_dc_actual("DCH")

    if args.use_hourly_da:
        da_hourly = _load_forecast("da_hourly", args.period)
        da_forecast = upsample_da_forecast(da_hourly)
    else:
        da_forecast = _load_forecast("da", args.period)

    bm_forecast = _load_forecast("bm", args.period)
    dc_low_forecast = _load_forecast("dc_low", args.period)
    dc_high_forecast = _load_forecast("dc_high", args.period)

    spread_forecast = None
    if args.use_spread_model:
        spread_id = "spread_calibrated" if args.calibrated_spread else "spread"
        spread_forecast = _load_forecast(spread_id, args.period)

    for name, values in (
        ("DA actual", da_actual),
        ("BM actual", bm_actual),
        ("DC Low actual", dc_low_actual),
        ("DC High actual", dc_high_actual),
        ("DA forecast", da_forecast),
        ("BM forecast", bm_forecast),
    ):
        _log_data_summary(name, values)

    optimisation_settings = DEFAULT_OPT_SETTINGS.copy()
    optimisation_settings.update(
        {
            "n_scenarios": args.n_scenarios,
            "mean_weight": args.mean_weight,
            "cvar_alpha": args.cvar_alpha,
            "terminal_soc_mode": args.terminal_mode,
            "terminal_soc_min": args.terminal_soc_min,
            "terminal_soc_tolerance": args.terminal_tolerance,
            "disable_partial_dc_blocks": not args.allow_partial_dc,
        }
    )

    results = run_backtest(
        da_actual=da_actual.to_frame("value"),
        bm_actual=bm_actual.to_frame("value"),
        dc_low_actual=dc_low_actual.to_frame("value"),
        dc_high_actual=dc_high_actual.to_frame("value"),
        da_forecast=da_forecast,
        bm_forecast=bm_forecast,
        dc_low_forecast=dc_low_forecast,
        dc_high_forecast=dc_high_forecast,
        start_date=start,
        end_date=end,
        seed=args.seed,
        spread_forecast=spread_forecast,
        opt_settings=optimisation_settings,
        strict=args.strict,
        correlation_da_model_id=("da_hourly" if args.use_hourly_da else "da"),
    )
    if results.empty:
        raise RuntimeError("The backtest produced no results")

    output_directory = DATA_DIR / "backtest"
    output_directory.mkdir(parents=True, exist_ok=True)
    da_label = "hourly_da" if args.use_hourly_da else "30min_da"
    output_path = output_directory / (
        f"backtest_{args.period}_{start}_{end}_{da_label}_seed{args.seed}.parquet"
    )
    results.to_parquet(output_path)

    solved = results[~results["solve_failed"].fillna(True)]
    total = float(solved["net_revenue"].sum())
    annualised = total / len(solved) * 365.0 if len(solved) else float("nan")

    print("\n" + "=" * 62)
    print(f"Backtest {args.period.upper()} from {start} to {end}")
    print(f"Solved days: {len(solved)} / {len(results)}")
    print(f"DA revenue: £{solved['da_revenue'].sum():,.2f}")
    print(f"BM SSP proxy revenue: £{solved['bm_proxy_revenue'].sum():,.2f}")
    print(f"DC Low revenue: £{solved['dc_low_revenue'].sum():,.2f}")
    print(f"DC High revenue: £{solved['dc_high_revenue'].sum():,.2f}")
    print(f"Degradation cost: £{solved['degradation_cost'].sum():,.2f}")
    print(f"Net revenue: £{total:,.2f}")
    print(f"Annualised sample mean: £{annualised:,.2f}")
    print(f"Saved to: {output_path}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
