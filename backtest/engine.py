"""Daily walk-forward backtesting engine for the stylised BESS model."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from config import settings
from excluded_dates import is_excluded
from market_calendar import realised_data_cutoff_time
from optimiser.model import build_model, degradation_cost_per_mwh
from optimiser.scenarios import (
    _estimate_dependence_parameters,
    sample_scenarios_multimarket,
)

logger = logging.getLogger(__name__)

DEFAULT_BATTERY = {
    "id": "bat_01",
    "capacity_mwh": 1.0,
    "max_power_mw": 1.0,
    "current_soc": 0.5,
    "round_trip_efficiency": 0.85,
    "design_cycle_life": 4000,
}

DEFAULT_OPT_SETTINGS = {
    "replacement_cost_per_mwh": "175000",
    "dc_efa_block_min_hours": "4",
    "dc_max_fraction": "0.5",
    "dc_response_duration_h": "0.25",
    "soc_min": "0.1",
    "soc_max": "0.9",
    "terminal_soc_mode": "initial",
    "terminal_soc_min": "0.1",
    "terminal_soc_tolerance": "0.0",
    "disable_partial_dc_blocks": True,
    "mean_weight": "1.0",
    "cvar_alpha": "0.90",
}

HORIZON = 48
SETTLE_PERIODS = 48
N_SCENARIOS = 20
PERIOD = pd.Timedelta(minutes=30)

ScenarioBuilder = Callable[
    [
        pd.Timestamp,
        int,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        int | None,
    ],
    dict,
]


def _ensure_utc_frame(
    frame: pd.DataFrame | pd.Series,
    name: str,
) -> pd.DataFrame | pd.Series:
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a DatetimeIndex")
    if result.index.tz is None:
        logger.warning("%s index is timezone-naive. Localising it to UTC.", name)
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")
    result = result.sort_index()
    if result.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate timestamps")
    return result


def _normalise_to_30min(
    df: pd.DataFrame,
    name: str,
    fill_limit: int | None = None,
) -> pd.DataFrame:
    """Compatibility helper with safe market-specific default fill limits."""
    frame = _ensure_utc_frame(df, name)
    assert isinstance(frame, pd.DataFrame)
    if frame.empty:
        return frame

    if fill_limit is None:
        lowered = name.lower()
        if "da" in lowered or "day_ahead" in lowered:
            fill_limit = 1
        elif "dc" in lowered:
            fill_limit = 7

    inferred = pd.infer_freq(frame.index)
    if inferred in ("30min", "30T"):
        return frame.asfreq("30min")

    resampler = frame.resample("30min")
    return resampler.ffill(limit=fill_limit) if fill_limit is not None else resampler.asfreq()


def _require_exact_grid(
    values: pd.DataFrame | pd.Series,
    expected_index: pd.DatetimeIndex,
    name: str,
) -> pd.DataFrame | pd.Series:
    """Reindex onto an exact grid and reject every missing value."""
    aligned = values.reindex(expected_index)
    missing_mask = (
        aligned.isna().any(axis=1)
        if isinstance(aligned, pd.DataFrame)
        else aligned.isna()
    )
    if bool(missing_mask.any()):
        missing = aligned.index[missing_mask]
        raise ValueError(
            f"{name} has {len(missing)} missing periods. "
            f"First missing period: {missing[0]}"
        )
    return aligned


def _first_column_or_value(frame: pd.DataFrame, name: str) -> pd.Series:
    if "value" in frame.columns:
        return frame["value"]
    if frame.shape[1] != 1:
        raise ValueError(
            f"{name} must contain a 'value' column or exactly one column. "
            f"Columns: {list(frame.columns)}"
        )
    return frame.iloc[:, 0]


def settle_revenue(
    model: pyo.ConcreteModel,
    da_actual: pd.Series,
    bm_actual: pd.Series,
    dc_low_actual: pd.Series,
    dc_high_actual: pd.Series,
    battery_id: str,
    n_periods: int,
    battery: dict | None = None,
    opt_settings: dict | None = None,
) -> dict[str, float]:
    """Settle first-stage decisions against realised prices.

    BM is an SSP-based full-acceptance proxy. It is not BOA settlement.
    """
    battery_data = DEFAULT_BATTERY if battery is None else battery
    settings_data = DEFAULT_OPT_SETTINGS if opt_settings is None else opt_settings

    series = {
        "da": da_actual,
        "bm": bm_actual,
        "dc_low": dc_low_actual,
        "dc_high": dc_high_actual,
    }
    for name, values in series.items():
        if len(values) != n_periods:
            raise ValueError(f"{name} actual has {len(values)} rows, expected {n_periods}")
        if values.isna().any():
            raise ValueError(f"{name} actual contains NaN values")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} actual contains non-finite values")

    replacement_cost = float(settings_data.get("replacement_cost_per_mwh", 175000.0))
    cycle_life = battery_data.get("design_cycle_life", 4000)
    degradation_per_mwh = degradation_cost_per_mwh(replacement_cost, cycle_life)

    da_revenue = 0.0
    bm_proxy_revenue = 0.0
    dc_low_revenue = 0.0
    dc_high_revenue = 0.0
    degradation_cost = 0.0

    da_prices = da_actual.to_numpy(dtype=float)
    bm_prices = bm_actual.to_numpy(dtype=float)
    dcl_prices = dc_low_actual.to_numpy(dtype=float)
    dch_prices = dc_high_actual.to_numpy(dtype=float)

    for t in range(n_periods):
        da_discharge = float(pyo.value(model.da_discharge[t, battery_id]))
        da_charge = float(pyo.value(model.da_charge[t, battery_id]))
        bm_offer = float(pyo.value(model.bm_offer[t, battery_id]))

        scenario_dispatch = np.array(
            [
                pyo.value(model.bm_dispatch[t, battery_id, scenario])
                for scenario in model.S
            ],
            dtype=float,
        )
        if not np.allclose(scenario_dispatch, bm_offer, atol=1e-7):
            raise ValueError(
                f"BM full-acceptance equality is violated at period {t}: "
                f"offer={bm_offer}, dispatch={scenario_dispatch.tolist()}"
            )

        dc_low = float(pyo.value(model.dc_low[t, battery_id]))
        dc_high = float(pyo.value(model.dc_high[t, battery_id]))

        da_revenue += da_prices[t] * (da_discharge - da_charge) * 0.5
        bm_proxy_revenue += bm_prices[t] * bm_offer * 0.5
        dc_low_revenue += dcl_prices[t] * dc_low * 0.5
        dc_high_revenue += dch_prices[t] * dc_high * 0.5
        degradation_cost += (
            da_discharge + da_charge + bm_offer
        ) * 0.5 * degradation_per_mwh

    gross = da_revenue + bm_proxy_revenue + dc_low_revenue + dc_high_revenue
    net = gross - degradation_cost
    return {
        "da_revenue": round(da_revenue, 2),
        "bm_proxy_revenue": round(bm_proxy_revenue, 2),
        "dc_low_revenue": round(dc_low_revenue, 2),
        "dc_high_revenue": round(dc_high_revenue, 2),
        "gross_revenue": round(gross, 2),
        "degradation_cost": round(degradation_cost, 2),
        "net_revenue": round(net, 2),
    }


def _read_ending_soc(
    model: pyo.ConcreteModel,
    battery_id: str,
    settle_t: int | None = None,
) -> float:
    target_t = (
        settle_t
        if settle_t is not None
        else max(t for t, b, s in model.soc if b == battery_id)
    )
    values = np.array(
        [pyo.value(model.soc[target_t, battery_id, scenario]) for scenario in model.S],
        dtype=float,
    )
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("No finite ending SoC values were found")
    if not np.allclose(values, values[0], atol=1e-7):
        raise ValueError(
            "Scenario SoC paths differ under the full-acceptance formulation"
        )
    return float(values[0])


def _seed_for_date(base_seed: int, current_date: date) -> int:
    """Derive a stable independent seed from the base seed and calendar date."""
    sequence = np.random.SeedSequence([int(base_seed), int(current_date.toordinal())])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _failure_row(
    current_date: date,
    reason: str,
    stage: str,
    *,
    state_reset: bool = False,
    starting_soc: float | None = None,
) -> dict:
    return {
        "date": current_date,
        "solve_failed": True,
        "failure_reason": reason,
        "failure_stage": stage,
        "da_revenue": np.nan,
        "bm_proxy_revenue": np.nan,
        "dc_low_revenue": np.nan,
        "dc_high_revenue": np.nan,
        "gross_revenue": np.nan,
        "degradation_cost": np.nan,
        "net_revenue": np.nan,
        "state_reset": state_reset,
        "starting_soc": starting_soc,
        "ending_soc": np.nan,
    }


def run_backtest(
    da_actual: pd.DataFrame,
    bm_actual: pd.DataFrame,
    dc_low_actual: pd.DataFrame,
    dc_high_actual: pd.DataFrame,
    da_forecast: pd.DataFrame,
    bm_forecast: pd.DataFrame,
    dc_low_forecast: pd.DataFrame,
    dc_high_forecast: pd.DataFrame,
    battery: dict | None = None,
    opt_settings: dict | None = None,
    step_days: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
    seed: int = 42,
    scenario_builder: ScenarioBuilder | None = None,
    spread_forecast: pd.DataFrame | None = None,
    strict: bool = False,
    exclude_flagged_dates: bool = True,
    correlation_da_model_id: str = "da",
) -> pd.DataFrame:
    """Run a daily 24-hour walk-forward backtest on an exact UTC grid."""
    if step_days != 1:
        raise ValueError("Only daily steps are supported")

    battery_data = copy.deepcopy(DEFAULT_BATTERY if battery is None else battery)
    settings_data = DEFAULT_OPT_SETTINGS.copy()
    if opt_settings:
        settings_data.update(opt_settings)

    da_actual_utc = _ensure_utc_frame(da_actual, "da_actual")
    bm_actual_utc = _ensure_utc_frame(bm_actual, "bm_actual")
    dcl_actual_utc = _ensure_utc_frame(dc_low_actual, "dc_low_actual")
    dch_actual_utc = _ensure_utc_frame(dc_high_actual, "dc_high_actual")
    da_forecast_utc = _ensure_utc_frame(da_forecast, "da_forecast")
    bm_forecast_utc = _ensure_utc_frame(bm_forecast, "bm_forecast")
    dcl_forecast_utc = _ensure_utc_frame(dc_low_forecast, "dc_low_forecast")
    dch_forecast_utc = _ensure_utc_frame(dc_high_forecast, "dc_high_forecast")
    spread_forecast_utc = (
        _ensure_utc_frame(spread_forecast, "spread_forecast")
        if spread_forecast is not None
        else None
    )

    assert isinstance(da_actual_utc, pd.DataFrame)
    assert isinstance(bm_actual_utc, pd.DataFrame)
    assert isinstance(dcl_actual_utc, pd.DataFrame)
    assert isinstance(dch_actual_utc, pd.DataFrame)
    assert isinstance(da_forecast_utc, pd.DataFrame)
    assert isinstance(bm_forecast_utc, pd.DataFrame)
    assert isinstance(dcl_forecast_utc, pd.DataFrame)
    assert isinstance(dch_forecast_utc, pd.DataFrame)
    assert spread_forecast_utc is None or isinstance(spread_forecast_utc, pd.DataFrame)

    da_actual_30 = _normalise_to_30min(da_actual_utc, "da_actual", fill_limit=1)
    bm_actual_30 = bm_actual_utc.asfreq("30min")
    dcl_actual_30 = _normalise_to_30min(dcl_actual_utc, "dc_low_actual", fill_limit=7)
    dch_actual_30 = _normalise_to_30min(dch_actual_utc, "dc_high_actual", fill_limit=7)

    all_frames = [
        da_actual_30,
        bm_actual_30,
        dcl_actual_30,
        dch_actual_30,
        da_forecast_utc,
        bm_forecast_utc,
        dcl_forecast_utc,
        dch_forecast_utc,
    ]
    if spread_forecast_utc is not None:
        all_frames.append(spread_forecast_utc)
    if any(frame.empty for frame in all_frames):
        raise ValueError("At least one actual or forecast frame is empty")

    overlap_start = max(frame.index.min() for frame in all_frames)
    overlap_end = min(frame.index.max() for frame in all_frames)
    first_date = pd.Timestamp(start_date).date() if start_date else overlap_start.date()
    last_date = pd.Timestamp(end_date).date() if end_date else overlap_end.date()
    step_dates = list(pd.date_range(first_date, last_date, freq="D").date)
    if not step_dates:
        return pd.DataFrame()

    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        raise RuntimeError("The appsi_highs solver is not available")
    solver.options["time_limit"] = 120

    results: list[dict] = []
    current_soc = float(battery_data["current_soc"])
    initial_soc = current_soc
    last_solved_date: date | None = None

    for current_date in step_dates:
        if exclude_flagged_dates and is_excluded(current_date):
            logger.info("Skipping excluded structural date %s", current_date)
            continue

        window_start = pd.Timestamp(current_date, tz="UTC")
        full_index = pd.date_range(window_start, periods=HORIZON, freq="30min")

        try:
            da_window = _require_exact_grid(da_actual_30, full_index, "da_actual")
            bm_window = _require_exact_grid(bm_actual_30, full_index, "bm_actual")
            dcl_window = _require_exact_grid(dcl_actual_30, full_index, "dc_low_actual")
            dch_window = _require_exact_grid(dch_actual_30, full_index, "dc_high_actual")
            da_fc_window = _require_exact_grid(da_forecast_utc, full_index, "da_forecast")
            bm_fc_window = _require_exact_grid(bm_forecast_utc, full_index, "bm_forecast")
            dcl_fc_window = _require_exact_grid(
                dcl_forecast_utc,
                full_index,
                "dc_low_forecast",
            )
            dch_fc_window = _require_exact_grid(
                dch_forecast_utc,
                full_index,
                "dc_high_forecast",
            )
            spread_fc_window = (
                _require_exact_grid(
                    spread_forecast_utc,
                    full_index,
                    "spread_forecast",
                )
                if spread_forecast_utc is not None
                else None
            )
        except ValueError as exc:
            if strict:
                raise RuntimeError(f"Backtest data failure on {current_date}: {exc}") from exc
            logger.warning("Skipping %s because %s", current_date, exc)
            results.append(
                _failure_row(
                    current_date,
                    str(exc),
                    "data_validation",
                    starting_soc=current_soc,
                )
            )
            continue

        assert isinstance(da_window, pd.DataFrame)
        assert isinstance(bm_window, pd.DataFrame)
        assert isinstance(dcl_window, pd.DataFrame)
        assert isinstance(dch_window, pd.DataFrame)
        assert isinstance(da_fc_window, pd.DataFrame)
        assert isinstance(bm_fc_window, pd.DataFrame)
        assert isinstance(dcl_fc_window, pd.DataFrame)
        assert isinstance(dch_fc_window, pd.DataFrame)
        assert spread_fc_window is None or isinstance(spread_fc_window, pd.DataFrame)

        state_reset = False
        if last_solved_date is not None and current_date != last_solved_date + timedelta(days=1):
            if str(settings_data.get("terminal_soc_mode", "initial")) != "initial":
                current_soc = initial_soc
                state_reset = True
                logger.warning("Resetting SoC before non-contiguous date %s", current_date)

        try:
            scenario_seed = _seed_for_date(seed, current_date)
            if scenario_builder is None:
                cutoff = realised_data_cutoff_time(window_start)
                correlation, temporal_correlation = _estimate_dependence_parameters(
                    cutoff_date=cutoff,
                    use_spread=spread_fc_window is not None,
                    da_model_id=correlation_da_model_id,
                )
                scenarios = sample_scenarios_multimarket(
                    da_fc_window,
                    bm_fc_window,
                    dcl_fc_window,
                    dch_fc_window,
                    n=int(settings_data.get("n_scenarios", N_SCENARIOS)),
                    seed=scenario_seed,
                    corr_matrix=correlation,
                    spread_fc=spread_fc_window,
                    temporal_correlation=temporal_correlation,
                )
            else:
                scenarios = scenario_builder(
                    window_start,
                    HORIZON,
                    da_fc_window,
                    bm_fc_window,
                    dcl_fc_window,
                    dch_fc_window,
                    scenario_seed,
                )

            for market in ("da", "bm", "dc_low", "dc_high"):
                values = np.asarray(scenarios[market], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"{market} scenarios contain non-finite values")

            step_battery = copy.deepcopy(battery_data)
            step_battery["current_soc"] = current_soc
            model = build_model(
                [step_battery],
                scenarios,
                settings_data,
                window_start=window_start,
            )
            solve_result = solver.solve(model)
            condition = solve_result.solver.termination_condition
            if condition != TerminationCondition.optimal:
                raise RuntimeError(f"Solver termination condition: {condition}")

            revenue = settle_revenue(
                model,
                _first_column_or_value(da_window, "da_actual"),
                _first_column_or_value(bm_window, "bm_actual"),
                _first_column_or_value(dcl_window, "dc_low_actual"),
                _first_column_or_value(dch_window, "dc_high_actual"),
                step_battery["id"],
                SETTLE_PERIODS,
                battery=step_battery,
                opt_settings=settings_data,
            )
            ending_soc = _read_ending_soc(
                model,
                step_battery["id"],
                settle_t=SETTLE_PERIODS,
            )

            revenue.update(
                {
                    "date": current_date,
                    "solve_failed": False,
                    "failure_reason": None,
                    "failure_stage": None,
                    "state_reset": state_reset,
                    "starting_soc": round(current_soc, 8),
                    "ending_soc": round(ending_soc, 8),
                }
            )
            results.append(revenue)
            current_soc = ending_soc
            last_solved_date = current_date
            del model

        except Exception as exc:
            if strict:
                raise RuntimeError(f"Backtest failed on {current_date}: {exc}") from exc
            logger.exception("Backtest failed on %s", current_date)
            results.append(
                _failure_row(
                    current_date,
                    str(exc),
                    "scenario_optimisation_or_settlement",
                    state_reset=state_reset,
                    starting_soc=current_soc,
                )
            )

    if not results:
        return pd.DataFrame()

    output = pd.DataFrame(results).set_index("date").sort_index()

    quality_path = settings.data_dir / "quality" / "da_quality_by_date.parquet"
    legacy_path = settings.data_dir / "quality" / "da_imputed_by_date.parquet"
    selected_path = quality_path if quality_path.exists() else legacy_path
    if selected_path.exists():
        quality = pd.read_parquet(selected_path)
        quality.index = pd.to_datetime(quality.index).date
        for column in ("da_missing", "da_imputed"):
            if column in quality.columns:
                output[column] = output.index.map(quality[column]).fillna(False).astype(bool)

    return output
