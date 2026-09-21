"""Pyomo formulation for a stylised multi-market BESS optimiser.

The BM stream is explicitly a full-acceptance system-price proxy. The model
does not claim to reproduce Bid-Offer Acceptance selection or accepted-offer
settlement prices. All operational decisions are scenario-independent. The
scenario set affects expected revenue and optional CVaR risk terms.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from market_calendar import efa_block_groups, is_complete_efa_group

PERIOD_HOURS = 0.5


def _as_float(settings: dict[str, Any], key: str, default: float) -> float:
    return float(settings.get(key, default))


def _validate_battery(battery: dict[str, Any]) -> None:
    required = {
        "id",
        "capacity_mwh",
        "max_power_mw",
        "current_soc",
        "round_trip_efficiency",
        "design_cycle_life",
    }
    missing = required - set(battery)
    if missing:
        raise ValueError(f"Battery is missing keys: {sorted(missing)}")
    if float(battery["capacity_mwh"]) <= 0:
        raise ValueError("Battery capacity_mwh must be positive")
    if float(battery["max_power_mw"]) <= 0:
        raise ValueError("Battery max_power_mw must be positive")
    if not 0 < float(battery["round_trip_efficiency"]) <= 1:
        raise ValueError("round_trip_efficiency must be in (0, 1]")
    if int(battery["design_cycle_life"]) <= 0:
        raise ValueError("design_cycle_life must be positive")


def build_model(
    batteries: list[dict],
    scenarios: dict,
    settings_dict: dict,
    window_start=None,
    initial_dc=None,
) -> pyo.ConcreteModel:
    """Build the scenario-based BESS mixed-integer linear programme.

    ``initial_dc`` is rejected deliberately. A 24-hour UTC window splits the
    cross-midnight EFA block. This implementation handles that boundary by
    fixing incomplete edge blocks to zero when ``disable_partial_dc_blocks``
    is true. Carrying a commitment chosen from only part of a block would be
    myopic and is not used in the publication configuration.
    """
    if initial_dc is not None:
        raise ValueError(
            "initial_dc carry is not supported. Use disable_partial_dc_blocks=True "
            "or implement a same-origin lookahead covering complete EFA blocks."
        )

    required_markets = {"da", "bm", "dc_low", "dc_high"}
    missing_markets = required_markets - set(scenarios)
    if missing_markets:
        raise ValueError(f"Scenario dictionary is missing: {sorted(missing_markets)}")

    arrays = {
        market: np.asarray(scenarios[market], dtype=float)
        for market in required_markets
    }
    n_scenarios, n_periods = arrays["da"].shape
    if n_scenarios <= 0 or n_periods <= 0:
        raise ValueError("At least one scenario and one period are required")

    for market, values in arrays.items():
        if values.shape != (n_scenarios, n_periods):
            raise ValueError(
                f"Scenario {market!r} has shape {values.shape}. "
                f"Expected {(n_scenarios, n_periods)}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Scenario {market!r} contains non-finite prices")

    if not batteries:
        raise ValueError("At least one battery is required")
    for battery in batteries:
        _validate_battery(battery)

    battery_ids = [battery["id"] for battery in batteries]
    if len(set(battery_ids)) != len(battery_ids):
        raise ValueError("Battery IDs must be unique")
    battery_by_id = {battery["id"]: battery for battery in batteries}

    replacement_cost = _as_float(settings_dict, "replacement_cost_per_mwh", 175000.0)
    response_duration = _as_float(settings_dict, "dc_response_duration_h", 0.25)
    soc_min = _as_float(settings_dict, "soc_min", 0.10)
    soc_max = _as_float(settings_dict, "soc_max", 0.90)
    dc_max_fraction = _as_float(settings_dict, "dc_max_fraction", 0.50)
    mean_weight = _as_float(settings_dict, "mean_weight", 1.0)
    cvar_alpha = _as_float(settings_dict, "cvar_alpha", 0.90)
    terminal_mode = str(settings_dict.get("terminal_soc_mode", "initial"))
    terminal_floor = _as_float(settings_dict, "terminal_soc_min", soc_min)
    terminal_tolerance = _as_float(settings_dict, "terminal_soc_tolerance", 0.0)
    disable_partial_dc = bool(settings_dict.get("disable_partial_dc_blocks", True))

    if not 0 <= soc_min < soc_max <= 1:
        raise ValueError("Require 0 <= soc_min < soc_max <= 1")
    if response_duration <= 0:
        raise ValueError("dc_response_duration_h must be positive")
    if not 0 <= dc_max_fraction <= 1:
        raise ValueError("dc_max_fraction must be in [0, 1]")
    if not 0 <= mean_weight <= 1:
        raise ValueError("mean_weight must be in [0, 1]")
    if mean_weight < 1 and not 0 < cvar_alpha < 1:
        raise ValueError("cvar_alpha must be in (0, 1) when CVaR is active")
    if terminal_mode not in {"initial", "floor", "band"}:
        raise ValueError("terminal_soc_mode must be 'initial', 'floor' or 'band'")
    if terminal_tolerance < 0:
        raise ValueError("terminal_soc_tolerance must be non-negative")

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, n_periods - 1)
    model.T_SOC = pyo.RangeSet(0, n_periods)
    model.B = pyo.Set(initialize=battery_ids, ordered=True)
    model.S = pyo.RangeSet(0, n_scenarios - 1)

    model.da_charge = pyo.Var(model.T, model.B, within=pyo.NonNegativeReals)
    model.da_discharge = pyo.Var(model.T, model.B, within=pyo.NonNegativeReals)
    model.dc_low = pyo.Var(model.T, model.B, within=pyo.NonNegativeReals)
    model.dc_high = pyo.Var(model.T, model.B, within=pyo.NonNegativeReals)
    model.bm_offer = pyo.Var(model.T, model.B, within=pyo.NonNegativeReals)
    model.charge_mode = pyo.Var(model.T, model.B, within=pyo.Binary)

    # Kept with a scenario dimension for compatibility and auditability.
    # The equality below enforces full acceptance in every scenario.
    model.bm_dispatch = pyo.Var(
        model.T,
        model.B,
        model.S,
        within=pyo.NonNegativeReals,
    )
    model.soc = pyo.Var(model.T_SOC, model.B, model.S, within=pyo.NonNegativeReals)

    use_cvar = mean_weight < 1.0
    if use_cvar:
        model.zeta = pyo.Var(within=pyo.Reals)
        model.shortfall = pyo.Var(model.S, within=pyo.NonNegativeReals)

    def scenario_net_revenue(scenario: int):
        da_revenue = sum(
            arrays["da"][scenario, t]
            * (model.da_discharge[t, battery] - model.da_charge[t, battery])
            * PERIOD_HOURS
            for t in range(n_periods)
            for battery in battery_ids
        )
        bm_proxy_revenue = sum(
            arrays["bm"][scenario, t]
            * model.bm_dispatch[t, battery, scenario]
            * PERIOD_HOURS
            for t in range(n_periods)
            for battery in battery_ids
        )
        dc_low_revenue = sum(
            arrays["dc_low"][scenario, t]
            * model.dc_low[t, battery]
            * PERIOD_HOURS
            for t in range(n_periods)
            for battery in battery_ids
        )
        dc_high_revenue = sum(
            arrays["dc_high"][scenario, t]
            * model.dc_high[t, battery]
            * PERIOD_HOURS
            for t in range(n_periods)
            for battery in battery_ids
        )

        degradation = sum(
            (
                model.da_charge[t, battery]
                + model.da_discharge[t, battery]
                + model.bm_dispatch[t, battery, scenario]
            )
            * PERIOD_HOURS
            * (
                replacement_cost
                / (2.0 * float(battery_by_id[battery]["design_cycle_life"]))
            )
            for t in range(n_periods)
            for battery in battery_ids
        )
        return (
            da_revenue
            + bm_proxy_revenue
            + dc_low_revenue
            + dc_high_revenue
            - degradation
        )

    def objective_rule(_model):
        expected_revenue = sum(
            scenario_net_revenue(s) for s in range(n_scenarios)
        ) / n_scenarios
        if not use_cvar:
            return expected_revenue

        cvar = model.zeta - (
            1.0 / ((1.0 - cvar_alpha) * n_scenarios)
        ) * sum(model.shortfall[s] for s in range(n_scenarios))
        return mean_weight * expected_revenue + (1.0 - mean_weight) * cvar

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    if use_cvar:
        def shortfall_rule(_model, s):
            return model.shortfall[s] >= model.zeta - scenario_net_revenue(s)

        model.shortfall_constraint = pyo.Constraint(model.S, rule=shortfall_rule)

    def bm_full_acceptance_rule(_model, t, battery, scenario):
        return model.bm_dispatch[t, battery, scenario] == model.bm_offer[t, battery]

    model.bm_full_acceptance = pyo.Constraint(
        model.T,
        model.B,
        model.S,
        rule=bm_full_acceptance_rule,
    )

    def soc_rule(_model, t, battery, scenario):
        battery_data = battery_by_id[battery]
        if t == 0:
            return model.soc[t, battery, scenario] == float(
                battery_data["current_soc"]
            )

        eta = math.sqrt(float(battery_data["round_trip_efficiency"]))
        capacity = float(battery_data["capacity_mwh"])
        return model.soc[t, battery, scenario] == (
            model.soc[t - 1, battery, scenario]
            + model.da_charge[t - 1, battery] * eta * PERIOD_HOURS / capacity
            - (
                model.da_discharge[t - 1, battery]
                + model.bm_dispatch[t - 1, battery, scenario]
            )
            * PERIOD_HOURS
            / (eta * capacity)
        )

    model.soc_dynamics = pyo.Constraint(
        model.T_SOC,
        model.B,
        model.S,
        rule=soc_rule,
    )

    model.soc_lower = pyo.Constraint(
        model.T_SOC,
        model.B,
        model.S,
        rule=lambda _m, t, b, s: model.soc[t, b, s] >= soc_min,
    )
    model.soc_upper = pyo.Constraint(
        model.T_SOC,
        model.B,
        model.S,
        rule=lambda _m, t, b, s: model.soc[t, b, s] <= soc_max,
    )

    # DC Low is a discharge-direction service and DC High is a charge-direction service.
    def discharge_power_rule(_model, t, battery, scenario):
        return (
            model.da_discharge[t, battery]
            + model.bm_dispatch[t, battery, scenario]
            + model.dc_low[t, battery]
            <= float(battery_by_id[battery]["max_power_mw"])
        )

    def charge_power_rule(_model, t, battery):
        return (
            model.da_charge[t, battery] + model.dc_high[t, battery]
            <= float(battery_by_id[battery]["max_power_mw"])
        )

    model.discharge_power_limit = pyo.Constraint(
        model.T,
        model.B,
        model.S,
        rule=discharge_power_rule,
    )
    model.charge_power_limit = pyo.Constraint(
        model.T,
        model.B,
        rule=charge_power_rule,
    )

    def charge_mode_rule(_model, t, battery):
        return model.da_charge[t, battery] <= (
            float(battery_by_id[battery]["max_power_mw"])
            * model.charge_mode[t, battery]
        )

    def discharge_mode_rule(_model, t, battery):
        return model.da_discharge[t, battery] <= (
            float(battery_by_id[battery]["max_power_mw"])
            * (1 - model.charge_mode[t, battery])
        )

    def bm_mode_rule(_model, t, battery):
        return model.bm_offer[t, battery] <= (
            float(battery_by_id[battery]["max_power_mw"])
            * (1 - model.charge_mode[t, battery])
        )

    model.charge_mode_limit = pyo.Constraint(model.T, model.B, rule=charge_mode_rule)
    model.discharge_mode_limit = pyo.Constraint(
        model.T,
        model.B,
        rule=discharge_mode_rule,
    )
    model.bm_mode_limit = pyo.Constraint(model.T, model.B, rule=bm_mode_rule)

    if window_start is not None:
        groups = efa_block_groups(window_start, n_periods)
        period_index = pd.date_range(window_start, periods=n_periods, freq="30min")
    else:
        block_size = int(_as_float(settings_dict, "dc_efa_block_min_hours", 4.0) * 2)
        groups = [
            (start, min(start + block_size, n_periods))
            for start in range(0, n_periods, block_size)
        ]
        period_index = None

    consistency_pairs = [
        (start, t)
        for start, end in groups
        for t in range(start + 1, end)
    ]
    model.DC_PAIRS = pyo.Set(dimen=2, initialize=consistency_pairs)
    model.dc_low_consistency = pyo.Constraint(
        model.DC_PAIRS,
        model.B,
        rule=lambda _m, start, t, b: model.dc_low[t, b] == model.dc_low[start, b],
    )
    model.dc_high_consistency = pyo.Constraint(
        model.DC_PAIRS,
        model.B,
        rule=lambda _m, start, t, b: model.dc_high[t, b] == model.dc_high[start, b],
    )

    model.dc_low_cap = pyo.Constraint(
        model.T,
        model.B,
        rule=lambda _m, t, b: model.dc_low[t, b]
        <= float(battery_by_id[b]["max_power_mw"]) * dc_max_fraction,
    )
    model.dc_high_cap = pyo.Constraint(
        model.T,
        model.B,
        rule=lambda _m, t, b: model.dc_high[t, b]
        <= float(battery_by_id[b]["max_power_mw"]) * dc_max_fraction,
    )

    def dc_soc_lower_rule(_model, t, battery, scenario):
        battery_data = battery_by_id[battery]
        eta = math.sqrt(float(battery_data["round_trip_efficiency"]))
        capacity = float(battery_data["capacity_mwh"])
        return model.soc[t, battery, scenario] >= (
            soc_min
            + model.dc_low[t, battery] * response_duration / (eta * capacity)
        )

    def dc_soc_upper_rule(_model, t, battery, scenario):
        battery_data = battery_by_id[battery]
        eta = math.sqrt(float(battery_data["round_trip_efficiency"]))
        capacity = float(battery_data["capacity_mwh"])
        return model.soc[t, battery, scenario] <= (
            soc_max
            - model.dc_high[t, battery] * response_duration * eta / capacity
        )

    model.dc_soc_lower = pyo.Constraint(
        model.T,
        model.B,
        model.S,
        rule=dc_soc_lower_rule,
    )
    model.dc_soc_upper = pyo.Constraint(
        model.T,
        model.B,
        model.S,
        rule=dc_soc_upper_rule,
    )

    if disable_partial_dc and period_index is not None:
        partial_periods = [
            t
            for start, end in groups
            if not is_complete_efa_group(period_index, start, end)
            for t in range(start, end)
        ]
        for t in partial_periods:
            for battery in battery_ids:
                model.dc_low[t, battery].fix(0.0)
                model.dc_high[t, battery].fix(0.0)

    def terminal_lower_rule(_model, battery, scenario):
        initial = float(battery_by_id[battery]["current_soc"])
        if terminal_mode == "initial":
            return model.soc[n_periods, battery, scenario] == initial
        if terminal_mode == "floor":
            return model.soc[n_periods, battery, scenario] >= terminal_floor
        return model.soc[n_periods, battery, scenario] >= initial - terminal_tolerance

    model.terminal_soc_lower = pyo.Constraint(
        model.B,
        model.S,
        rule=terminal_lower_rule,
    )

    if terminal_mode == "band":
        model.terminal_soc_upper = pyo.Constraint(
            model.B,
            model.S,
            rule=lambda _m, b, s: model.soc[n_periods, b, s]
            <= float(battery_by_id[b]["current_soc"]) + terminal_tolerance,
        )

    model._scenario_net_revenue = scenario_net_revenue
    model._efa_groups = groups
    return model
