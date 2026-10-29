import numpy as np
import pytest

pyo = pytest.importorskip("pyomo.environ")
from pyomo.opt import TerminationCondition

from optimiser.model import build_model


def scenarios(periods=4, scenario_count=1, price=0.0):
    return {
        "da": np.full((scenario_count, periods), price, dtype=float),
        "bm": np.full((scenario_count, periods), price, dtype=float),
        "dc_low": np.full((scenario_count, periods), price, dtype=float),
        "dc_high": np.full((scenario_count, periods), price, dtype=float),
        "seed": 1,
    }


def battery():
    return [
        {
            "id": "test_bat",
            "current_soc": 0.5,
            "max_power_mw": 1.0,
            "capacity_mwh": 1.0,
            "round_trip_efficiency": 0.85,
            "design_cycle_life": 4000,
        }
    ]


SETTINGS = {
    "replacement_cost_per_mwh": "175000",
    "dc_efa_block_min_hours": "2",
    "dc_response_duration_h": "0.25",
    "dc_max_fraction": "0.5",
    "terminal_soc_mode": "initial",
    "disable_partial_dc_blocks": False,
}


def solve(model):
    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        pytest.skip("appsi_highs is unavailable")
    result = solver.solve(model)
    assert result.solver.termination_condition == TerminationCondition.optimal


def test_zero_prices_produce_no_energy_dispatch():
    model = build_model(battery(), scenarios(), SETTINGS)
    solve(model)
    for t in range(4):
        assert abs(pyo.value(model.da_charge[t, "test_bat"])) < 1e-7
        assert abs(pyo.value(model.da_discharge[t, "test_bat"])) < 1e-7
        assert abs(pyo.value(model.bm_offer[t, "test_bat"])) < 1e-7


def test_simple_arbitrage_and_cyclic_terminal_soc():
    values = scenarios()
    values["da"][0, 0] = -100.0
    values["da"][0, 2] = 100.0
    model = build_model(battery(), values, SETTINGS)
    solve(model)
    assert pyo.value(model.da_charge[0, "test_bat"]) > 1e-6
    assert pyo.value(model.da_discharge[2, "test_bat"]) > 1e-6
    assert abs(pyo.value(model.soc[4, "test_bat", 0]) - 0.5) < 1e-7


def test_bm_full_acceptance_and_common_soc():
    values = scenarios(scenario_count=2)
    values["bm"][0, 2] = 200.0
    values["bm"][1, 2] = 50.0
    model = build_model(battery(), values, SETTINGS)
    solve(model)

    for t in range(4):
        offer = pyo.value(model.bm_offer[t, "test_bat"])
        for s in model.S:
            assert abs(pyo.value(model.bm_dispatch[t, "test_bat", s]) - offer) < 1e-7

    for t in model.T_SOC:
        soc_values = [pyo.value(model.soc[t, "test_bat", s]) for s in model.S]
        assert np.allclose(soc_values, soc_values[0], atol=1e-7)


def test_directional_power_limits():
    values = scenarios(price=0.0)
    values["dc_low"][:] = 100.0
    values["dc_high"][:] = 100.0
    model = build_model(battery(), values, SETTINGS)
    solve(model)
    for t in range(4):
        discharge_direction = (
            pyo.value(model.da_discharge[t, "test_bat"])
            + pyo.value(model.bm_offer[t, "test_bat"])
            + pyo.value(model.dc_low[t, "test_bat"])
        )
        charge_direction = (
            pyo.value(model.da_charge[t, "test_bat"])
            + pyo.value(model.dc_high[t, "test_bat"])
        )
        assert discharge_direction <= 1.0 + 1e-7
        assert charge_direction <= 1.0 + 1e-7


def test_dc_low_energy_is_deliverable_after_scheduled_discharge():
    values = scenarios(periods=1)
    values["da"][0, 0] = 100.0
    values["dc_low"][0, 0] = 500.0
    low_soc_battery = battery()
    low_soc_battery[0]["current_soc"] = 0.55
    settings = {
        **SETTINGS,
        "dc_max_fraction": "0.2",
        "terminal_soc_mode": "floor",
        "terminal_soc_min": "0.1",
    }

    model = build_model(low_soc_battery, values, settings)
    solve(model)

    eta = np.sqrt(low_soc_battery[0]["round_trip_efficiency"])
    ending_soc = pyo.value(model.soc[1, "test_bat", 0])
    dc_low = pyo.value(model.dc_low[0, "test_bat"])
    post_response_soc = ending_soc - dc_low * 0.25 / eta
    assert dc_low > 1e-6
    assert post_response_soc >= 0.1 - 1e-7


def test_dc_high_capacity_is_deliverable_after_scheduled_charge():
    values = scenarios(periods=1)
    values["da"][0, 0] = -100.0
    values["dc_high"][0, 0] = 500.0
    mid_soc_battery = battery()
    mid_soc_battery[0]["current_soc"] = 0.55
    settings = {
        **SETTINGS,
        "dc_max_fraction": "0.2",
        "terminal_soc_mode": "floor",
        "terminal_soc_min": "0.1",
    }

    model = build_model(mid_soc_battery, values, settings)
    solve(model)

    eta = np.sqrt(mid_soc_battery[0]["round_trip_efficiency"])
    ending_soc = pyo.value(model.soc[1, "test_bat", 0])
    dc_high = pyo.value(model.dc_high[0, "test_bat"])
    post_response_soc = ending_soc + dc_high * 0.25 * eta
    assert dc_high > 1e-6
    assert post_response_soc <= 0.9 + 1e-7
