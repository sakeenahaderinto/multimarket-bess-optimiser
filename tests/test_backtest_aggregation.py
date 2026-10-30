import numpy as np
import pandas as pd
import pytest

from backtest.aggregation import aggregate_seed_results, shared_clean_dates

REVENUE_COLUMNS = ["net_revenue", "gross_revenue"]


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("date")


def test_single_seed_aggregation_preserves_exact_failure_details():
    failure_date = pd.Timestamp("2025-01-02").date()
    frame = _frame(
        [
            {
                "date": pd.Timestamp("2025-01-01").date(),
                "solve_failed": False,
                "failure_reason": None,
                "failure_stage": None,
                "net_revenue": 10.0,
                "gross_revenue": 12.0,
                "state_reset": False,
                "ending_soc": 0.5,
            },
            {
                "date": failure_date,
                "solve_failed": True,
                "failure_reason": "solver failed",
                "failure_stage": "scenario_optimisation_or_settlement",
                "net_revenue": np.nan,
                "gross_revenue": np.nan,
                "state_reset": False,
                "ending_soc": np.nan,
            },
        ]
    )

    result = aggregate_seed_results([frame], REVENUE_COLUMNS)

    assert result.loc[failure_date, "failure_reason"] == "solver failed"
    assert (
        result.loc[failure_date, "failure_stage"]
        == "scenario_optimisation_or_settlement"
    )
    assert result.loc[failure_date, "seed_run_count"] == 1
    assert result.loc[failure_date, "seed_success_count"] == 0
    assert np.isnan(result.loc[failure_date, "net_revenue_seed_std"])


def test_seed_aggregation_averages_only_unanimous_successes():
    dates = [pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-01-02").date()]
    first = _frame(
        [
            {
                "date": dates[0],
                "solve_failed": False,
                "failure_reason": None,
                "failure_stage": None,
                "net_revenue": 10.0,
                "gross_revenue": 12.0,
            },
            {
                "date": dates[1],
                "solve_failed": False,
                "failure_reason": None,
                "failure_stage": None,
                "net_revenue": 20.0,
                "gross_revenue": 22.0,
            },
        ]
    )
    second = _frame(
        [
            {
                "date": dates[0],
                "solve_failed": False,
                "failure_reason": None,
                "failure_stage": None,
                "net_revenue": 14.0,
                "gross_revenue": 16.0,
            },
            {
                "date": dates[1],
                "solve_failed": True,
                "failure_reason": "solver failed",
                "failure_stage": "scenario_optimisation_or_settlement",
                "net_revenue": np.nan,
                "gross_revenue": np.nan,
            },
        ]
    )

    result = aggregate_seed_results([first, second], REVENUE_COLUMNS)

    assert list(result.index) == dates
    assert result.loc[dates[0], "net_revenue"] == pytest.approx(12.0)
    assert result.loc[dates[0], "net_revenue_seed_std"] == pytest.approx(np.sqrt(8.0))
    assert not bool(result.loc[dates[0], "solve_failed"])
    assert result.loc[dates[0], "seed_success_count"] == 2

    assert bool(result.loc[dates[1], "solve_failed"])
    assert result.loc[dates[1], "failure_stage"] == "seed_aggregation"
    assert "1 of 2" in result.loc[dates[1], "failure_reason"]
    assert np.isnan(result.loc[dates[1], "net_revenue"])
    assert np.isnan(result.loc[dates[1], "net_revenue_seed_std"])
    assert result.loc[dates[1], "seed_run_count"] == 2
    assert result.loc[dates[1], "seed_success_count"] == 1


def test_seed_aggregation_treats_absent_date_as_unsuccessful():
    first_date = pd.Timestamp("2025-01-01").date()
    second_date = pd.Timestamp("2025-01-02").date()
    first = _frame(
        [
            {
                "date": first_date,
                "solve_failed": False,
                "failure_reason": None,
                "failure_stage": None,
                "net_revenue": 10.0,
                "gross_revenue": 12.0,
            }
        ]
    )
    second = _frame(
        [
            {
                "date": second_date,
                "solve_failed": False,
                "failure_reason": None,
                "failure_stage": None,
                "net_revenue": 14.0,
                "gross_revenue": 16.0,
            }
        ]
    )

    result = aggregate_seed_results([first, second], REVENUE_COLUMNS)

    assert list(result.index) == [first_date, second_date]
    assert result["solve_failed"].all()
    assert result["seed_success_count"].tolist() == [1, 1]
    assert result["net_revenue"].isna().all()


def test_shared_clean_dates_excludes_quality_and_state_flags_from_any_strategy():
    dates = pd.date_range("2025-01-01", periods=4, freq="D").date
    first = pd.DataFrame(
        {
            "solve_failed": [False, False, False, True],
            "da_missing": [False, True, False, False],
            "state_reset": [False, False, False, False],
        },
        index=dates,
    )
    second = pd.DataFrame(
        {
            "solve_failed": [False, False, False, False],
            "da_imputed": [False, False, False, False],
            "state_reset": [False, False, True, False],
        },
        index=dates,
    )

    assert shared_clean_dates({"first": first, "second": second}) == {dates[0]}
