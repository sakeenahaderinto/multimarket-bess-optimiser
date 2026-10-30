import pytest

pytest.importorskip("pyomo.environ")

import pandas as pd

import run_backtest as run_backtest_entry
from backtest.engine import (
    DEFAULT_BATTERY,
    _normalise_to_30min,
    _require_exact_grid,
    _seed_for_date,
    run_backtest,
    settle_revenue,
)


class _AvailableSolver:
    def available(self, exception_flag=False):
        del exception_flag
        return True

    options = {}


def test_da_normalisation_fills_only_the_matching_second_half_hour():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-01 00:00", tz="UTC"),
            pd.Timestamp("2025-01-01 01:00", tz="UTC"),
            pd.Timestamp("2025-01-01 03:00", tz="UTC"),
        ]
    )
    frame = pd.DataFrame({"value": [10.0, 20.0, 40.0]}, index=index)
    output = _normalise_to_30min(frame, "da_actual", fill_limit=1)
    assert output.loc["2025-01-01 00:30", "value"] == 10.0
    assert output.loc["2025-01-01 01:30", "value"] == 20.0
    assert pd.isna(output.loc["2025-01-01 02:00", "value"])
    assert pd.isna(output.loc["2025-01-01 02:30", "value"])


def test_exact_grid_rejects_missing_value():
    index = pd.date_range("2025-01-01", periods=3, freq="30min", tz="UTC")
    frame = pd.DataFrame({"value": [1.0, None, 3.0]}, index=index)
    with pytest.raises(ValueError, match="missing periods"):
        _require_exact_grid(frame, index, "test")


def test_daily_seed_is_stable_and_date_specific():
    first = _seed_for_date(42, pd.Timestamp("2025-01-01").date())
    repeat = _seed_for_date(42, pd.Timestamp("2025-01-01").date())
    next_day = _seed_for_date(42, pd.Timestamp("2025-01-02").date())
    assert first == repeat
    assert first != next_day


def test_settlement_rejects_negative_degradation_cost():
    actual = pd.Series([1.0])

    with pytest.raises(ValueError, match="replacement_cost_per_mwh must be non-negative"):
        settle_revenue(
            object(),
            actual,
            actual,
            actual,
            actual,
            "battery",
            1,
            opt_settings={"replacement_cost_per_mwh": -1.0},
        )


def test_settlement_rejects_infinite_actual_price():
    finite = pd.Series([1.0])
    infinite = pd.Series([float("inf")])

    with pytest.raises(ValueError, match="da actual contains non-finite values"):
        settle_revenue(
            object(),
            infinite,
            finite,
            finite,
            finite,
            "battery",
            1,
        )


def test_forecast_loader_rejects_crossed_quantiles(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    forecast_directory = tmp_path / "forecasts"
    forecast_directory.mkdir()
    index = pd.date_range("2025-01-01", periods=2, freq="30min", tz="UTC")
    pd.DataFrame(
        {"q10": [0.0, 3.0], "q50": [1.0, 2.0], "q90": [2.0, 4.0]},
        index=index,
    ).to_parquet(forecast_directory / "da_oos_2025.parquet")
    monkeypatch.setattr(run_backtest_entry, "DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match="da forecast contains crossed quantiles"):
        run_backtest_entry._load_forecast("da", "oos")


def test_bm_actual_loader_selects_latest_validated_publication_vintage(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        {
            "startTime": ["2025-01-01 00:00Z", "2025-01-01 00:00Z"],
            "createdDateTime": ["2025-01-01 00:30Z", "2025-01-01 02:00Z"],
            "systemSellPrice": [40.0, 75.0],
        }
    ).to_parquet(processed / "bmrs_processed.parquet")
    monkeypatch.setattr(run_backtest_entry, "DATA_DIR", tmp_path)

    result = run_backtest_entry._load_bm_actual()

    timestamp = pd.Timestamp("2025-01-01 00:00Z")
    assert result.loc[timestamp, "value"] == 75.0
    assert result.loc[timestamp, "bm_target_created_time"] == pd.Timestamp(
        "2025-01-01 02:00Z"
    )


def test_non_strict_data_failure_is_recorded_in_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backtest.engine.pyo.SolverFactory",
        lambda _name: _AvailableSolver(),
    )
    index = pd.date_range("2025-01-01", periods=47, freq="30min", tz="UTC")
    actual = pd.DataFrame({"value": 1.0}, index=index)
    forecast = pd.DataFrame(
        {"q10": 0.0, "q50": 1.0, "q90": 2.0},
        index=index,
    )

    result = run_backtest(
        actual,
        actual,
        actual,
        actual,
        forecast,
        forecast,
        forecast,
        forecast,
        battery=DEFAULT_BATTERY,
        start_date="2025-01-01",
        end_date="2025-01-01",
        strict=False,
        exclude_flagged_dates=False,
    )

    row = result.loc[pd.Timestamp("2025-01-01").date()]
    assert bool(row["solve_failed"])
    assert row["failure_stage"] == "data_validation"
    assert "missing periods" in row["failure_reason"]
    assert row["starting_soc"] == DEFAULT_BATTERY["current_soc"]
    assert pd.isna(row["ending_soc"])
