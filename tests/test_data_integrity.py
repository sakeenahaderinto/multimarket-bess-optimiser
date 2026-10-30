import pandas as pd
import pytest

from data_integrity import (
    collapse_bm_vintage_duplicates,
    collapse_identical_duplicates,
    select_latest_bm_vintages,
    validate_quantile_forecast,
)


def _duplicate_index() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-01 00:00", tz="UTC"),
            pd.Timestamp("2025-01-01 00:00", tz="UTC"),
            pd.Timestamp("2025-01-01 01:00", tz="UTC"),
        ]
    )


def test_identical_dataframe_duplicates_are_collapsed():
    frame = pd.DataFrame(
        {"value": [10.0, 10.0, 20.0], "source": ["a", "b", "c"]},
        index=_duplicate_index(),
    )

    result = collapse_identical_duplicates(
        frame,
        "DA input",
        comparison_columns=["value"],
    )

    assert result.index.is_unique
    assert result["value"].tolist() == [10.0, 20.0]
    assert result["source"].tolist() == ["a", "c"]


def test_conflicting_dataframe_duplicates_are_rejected():
    frame = pd.DataFrame({"q50": [10.0, 11.0, 20.0]}, index=_duplicate_index())

    with pytest.raises(
        ValueError,
        match=r"forecast contains conflicting duplicate values.*column 'q50'",
    ):
        collapse_identical_duplicates(frame, "forecast")


def test_identical_series_duplicates_are_collapsed_with_name_preserved():
    series = pd.Series([10.0, 10.0, 20.0], index=_duplicate_index(), name="price")

    result = collapse_identical_duplicates(series, "price actuals")

    assert isinstance(result, pd.Series)
    assert result.name == "price"
    assert result.tolist() == [10.0, 20.0]


def test_conflicting_series_duplicates_are_rejected():
    series = pd.Series([10.0, 12.0, 20.0], index=_duplicate_index(), name="price")

    with pytest.raises(
        ValueError,
        match=r"actuals contains conflicting duplicate values.*column 'price'",
    ):
        collapse_identical_duplicates(series, "actuals")


def test_identical_bm_composite_key_duplicates_collapse_before_latest_selection():
    records = pd.DataFrame(
        {
            "startTime": [
                "2025-01-01 00:00Z",
                "2025-01-01 00:00Z",
                "2025-01-01 00:00Z",
            ],
            "createdDateTime": [
                "2025-01-01 00:30Z",
                "2025-01-01 00:30Z",
                "2025-01-01 01:00Z",
            ],
            "systemSellPrice": [40.0, 40.0, 55.0],
            "source_file": ["a", "b", "c"],
        }
    )

    collapsed = collapse_bm_vintage_duplicates(
        records,
        "BM input",
        comparison_columns=["systemSellPrice"],
    )
    latest = select_latest_bm_vintages(
        records,
        "BM input",
        comparison_columns=["systemSellPrice"],
    )

    assert len(collapsed) == 2
    assert collapsed.iloc[0]["source_file"] == "a"
    assert latest.iloc[0]["systemSellPrice"] == 55.0
    assert latest.iloc[0]["createdDateTime"] == pd.Timestamp("2025-01-01 01:00Z")


def test_conflicting_bm_composite_key_duplicate_is_rejected():
    records = pd.DataFrame(
        {
            "startTime": ["2025-01-01 00:00Z", "2025-01-01 00:00Z"],
            "createdDateTime": ["2025-01-01 01:00Z", "2025-01-01 01:00Z"],
            "systemSellPrice": [55.0, 65.0],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            r"conflicting duplicate BM vintage.*settlement 2025-01-01 00:00:00"
            r"\+00:00.*publication 2025-01-01 01:00:00\+00:00"
            r".*column 'systemSellPrice'"
        ),
    ):
        select_latest_bm_vintages(
            records,
            "BM input",
            comparison_columns=["systemSellPrice"],
        )


def test_quantile_forecast_rejects_crossing_with_timestamp_and_values():
    index = pd.date_range("2025-01-01", periods=2, freq="30min", tz="UTC")
    forecast = pd.DataFrame(
        {"q10": [0.0, 3.0], "q50": [1.0, 2.0], "q90": [2.0, 4.0]},
        index=index,
    )

    with pytest.raises(
        ValueError,
        match=r"crossed quantiles at 2025-01-01 00:30:00\+00:00: q10=3.0, q50=2.0",
    ):
        validate_quantile_forecast(forecast, "DA forecast")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_quantile_forecast_rejects_non_finite_values(invalid: float):
    index = pd.date_range("2025-01-01", periods=1, freq="30min", tz="UTC")
    forecast = pd.DataFrame(
        {"q10": [0.0], "q50": [invalid], "q90": [2.0]},
        index=index,
    )

    with pytest.raises(ValueError, match="non-finite q50.*2025-01-01"):
        validate_quantile_forecast(forecast, "BM forecast")
