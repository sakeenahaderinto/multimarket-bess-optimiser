import pandas as pd
import pytest

from data_integrity import collapse_identical_duplicates


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
