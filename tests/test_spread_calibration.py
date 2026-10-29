import numpy as np
import pandas as pd
import pytest

import calibrate_spread
from calibrate_spread import (
    _aligned_calibration_history,
    apply_scales,
    calibration_history_available_by,
    rolling_calibrate,
)


def forecasts(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "q10": np.zeros(len(index)),
            "q50": np.ones(len(index)),
            "q90": np.full(len(index), 2.0),
        },
        index=index,
    )


def test_rolling_calibration_uses_only_outcomes_published_by_each_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.DatetimeIndex(
        [
            "2024-01-01 00:00Z",
            "2024-01-01 12:00Z",
            "2024-01-02 00:00Z",
            "2024-01-03 00:00Z",
        ]
    )
    actual = pd.Series([1.0, 1.0, 1.0, 1.0], index=index)
    availability = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01 01:00Z",
                "2024-01-01 13:00Z",
                "2024-01-02 01:00Z",
                "2024-01-03 01:00Z",
            ]
        ),
        index=index,
    )
    history_sizes: list[int] = []

    def fake_scales(
        historical_actual: np.ndarray,
        _q10: np.ndarray,
        _q50: np.ndarray,
        _q90: np.ndarray,
    ) -> tuple[float, float]:
        history_sizes.append(len(historical_actual))
        return float(len(historical_actual)), float(len(historical_actual))

    monkeypatch.setattr(calibrate_spread, "find_tail_scales", fake_scales)

    _, scales = rolling_calibrate(
        forecasts(index),
        actual,
        availability,
        min_history_days=1,
    )

    assert history_sizes == [1, 3]
    assert scales.loc[pd.Timestamp("2024-01-02").date(), "history_observations"] == 1
    assert scales.loc[pd.Timestamp("2024-01-03").date(), "history_observations"] == 3


def test_fixed_calibration_history_excludes_unpublished_outcome() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
    actual = pd.Series([1.0, 2.0, 3.0], index=index)
    availability = pd.Series(
        pd.to_datetime(
            ["2024-01-01 00:20Z", "2024-01-01 03:00Z", "2024-01-01 01:20Z"]
        ),
        index=index,
    )

    history = calibration_history_available_by(
        forecasts(index),
        actual,
        availability,
        "2024-01-01 02:00Z",
    )

    assert history.index.equals(index[[0, 2]])


def test_calibration_rejects_publication_before_settlement() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    actual = pd.Series([1.0, 2.0], index=index)
    availability = pd.Series(
        pd.to_datetime(["2023-12-31 23:59Z", "2024-01-01 01:00Z"]),
        index=index,
    )

    with pytest.raises(ValueError, match="precedes settlement start"):
        _aligned_calibration_history(forecasts(index), actual, availability)


@pytest.mark.parametrize("invalid_scale", [-1.0, float("nan"), float("inf")])
def test_apply_scales_rejects_invalid_scale(invalid_scale: float) -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")

    with pytest.raises(ValueError, match="lower_scale must be finite and non-negative"):
        apply_scales(forecasts(index), invalid_scale, 1.0)
