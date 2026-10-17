import numpy as np
import pandas as pd
import pytest

from optimiser import scenarios
from optimiser.scenarios import (
    BM_TARGET_AVAILABILITY_COL,
    _estimate_correlation_matrix,
    _residual_history_available_by,
)


def residual_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    values = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "da": values,
            "bm": values + 1.0,
            "dc_low": values + 2.0,
            "dc_high": values + 3.0,
            BM_TARGET_AVAILABILITY_COL: index + pd.Timedelta(minutes=20),
        },
        index=index,
    )


def test_residual_history_requires_publication_by_cutoff() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    residuals = residual_frame(index)
    residuals.loc[index[1], BM_TARGET_AVAILABILITY_COL] = pd.Timestamp(
        "2024-01-01 03:00Z"
    )

    history = _residual_history_available_by(residuals, "2024-01-01 02:00Z")

    assert history.index.equals(index[[0, 2, 3]])
    assert BM_TARGET_AVAILABILITY_COL not in history.columns


def test_residual_history_requires_publication_metadata() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    residuals = residual_frame(index).drop(columns=BM_TARGET_AVAILABILITY_COL)

    with pytest.raises(ValueError, match="missing 'bm_target_created_time'"):
        _residual_history_available_by(residuals, "2024-01-02")


def test_residual_history_rejects_invalid_publication_time() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    residuals = residual_frame(index)
    residuals.loc[index[0], BM_TARGET_AVAILABILITY_COL] = pd.NaT

    with pytest.raises(ValueError, match="no valid BM publication time"):
        _residual_history_available_by(residuals, "2024-01-02")


def test_correlation_estimation_excludes_unpublished_residuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2024-01-01", periods=600, freq="30min", tz="UTC")
    signal = np.sin(np.arange(600, dtype=float) / 13.0) + np.arange(600) / 1000.0
    residuals = pd.DataFrame(
        {
            "da": signal,
            "bm": signal.copy(),
            "dc_low": np.cos(np.arange(600, dtype=float) / 11.0),
            "dc_high": np.sin(np.arange(600, dtype=float) / 17.0),
            BM_TARGET_AVAILABILITY_COL: index + pd.Timedelta(minutes=20),
        },
        index=index,
    )
    residuals.loc[index[500:], "bm"] *= -1.0
    cutoff = index[-1] + pd.Timedelta(hours=1)
    residuals.loc[index[500:], BM_TARGET_AVAILABILITY_COL] = cutoff + pd.Timedelta(days=1)
    monkeypatch.setattr(
        scenarios,
        "_historical_residual_frame",
        lambda _use_spread, _da_model_id: residuals,
    )

    correlation = _estimate_correlation_matrix(cutoff_date=cutoff)

    assert correlation.shape == (4, 4)
    assert correlation[0, 1] > 0.999
