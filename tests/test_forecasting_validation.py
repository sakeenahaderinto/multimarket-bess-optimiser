import numpy as np
import pandas as pd
import pytest

from forecasting import base
from forecasting.base import BaseForecaster


class SmallForecaster(BaseForecaster):
    model_id = "test"
    target_col = "target"
    feature_cols = ["feature"]

    VAL_WINDOW = 2
    MIN_FOLD_ROWS = 3


def frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": np.arange(len(index), dtype=float),
            "target": np.arange(len(index), dtype=float) + 10.0,
        },
        index=index,
    )


def test_inner_validation_is_chronological_and_disjoint() -> None:
    forecaster = SmallForecaster()
    history = frame(pd.date_range("2024-01-01", periods=7, freq="30min", tz="UTC"))

    split = forecaster._split_inner_validation(history)

    assert split is not None
    inner_train, inner_validation = split
    assert inner_train.index.equals(history.index[:5])
    assert inner_validation.index.equals(history.index[5:])
    assert inner_train.index.max() < inner_validation.index.min()
    assert inner_train.index.intersection(inner_validation.index).empty


def test_inner_validation_requires_enough_fit_and_tuning_rows() -> None:
    forecaster = SmallForecaster()
    too_short = frame(pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC"))

    assert forecaster._split_inner_validation(too_short) is None


def test_inner_validation_rejects_unsorted_history() -> None:
    forecaster = SmallForecaster()
    history = frame(pd.date_range("2024-01-01", periods=5, freq="30min", tz="UTC"))

    with pytest.raises(ValueError, match="sorted chronologically"):
        forecaster._split_inner_validation(history.iloc[::-1])


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("INNER_VAL_WINDOW", 0),
        ("INNER_VAL_WINDOW", -1),
        ("MIN_INNER_TRAIN_ROWS", 0),
        ("MIN_INNER_TRAIN_ROWS", -1),
    ],
)
def test_inner_validation_rejects_non_positive_window(attribute: str, value: int) -> None:
    forecaster = SmallForecaster()
    setattr(forecaster, attribute, value)
    history = frame(pd.date_range("2024-01-01", periods=7, freq="30min", tz="UTC"))

    with pytest.raises(ValueError, match="must be positive"):
        forecaster._split_inner_validation(history)


def test_early_stopping_never_sees_outer_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        instances: list["FakeModel"] = []

        def __init__(self, **params: object) -> None:
            self.params = params
            self.best_iteration_ = 17
            self.fit_x: pd.DataFrame | None = None
            self.eval_x: pd.DataFrame | None = None
            self.predict_x: pd.DataFrame | None = None
            self.instances.append(self)

        def fit(
            self,
            x: pd.DataFrame,
            _y: pd.Series,
            *,
            eval_X: pd.DataFrame | None = None,
            eval_y: pd.Series | None = None,
            callbacks: list[object] | None = None,
        ) -> "FakeModel":
            self.fit_x = x
            self.eval_x = eval_X
            return self

        def predict(self, x: pd.DataFrame) -> np.ndarray:
            self.predict_x = x
            return np.full(len(x), 123.0)

    monkeypatch.setattr(base.lgb, "LGBMRegressor", FakeModel)
    monkeypatch.setattr(base.lgb, "early_stopping", lambda **_kwargs: object())

    all_rows = frame(pd.date_range("2024-01-01", periods=9, freq="30min", tz="UTC"))
    inner_train = all_rows.iloc[:4]
    inner_validation = all_rows.iloc[4:6]
    pre_origin_refit = all_rows.iloc[:6]
    outer_validation = all_rows.iloc[6:]

    model, predictions, best_iteration = SmallForecaster()._fit_quantile(
        0.5,
        inner_train[["feature"]],
        inner_train["target"],
        inner_validation[["feature"]],
        inner_validation["target"],
        pre_origin_refit[["feature"]],
        pre_origin_refit["target"],
        outer_validation[["feature"]],
    )

    selection_model, refit_model = FakeModel.instances
    assert selection_model.fit_x is not None
    assert selection_model.fit_x.index.equals(inner_train.index)
    assert selection_model.eval_x is not None
    assert selection_model.eval_x.index.equals(inner_validation.index)
    assert selection_model.eval_x.index.intersection(outer_validation.index).empty

    assert model is refit_model
    assert refit_model.params["n_estimators"] == 17
    assert refit_model.fit_x is not None
    assert refit_model.fit_x.index.equals(pre_origin_refit.index)
    assert refit_model.eval_x is None
    assert refit_model.predict_x is not None
    assert refit_model.predict_x.index.equals(outer_validation.index)
    assert best_iteration == 17
    np.testing.assert_array_equal(predictions, np.full(3, 123.0))
