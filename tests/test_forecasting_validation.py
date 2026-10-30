import numpy as np
import pandas as pd
import pytest

from forecasting import base
from forecasting.base import BaseForecaster
from forecasting.bm_da_spread import BMDASpreadForecaster
from forecasting.bm_price import BMPriceForecaster
from forecasting.dc_tender import DCHighForecaster, DCLowForecaster
from market_calendar import efa_block_keys


class SmallForecaster(BaseForecaster):
    model_id = "test"
    target_col = "target"
    feature_cols = ["feature"]

    VAL_WINDOW = 2
    MIN_FOLD_ROWS = 3


class PublishedTargetForecaster(SmallForecaster):
    TARGET_AVAILABILITY_COL = "target_created_time"


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


@pytest.mark.parametrize("forecaster", [DCLowForecaster(), DCHighForecaster()])
def test_dc_inner_validation_keeps_efa_blocks_indivisible(forecaster) -> None:
    forecaster.INNER_VAL_WINDOW = 5
    forecaster.MIN_INNER_TRAIN_ROWS = 3
    history = frame(pd.date_range("2024-01-01", periods=30, freq="30min", tz="UTC"))

    split = forecaster._split_inner_validation(history)

    assert split is not None
    inner_train, inner_validation = split
    assert len(inner_validation) >= 5
    assert not set(efa_block_keys(inner_train.index)).intersection(
        efa_block_keys(inner_validation.index)
    )


def test_dc_outer_validation_rejects_shared_efa_block() -> None:
    forecaster = DCLowForecaster()
    index = pd.date_range("2024-01-01 07:00", periods=8, freq="30min", tz="UTC")

    with pytest.raises(ValueError, match="share validation group"):
        forecaster._require_disjoint_validation_groups(
            index[:4],
            index[4:],
            "outer training and validation",
        )


def test_dc_sample_weights_give_each_efa_block_equal_total_weight() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-01-01 07:00Z",
            "2024-01-01 07:30Z",
            "2024-01-01 08:00Z",
            "2024-01-01 11:00Z",
            "2024-01-01 11:30Z",
        ]
    )
    weights = DCLowForecaster()._sample_weight(index)
    groups = efa_block_keys(index)
    totals = pd.Series(weights).groupby(groups).sum()

    assert weights.mean() == pytest.approx(1.0)
    assert totals.nunique() == 1


def test_dc_fit_passes_group_weights_to_lightgbm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WeightedFakeModel:
        instances: list["WeightedFakeModel"] = []

        def __init__(self, **_params: object) -> None:
            self.best_iteration_ = 9
            self.sample_weight = None
            self.eval_sample_weight = None
            self.instances.append(self)

        def fit(
            self,
            _x: pd.DataFrame,
            _y: pd.Series,
            *,
            sample_weight=None,
            eval_X=None,
            eval_y=None,
            eval_sample_weight=None,
            callbacks=None,
        ) -> "WeightedFakeModel":
            del eval_X, eval_y, callbacks
            self.sample_weight = sample_weight
            self.eval_sample_weight = eval_sample_weight
            return self

        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(x))

    monkeypatch.setattr(base.lgb, "LGBMRegressor", WeightedFakeModel)
    monkeypatch.setattr(base.lgb, "early_stopping", lambda **_kwargs: object())
    rows = frame(pd.date_range("2024-01-01 03:00", periods=24, freq="30min", tz="UTC"))
    forecaster = DCLowForecaster()

    forecaster._fit_quantile(
        0.5,
        rows.iloc[:8][["feature"]],
        rows.iloc[:8]["target"],
        rows.iloc[8:16][["feature"]],
        rows.iloc[8:16]["target"],
        rows.iloc[:16][["feature"]],
        rows.iloc[:16]["target"],
        rows.iloc[16:][["feature"]],
    )

    selection, refit = WeightedFakeModel.instances
    assert selection.sample_weight is not None
    assert selection.eval_sample_weight is not None
    assert len(selection.eval_sample_weight) == 1
    assert refit.sample_weight is not None


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


def test_training_rows_require_target_publication_by_cutoff() -> None:
    forecaster = PublishedTargetForecaster()
    history = frame(pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC"))
    history["target_created_time"] = pd.to_datetime(
        [
            "2024-01-01 00:20Z",
            "2024-01-01 00:50Z",
            "2024-01-01 02:01Z",
            "2024-01-01 02:30Z",
        ]
    )

    available = forecaster._training_rows_available_by(history, "2024-01-01 02:00Z")

    assert available.index.equals(history.index[:2])
    assert (available["target_created_time"] <= pd.Timestamp("2024-01-01 02:00Z")).all()


def test_training_rows_require_target_availability_metadata() -> None:
    forecaster = PublishedTargetForecaster()
    history = frame(pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC"))

    with pytest.raises(ValueError, match="availability column"):
        forecaster._training_rows_available_by(history, "2024-01-02")


def test_training_rows_reject_missing_publication_for_present_target() -> None:
    forecaster = PublishedTargetForecaster()
    history = frame(pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC"))
    history["target_created_time"] = [pd.NaT, pd.Timestamp("2024-01-01 01:00Z")]

    with pytest.raises(ValueError, match="no valid publication time"):
        forecaster._training_rows_available_by(history, "2024-01-02")


def test_training_rows_reject_publication_before_settlement() -> None:
    forecaster = PublishedTargetForecaster()
    history = frame(pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC"))
    history["target_created_time"] = pd.to_datetime(
        ["2023-12-31 23:59Z", "2024-01-01 01:00Z"]
    )

    with pytest.raises(ValueError, match="precedes settlement start"):
        forecaster._training_rows_available_by(history, "2024-01-02")


def test_published_targets_require_final_training_cutoff() -> None:
    forecaster = PublishedTargetForecaster()
    history = frame(pd.date_range("2024-01-01", periods=7, freq="30min", tz="UTC"))

    with pytest.raises(ValueError, match="final_information_cutoff is required"):
        forecaster.train(history)


def test_bm_and_spread_models_use_bm_target_publication_time() -> None:
    assert BMPriceForecaster.TARGET_AVAILABILITY_COL == "bm_target_created_time"
    assert BMDASpreadForecaster.TARGET_AVAILABILITY_COL == "bm_target_created_time"


def test_walk_forward_training_filters_targets_at_each_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingForecaster(PublishedTargetForecaster):
        MIN_TRAIN = 6
        FOLD_STEP = 2
        VAL_WINDOW = 2
        MIN_FOLD_ROWS = 2
        refit_indexes: list[pd.DatetimeIndex] = []

        def _fit_quantile(
            self,
            q: float,
            x_inner_train: pd.DataFrame,
            y_inner_train: pd.Series,
            x_inner_validation: pd.DataFrame,
            y_inner_validation: pd.Series,
            x_refit: pd.DataFrame,
            y_refit: pd.Series,
            x_outer_validation: pd.DataFrame,
        ) -> tuple[object, np.ndarray, int]:
            del (
                x_inner_train,
                y_inner_train,
                x_inner_validation,
                y_inner_validation,
                y_refit,
            )
            self.refit_indexes.append(x_refit.index)
            return object(), np.full(len(x_outer_validation), q), 10

    class FakeFinalModel:
        def __init__(self, **_params: object) -> None:
            pass

        def fit(self, _x: pd.DataFrame, _y: pd.Series) -> "FakeFinalModel":
            return self

    monkeypatch.setattr(
        base,
        "realised_data_cutoff_time",
        lambda timestamp: pd.Timestamp(timestamp) - pd.Timedelta(minutes=1),
    )
    monkeypatch.setattr(base.lgb, "LGBMRegressor", FakeFinalModel)

    history = frame(pd.date_range("2024-01-01", periods=12, freq="30min", tz="UTC"))
    history["target_created_time"] = history.index + pd.Timedelta(minutes=5)
    delayed_timestamp = history.index[4]
    history.loc[delayed_timestamp, "target_created_time"] = history.index[7]

    RecordingForecaster().train(
        history,
        final_information_cutoff=history.index[-1] + pd.Timedelta(hours=1),
    )

    first_fold_refits = RecordingForecaster.refit_indexes[:3]
    second_fold_refits = RecordingForecaster.refit_indexes[3:6]
    assert all(delayed_timestamp not in index for index in first_fold_refits)
    assert all(delayed_timestamp in index for index in second_fold_refits)


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
