"""Shared walk-forward quantile-forecasting framework."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import settings
from market_calendar import aggregate_to_efa_blocks, realised_data_cutoff_time

logger = logging.getLogger(__name__)

TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"
RANDOM_STATE = 42
QUANTILES = (0.1, 0.5, 0.9)

LGBM_SHARED_PARAMS = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": RANDOM_STATE,
}
CV_N_ESTIMATORS = 500

BASE_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "year_sin",
    "year_cos",
    "accepted_offer_volume_lag_48",
    "accepted_offer_volume_lag_96",
    "accepted_offer_volume_lag_144",
    "accepted_offer_volume_lag_336",
    "price_lag_48",
    "price_lag_96",
    "price_lag_144",
    "price_lag_336",
    "accepted_offer_volume_roll_mean_8",
    "accepted_offer_volume_roll_std_8",
    "price_roll_mean_8",
    "price_roll_std_8",
    "accepted_offer_volume_roll_mean_48",
    "accepted_offer_volume_roll_std_48",
    "price_roll_mean_48",
    "price_roll_std_48",
    "accepted_offer_volume_roll_mean_336",
    "accepted_offer_volume_roll_std_336",
    "price_roll_mean_336",
    "price_roll_std_336",
    "wind_speed",
    "cloud_cover",
    "bm_imbalance_volume_lag_48",
]


class BaseForecaster:
    model_id: str
    target_col: str
    feature_cols: list[str]

    MIN_TRAIN = 48 * 28
    FOLD_STEP = 48 * 7
    VAL_WINDOW = 48 * 7
    MIN_FOLD_ROWS = 48 * 7

    # These columns are fixed delivery-day snapshots rather than ordinary shifts.
    GATE_ORIGIN_COLS: frozenset[str] = frozenset(
        {
            "accepted_offer_volume_lag_48",
            "bm_price_lag_48",
            "bm_imbalance_volume_lag_48",
            "wind_change_lag_48",
            "spread_lag_48",
        }
    )

    def load_features(self) -> pd.DataFrame:
        return pd.read_parquet(settings.data_dir / "features" / "features.parquet")

    @staticmethod
    def _pinball(actuals: np.ndarray, predictions: np.ndarray, q: float) -> float:
        errors = actuals - predictions
        return float(np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors)))

    @staticmethod
    def evaluate_forecast_metrics(
        actuals: np.ndarray,
        q10: np.ndarray,
        q50: np.ndarray,
        q90: np.ndarray,
    ) -> dict[str, float]:
        arrays = [np.asarray(values, dtype=float) for values in (actuals, q10, q50, q90)]
        if not all(values.shape == arrays[0].shape for values in arrays):
            raise ValueError("Forecast metric arrays must have identical shapes")

        finite = np.logical_and.reduce([np.isfinite(values) for values in arrays])
        if not finite.any():
            raise ValueError("No finite observations are available for forecast metrics")

        actual, lower, median, upper = [values[finite] for values in arrays]
        if np.any(lower > median) or np.any(median > upper):
            raise ValueError("Quantiles are crossed in forecast metric input")

        mae = float(np.mean(np.abs(actual - median)))
        pb10 = BaseForecaster._pinball(actual, lower, 0.1)
        pb90 = BaseForecaster._pinball(actual, upper, 0.9)
        coverage = float(np.mean((actual >= lower) & (actual <= upper)) * 100.0)

        alpha = 0.2
        width = upper - lower
        penalty_low = np.where(actual < lower, (2.0 / alpha) * (lower - actual), 0.0)
        penalty_high = np.where(actual > upper, (2.0 / alpha) * (actual - upper), 0.0)
        winkler = float(np.mean(width + penalty_low + penalty_high))

        return {
            "mae": round(mae, 3),
            "pinball_q10": round(pb10, 3),
            "pinball_q90": round(pb90, 3),
            "coverage_pct": round(coverage, 1),
            "winkler_score": round(winkler, 3),
        }

    def _report_crossing_rate(self, df: pd.DataFrame, context: str) -> pd.DataFrame:
        if df.empty:
            return df

        required = ["q10", "q50", "q90"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing quantile columns: {missing}")

        crossings = (df["q10"] > df["q50"]) | (df["q50"] > df["q90"])
        n_crossings = int(crossings.sum())
        rate = n_crossings / len(df)
        logger.info(
            "%s [%s] quantile crossing rate: %.2f%% (%d of %d)",
            self.model_id,
            context,
            rate * 100.0,
            n_crossings,
            len(df),
        )

        if n_crossings == 0:
            return df

        corrected = df.copy()
        corrected[required] = np.sort(corrected[required].to_numpy(dtype=float), axis=1)
        return corrected

    @staticmethod
    def _check_lag_alignment(
        df: pd.DataFrame,
        target: str,
        lag_col: str,
        lag_n: int,
    ) -> None:
        if lag_col not in df.columns or target not in df.columns:
            return

        residual = (df[lag_col] - df[target].shift(lag_n)).dropna()
        if residual.empty:
            logger.warning("No rows were available for lag check %s", lag_col)
            return

        maximum = float(residual.abs().max())
        if maximum > 1e-6:
            raise ValueError(
                f"Lag alignment error: {lag_col!r} does not equal "
                f"{target!r}.shift({lag_n}). Maximum residual: {maximum:.6f}"
            )

    def _fit_quantile(
        self,
        q: float,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[lgb.LGBMRegressor, np.ndarray, int]:
        model = lgb.LGBMRegressor(
            **LGBM_SHARED_PARAMS,
            alpha=q,
            n_estimators=CV_N_ESTIMATORS,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        best_iteration = int(model.best_iteration_ or CV_N_ESTIMATORS)
        return model, model.predict(x_val), best_iteration

    def train(self, df: pd.DataFrame) -> tuple[dict[float, lgb.LGBMRegressor], pd.DataFrame]:
        """Run weekly-refit expanding-window CV and train final quantile models."""
        logger.info("Starting walk-forward CV for %s", self.model_id)
        columns_needed = self.feature_cols + [self.target_col]

        fold_predictions: dict[float, list[float]] = {q: [] for q in QUANTILES}
        fold_actuals: list[float] = []
        fold_ids: list[int] = []
        oof_predictions: dict[float, list[float]] = {q: [] for q in QUANTILES}
        oof_timestamps: list[pd.Timestamp] = []
        iterations: dict[float, list[int]] = {q: [] for q in QUANTILES}

        fold_ends = range(
            self.MIN_TRAIN,
            len(df) - self.VAL_WINDOW + 1,
            self.FOLD_STEP,
        )
        total_folds = len(fold_ends)
        folds_run = 0
        folds_skipped = 0
        started = time.time()

        for fold_number, fold_end in enumerate(fold_ends):
            first_validation_timestamp = df.index[fold_end]
            information_cutoff = realised_data_cutoff_time(first_validation_timestamp)

            train_slice = df[df.index < information_cutoff].dropna(
                subset=columns_needed
            )
            validation_slice = df.iloc[
                fold_end : fold_end + self.VAL_WINDOW
            ].dropna(subset=columns_needed)

            if len(train_slice) < self.MIN_FOLD_ROWS or validation_slice.empty:
                folds_skipped += 1
                continue

            x_train = train_slice[self.feature_cols]
            y_train = train_slice[self.target_col]
            x_val = validation_slice[self.feature_cols]
            y_val = validation_slice[self.target_col]

            for q in QUANTILES:
                _, predictions, best_iteration = self._fit_quantile(
                    q,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                )
                fold_predictions[q].extend(predictions.tolist())
                oof_predictions[q].extend(predictions.tolist())
                iterations[q].append(best_iteration)

            fold_actuals.extend(y_val.to_list())
            fold_ids.extend([fold_number] * len(y_val))
            oof_timestamps.extend(validation_slice.index.to_list())
            folds_run += 1

            if fold_number and fold_number % 10 == 0:
                elapsed = time.time() - started
                remaining = elapsed / fold_number * (total_folds - fold_number)
                logger.info(
                    "Fold %d/%d, train=%d, validation=%d, estimated remaining %.0fs",
                    fold_number,
                    total_folds,
                    len(train_slice),
                    len(validation_slice),
                    remaining,
                )

        logger.info("Folds run=%d, skipped=%d", folds_run, folds_skipped)
        if not fold_actuals:
            raise RuntimeError("No validation observations were collected")

        actual_array = np.asarray(fold_actuals, dtype=float)
        median_array = np.asarray(fold_predictions[0.5], dtype=float)
        logger.info(
            "CV metrics for %s: MAE=%.3f, pinball q10=%.3f, pinball q90=%.3f",
            self.model_id,
            np.mean(np.abs(actual_array - median_array)),
            self._pinball(actual_array, np.asarray(fold_predictions[0.1]), 0.1),
            self._pinball(actual_array, np.asarray(fold_predictions[0.9]), 0.9),
        )

        oof_df = pd.DataFrame(
            {
                "q10": oof_predictions[0.1],
                "q50": oof_predictions[0.5],
                "q90": oof_predictions[0.9],
            },
            index=pd.DatetimeIndex(oof_timestamps, name=df.index.name),
        ).sort_index()

        if oof_df.index.has_duplicates:
            raise RuntimeError("OOF forecast timestamps overlap between folds")
        oof_df = self._report_crossing_rate(oof_df, "OOF")

        best_n_estimators: dict[float, int] = {}
        for q in QUANTILES:
            median_iteration = int(np.median(iterations[q])) if iterations[q] else 200
            best_n_estimators[q] = max(50, median_iteration)
            logger.info(
                "%s q%d final estimator count=%d",
                self.model_id,
                int(q * 100),
                best_n_estimators[q],
            )

        clean = df.dropna(subset=columns_needed)
        x_full = clean[self.feature_cols]
        y_full = clean[self.target_col]
        if x_full.empty:
            raise RuntimeError("No complete training rows are available")

        final_models: dict[float, lgb.LGBMRegressor] = {}
        for q in QUANTILES:
            model = lgb.LGBMRegressor(
                **LGBM_SHARED_PARAMS,
                alpha=q,
                n_estimators=best_n_estimators[q],
            )
            model.fit(x_full, y_full)
            final_models[q] = model

        return final_models, oof_df

    def save_models(self, models: dict[float, lgb.LGBMRegressor]) -> dict[float, Path]:
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[float, Path] = {}
        for quantile, model in models.items():
            path = settings.models_dir / f"{self.model_id}_q{int(quantile * 100)}.txt"
            model.booster_.save_model(str(path))
            paths[quantile] = path
        return paths

    def predict(self, models: dict[float, lgb.LGBMRegressor], df: pd.DataFrame) -> pd.DataFrame:
        forecasts = pd.DataFrame(index=df.index)
        for quantile, model in models.items():
            forecasts[f"q{int(quantile * 100)}"] = model.predict(df[self.feature_cols])
        return self._report_crossing_rate(forecasts, "OOS")

    def save_oof_forecasts(self, forecasts: pd.DataFrame) -> None:
        if forecasts.empty:
            raise RuntimeError(f"No OOF forecasts were generated for {self.model_id}")
        directory = settings.data_dir / "forecasts"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.model_id}_oof_2022_2024.parquet"
        forecasts[["q10", "q50", "q90"]].to_parquet(path)
        logger.info("OOF forecasts written to %s", path)

    def save_oos_forecasts(self, forecasts: pd.DataFrame) -> None:
        if forecasts.empty:
            raise RuntimeError(f"No OOS forecasts were generated for {self.model_id}")
        directory = settings.data_dir / "forecasts"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.model_id}_oos_2025.parquet"
        forecasts[["q10", "q50", "q90"]].to_parquet(path)
        logger.info("OOS forecasts written to %s", path)

    def _evaluate_oos(
        self,
        target: pd.Series,
        forecasts: pd.DataFrame,
    ) -> None:
        common = target.dropna().index.intersection(forecasts.dropna().index)
        if common.empty:
            logger.warning("No OOS target rows are available for %s", self.model_id)
            return

        actual = target.reindex(common)
        predicted = forecasts.reindex(common)

        if self.model_id in {"dc_low", "dc_high"}:
            actual = aggregate_to_efa_blocks(actual, how="mean")
            predicted = aggregate_to_efa_blocks(predicted, how="mean")
            common_blocks = actual.index.intersection(predicted.dropna().index)
            actual = actual.reindex(common_blocks)
            predicted = predicted.reindex(common_blocks)

        metrics = self.evaluate_forecast_metrics(
            actual.to_numpy(dtype=float),
            predicted["q10"].to_numpy(dtype=float),
            predicted["q50"].to_numpy(dtype=float),
            predicted["q90"].to_numpy(dtype=float),
        )
        logger.info("OOS metrics for %s: %s", self.model_id, metrics)

    def run(self) -> None:
        logger.info("Training %s with target %s", self.model_id, self.target_col)
        df = self.load_features().sort_index()
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            raise ValueError("Feature table must use a timezone-aware DatetimeIndex")

        columns_needed = self.feature_cols + [self.target_col]
        missing = [column for column in columns_needed if column not in df.columns]
        if missing:
            raise ValueError(f"Feature table is missing columns: {missing}")

        for lag_column in [
            column for column in self.feature_cols if re.search(r"_lag_\d+$", column)
        ]:
            match = re.search(r"_lag_(\d+)$", lag_column)
            assert match is not None
            if lag_column in self.GATE_ORIGIN_COLS:
                continue
            source_column = lag_column[: match.start()]
            if source_column in df.columns:
                self._check_lag_alignment(
                    df,
                    target=source_column,
                    lag_col=lag_column,
                    lag_n=int(match.group(1)),
                )

        oos_start = pd.Timestamp(TEST_START_DATE, tz="UTC")
        training_cutoff = realised_data_cutoff_time(oos_start)
        train_validation = df[df.index < training_cutoff].copy()
        test = df[df.index >= oos_start].copy()

        logger.info(
            "Split for %s: train rows=%d before %s, OOS rows=%d",
            self.model_id,
            len(train_validation),
            training_cutoff,
            len(test),
        )

        models, oof = self.train(train_validation)
        self.save_models(models)
        self.save_oof_forecasts(oof)

        test_prediction_rows = test.dropna(subset=self.feature_cols)
        if test_prediction_rows.empty:
            raise RuntimeError(f"No valid OOS feature rows for {self.model_id}")

        forecasts = self.predict(models, test_prediction_rows)
        self.save_oos_forecasts(forecasts)
        self._evaluate_oos(test[self.target_col], forecasts)
